from typing import Optional, Tuple
import torch
import torch.nn as nn


class DecisionTransformer(nn.Module):

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_timestep: int = 4096,
        action_tanh: bool = True,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.d_model = int(d_model)
        self.action_tanh = bool(action_tanh)

        # modality embeddings
        self.embed_s = nn.Linear(self.state_dim, d_model)
        self.embed_a = nn.Linear(self.action_dim, d_model)
        self.embed_r = nn.Linear(1, d_model)  # RTG as 1-dim per step

        # per-timestep embedding (shared by 3 tokens of same timestep)
        self.time_embed = nn.Embedding(max_timestep, d_model)

        # type embedding: to disambiguate [R,S,A]
        self.type_embed = nn.Embedding(3, d_model)  # 0=R,1=S,2=A

        self.ln_in = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # action head
        self.pred_a = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.action_dim),
        )

        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def _causal_mask(L: int, device: torch.device):
        # allow attend to <= position (causal)
        mask = torch.full((L, L), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    @staticmethod
    def _interleave(r_tok, s_tok, a_tok):
        # (B,K,D) -> (B,3K,D) as [R1,S1,A1,R2,S2,A2,...]
        B, K, D = r_tok.shape
        out = torch.stack([r_tok, s_tok, a_tok], dim=2).reshape(B, 3*K, D)
        return out

    @staticmethod
    def _action_positions(K: int, device: torch.device):
        # indices 2,5,8,... in [R1,S1,A1,R2,S2,A2,...]
        return torch.arange(K, device=device) * 3 + 2

    def forward(
        self,
        states: torch.Tensor,        # (B,K,state_dim)
        actions: torch.Tensor,       # (B,K,action_dim)
        returns_to_go: torch.Tensor, # (B,K,1)
        timesteps: torch.Tensor,     # (B,K) int64
        attn_mask_pad: Optional[torch.Tensor] = None,  # (B,3K) 1=keep, 0=pad
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, K, _ = states.shape
        device = states.device

        r_tok = self.embed_r(returns_to_go)  # (B,K,D)
        s_tok = self.embed_s(states)         # (B,K,D)
        a_tok = self.embed_a(actions)        # (B,K,D)

        # per-timestep embedding
        t_emb = self.time_embed(timesteps.clamp(min=0))  # (B,K,D)
        r_tok = r_tok + t_emb
        s_tok = s_tok + t_emb
        a_tok = a_tok + t_emb

        # type embedding
        type_r = self.type_embed.weight[0].view(1,1,-1)
        type_s = self.type_embed.weight[1].view(1,1,-1)
        type_a = self.type_embed.weight[2].view(1,1,-1)
        r_tok = r_tok + type_r
        s_tok = s_tok + type_s
        a_tok = a_tok + type_a

        tokens = self._interleave(r_tok, s_tok, a_tok)   # (B,3K,D)
        tokens = self.ln_in(tokens)

        # causal attention mask
        causal_mask = self._causal_mask(tokens.size(1), device=device)

        key_padding_mask = None
        if attn_mask_pad is not None:
            key_padding_mask = ~(attn_mask_pad.bool())  # Transformer expects True at PAD

        h = self.transformer(tokens, mask=causal_mask, src_key_padding_mask=key_padding_mask)  # (B,3K,D)

        a_pos = self._action_positions(K, device=device)
        a_hidden = h[:, a_pos, :]                  # (B,K,D)
        a_pred = self.pred_a(a_hidden)             # (B,K,action_dim)
        if self.action_tanh:
            a_pred = torch.tanh(a_pred)            # map to [-1,1]
        return a_pred, a_hidden

    @torch.no_grad()
    def predict_next_action(
        self,
        states: torch.Tensor, actions: torch.Tensor, returns_to_go: torch.Tensor, timesteps: torch.Tensor,
        max_action: float = 1.0,
    ) -> torch.Tensor:
        self.eval()
        a_pred, _ = self.forward(states, actions, returns_to_go, timesteps)
        next_a = a_pred[:, -1, :]
        if self.action_tanh:
            next_a = next_a.clamp(-1.0, 1.0) * float(max_action)
        return next_a
