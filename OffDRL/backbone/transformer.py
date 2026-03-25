import math
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
from torch.nn import functional as F


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


def soft_clamp(x: torch.Tensor, _min=None, _max=None) -> torch.Tensor:
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x


class ResBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: Optional[nn.Module] = None,
        layer_norm: bool = True,
        with_residual: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = activation if activation is not None else Swish()
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else None
        self.with_residual = with_residual
        self.res_proj = nn.Linear(input_dim, output_dim) if (with_residual and input_dim != output_dim) else None
        self.layer_norm = nn.LayerNorm(output_dim) if layer_norm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear(x)
        y = self.activation(y)
        if self.dropout is not None:
            y = self.dropout(y)
        if self.with_residual:
            res = self.res_proj(x) if self.res_proj is not None else x
            y = y + res
        if self.layer_norm is not None:
            y = self.layer_norm(y)
        return y


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10_000, dropout: float = 0.0):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        x = x + self.pe[:, :T, :].to(x.device)
        return self.dropout(x)


class Transformer(nn.Module):

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 1024,
        pooling: str = "mean",
        use_cls_token: bool = False,
        out_dim: Optional[int] = None,
        prenorm: bool = False,
        pos_dropout: float = 0.0,
        device: str = "cpu"
    ):
        super().__init__()
        self.device = torch.device(device)
        self.input_dim = input_dim
        self.d_model = d_model
        self.pooling = pooling
        self.use_cls_token = use_cls_token

        self.in_proj = nn.Identity() if input_dim == d_model else nn.Linear(input_dim, d_model)

        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len, dropout=pos_dropout)

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        else:
            self.cls_token = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=prenorm,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.final_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, out_dim) if out_dim is not None else None
        self.output_dim = out_dim if out_dim is not None else d_model

        self.to(self.device)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        return_sequence: bool = False
    ) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        x = x.float().to(self.device)

        if x.dim() == 2:
            x = x.unsqueeze(1) 
        if x.dim() != 3:
            raise ValueError(f"expected 2D/3D input, got {x.shape}")

        B, T, _ = x.shape

        x = self.in_proj(x) 

        if self.use_cls_token:
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)
            if src_key_padding_mask is not None:
                pad = torch.zeros((B, 1), dtype=torch.bool, device=src_key_padding_mask.device)
                src_key_padding_mask = torch.cat([pad, src_key_padding_mask], dim=1)

        x = self.pos_enc(x)

        x = self.encoder(x, mask=attn_mask, src_key_padding_mask=src_key_padding_mask)

        if return_sequence:
            x = self.final_norm(x)
            if self.out_proj is not None:
                x = self.out_proj(x)
            return x 

        if self.pooling == "cls":
            pooled = x[:, 0]
        elif self.pooling == "last":
            pooled = x[:, -1]
        elif self.pooling == "mean":
            if src_key_padding_mask is not None:
                valid = (~src_key_padding_mask).float()
                denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
                pooled = (x * valid.unsqueeze(-1)).sum(dim=1) / denom
            else:
                pooled = x.mean(dim=1)
        else:
            raise ValueError(f"unknown pooling {self.pooling}")

        pooled = self.final_norm(pooled)
        if self.out_proj is not None:
            pooled = self.out_proj(pooled)
        return pooled  


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, D = 4, 6, 32
    x = torch.randn(B, T, D)
    pad = torch.tensor([[False]*6, [False]*4 + [True]*2, [False]*6, [False]*6])
    net = Transformer(input_dim=D, d_model=64, nhead=4, num_layers=2, out_dim=128, pooling="mean")
    y = net(x, src_key_padding_mask=pad)
    print("pooled:", y.shape)
    net2 = Transformer(input_dim=D, d_model=64, nhead=4, num_layers=2, use_cls_token=True, pooling="cls")
    y2 = net2(x, src_key_padding_mask=pad)
    print("cls:", y2.shape)
    y3 = net(x, src_key_padding_mask=pad, return_sequence=True)
    print("seq:", y3.shape)
