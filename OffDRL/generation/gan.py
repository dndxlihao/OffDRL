import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple, Optional, Union


class GAN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int,
        latent_dim: int,
        max_action: Union[int, float],
        device: str = "cpu"
    ) -> None:
        super(GAN, self).__init__()
        # Generator (obs + z -> action)
        self.g1 = nn.Linear(input_dim + latent_dim, hidden_dim)
        self.g2 = nn.Linear(hidden_dim, hidden_dim)
        self.g3 = nn.Linear(hidden_dim, action_dim)

        # Discriminator (obs + action -> logit)
        self.d1 = nn.Linear(input_dim + action_dim, hidden_dim)
        self.d2 = nn.Linear(hidden_dim, hidden_dim)
        self.d3 = nn.Linear(hidden_dim, 1)

        self.max_action = float(max_action)
        self.latent_dim = latent_dim
        self.device = torch.device(device)

        self.to(device=self.device)

    def generate(self, obs: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generator: obs (B, input_dim), z (B, latent_dim) -> action (B, action_dim)
        If z is None, sample standard normal.
        """
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        obs = obs.to(self.device).float()
        B = obs.shape[0]
        if z is None:
            z = torch.randn((B, self.latent_dim), device=self.device)
        else:
            if not isinstance(z, torch.Tensor):
                z = torch.as_tensor(z, dtype=torch.float32)
            z = z.to(self.device).float()

        x = torch.cat([obs, z], dim=1)
        x = F.relu(self.g1(x))
        x = F.relu(self.g2(x))
        a = self.g3(x)
        return self.max_action * torch.tanh(a)

    def discriminate(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Discriminator returns logits (no sigmoid).
        obs: (B, input_dim), action: (B, action_dim)
        """
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32)
        obs = obs.to(self.device).float()
        action = action.to(self.device).float()
        x = torch.cat([obs, action], dim=1)
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        logit = self.d3(x).reshape(-1)
        return logit  # shape (B,)

    def forward(
        self,
        obs: torch.Tensor,
        real_action: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Forward convenience:
        - returns (gen_action, logits_real_or_None, logits_fake)
        - gen_action: (B, action_dim)
        - logits_real: (B,) or None if real_action is None
        - logits_fake: (B,)
        """
        gen_action = self.generate(obs, z)
        logits_fake = self.discriminate(obs, gen_action)
        logits_real = None
        if real_action is not None:
            logits_real = self.discriminate(obs, real_action)
        return gen_action, logits_real, logits_fake

    def generator_loss(self, logits_fake: torch.Tensor) -> torch.Tensor:

        target = torch.ones_like(logits_fake, device=self.device)
        loss = F.binary_cross_entropy_with_logits(logits_fake, target)
        return loss

    def discriminator_loss(self, logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:

        target_real = torch.ones_like(logits_real, device=self.device)
        target_fake = torch.zeros_like(logits_fake, device=self.device)
        loss_real = F.binary_cross_entropy_with_logits(logits_real, target_real)
        loss_fake = F.binary_cross_entropy_with_logits(logits_fake, target_fake)
        return 0.5 * (loss_real + loss_fake)


if __name__ == "__main__":
    torch.manual_seed(0)
    B = 4
    obs_dim = 10
    act_dim = 2
    model = GAN(obs_dim, act_dim, hidden_dim=64, latent_dim=8, max_action=1.0, device="cpu")
    obs = torch.randn(B, obs_dim)
    real_a = torch.randn(B, act_dim).clamp(-1, 1)
    gen_a, logits_real, logits_fake = model(obs, real_action=real_a)
    print("gen_a.shape", gen_a.shape)
    print("logits_real", logits_real.shape if logits_real is not None else None)
    print("logits_fake", logits_fake.shape)
    g_loss = model.generator_loss(logits_fake)
    d_loss = model.discriminator_loss(logits_real, logits_fake)
    print("g_loss", g_loss.item(), "d_loss", d_loss.item())