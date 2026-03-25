import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.dim // 2
        emb = math.log(10000.0) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


def extract(buffer: torch.Tensor, timesteps: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    batch_size = timesteps.shape[0]
    out = buffer.gather(0, timesteps)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps: int, s: float = 0.008, dtype=torch.float32) -> torch.Tensor:
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.tensor(np.clip(betas, a_min=0.0, a_max=0.999), dtype=dtype)


def linear_beta_schedule(
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    dtype=torch.float32,
) -> torch.Tensor:
    return torch.tensor(np.linspace(beta_start, beta_end, timesteps), dtype=dtype)


def vp_beta_schedule(timesteps: int, dtype=torch.float32) -> torch.Tensor:
    t = np.arange(1, timesteps + 1)
    b_max = 10.0
    b_min = 0.1
    alpha = np.exp(-b_min / timesteps - 0.5 * (b_max - b_min) * (2 * t - 1) / (timesteps ** 2))
    betas = 1.0 - alpha
    return torch.tensor(betas, dtype=dtype)


class WeightedLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: Union[float, torch.Tensor] = 1.0,
    ) -> torch.Tensor:
        loss = self._loss(pred, target)
        return (loss * weights).mean()

    def _loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class WeightedL1(WeightedLoss):
    def _loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.abs(pred - target)


class WeightedL2(WeightedLoss):
    def _loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target, reduction="none")


class EMA:
    def __init__(self, beta: float) -> None:
        self.beta = float(beta)

    def update_model_average(self, averaged_model: nn.Module, current_model: nn.Module) -> None:
        for current_params, averaged_params in zip(current_model.parameters(), averaged_model.parameters()):
            averaged_params.data = self.update_average(averaged_params.data, current_params.data)

    def update_average(self, old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        if old is None:
            return new
        return old * self.beta + (1.0 - self.beta) * new


class DiffusionMLP(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        time_dim: int = 16,
    ) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
        )
        input_dim = int(obs_dim) + int(action_dim) + int(time_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(
        self,
        noisy_action: torch.Tensor,
        timesteps: torch.Tensor,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        time_emb = self.time_mlp(timesteps)
        x = torch.cat([noisy_action, time_emb, obs], dim=-1)
        return self.net(x)


class DiffusionKernel(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        max_action: float,
        model: Optional[nn.Module] = None,
        hidden_dim: int = 256,
        time_dim: int = 16,
        beta_schedule: str = "linear",
        n_timesteps: int = 100,
        loss_type: str = "l2",
        clip_denoised: bool = True,
        predict_epsilon: bool = True,
        action_low: Optional[Union[np.ndarray, torch.Tensor, float]] = None,
        action_high: Optional[Union[np.ndarray, torch.Tensor, float]] = None,
    ) -> None:
        super().__init__()

        if model is None:
            model = DiffusionMLP(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                time_dim=time_dim,
            )

        if beta_schedule == "linear":
            betas = linear_beta_schedule(n_timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(n_timesteps)
        elif beta_schedule == "vp":
            betas = vp_beta_schedule(n_timesteps)
        else:
            raise ValueError(f"Unsupported beta schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.max_action = float(max_action)
        self.model = model
        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = bool(clip_denoised)
        self.predict_epsilon = bool(predict_epsilon)
        self.loss_fn = {"l1": WeightedL1, "l2": WeightedL2}[loss_type]()

        default_low = torch.full((self.action_dim,), -self.max_action, dtype=torch.float32)
        default_high = torch.full((self.action_dim,), self.max_action, dtype=torch.float32)
        action_low_t = self._format_action_bound(action_low, default_low, "action_low")
        action_high_t = self._format_action_bound(action_high, default_high, "action_high")
        if not torch.all(action_low_t < action_high_t):
            raise ValueError("Each action_low entry must be strictly smaller than action_high.")

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )
        self.register_buffer("action_low", action_low_t)
        self.register_buffer("action_high", action_high_t)

    def _format_action_bound(
        self,
        bound: Optional[Union[np.ndarray, torch.Tensor, float]],
        default: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        if bound is None:
            return default
        bound_t = torch.as_tensor(bound, dtype=torch.float32).flatten()
        if bound_t.numel() == 1:
            bound_t = bound_t.repeat(self.action_dim)
        if bound_t.numel() != self.action_dim:
            raise ValueError(f"{name} must have shape ({self.action_dim},) or be a scalar.")
        return bound_t

    def clamp_action(self, action: torch.Tensor) -> torch.Tensor:
        low = self.action_low.to(device=action.device, dtype=action.dtype)
        high = self.action_high.to(device=action.device, dtype=action.dtype)
        return torch.max(torch.min(action, high), low)

    def predict_start_from_noise(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, timesteps, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, timesteps, x_t.shape) * noise
            )
        return noise

    def q_posterior(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        posterior_mean = (
            extract(self.posterior_mean_coef1, timesteps, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, timesteps, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, timesteps, x_t.shape)
        posterior_log_variance = extract(self.posterior_log_variance_clipped, timesteps, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance

    def p_mean_variance(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_recon = self.predict_start_from_noise(
            x_t=x,
            timesteps=timesteps,
            noise=self.model(x, timesteps, obs),
        )
        if self.clip_denoised:
            x_recon = self.clamp_action(x_recon)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon,
            x_t=x,
            timesteps=timesteps,
        )
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, timesteps=timesteps, obs=obs)
        noise = torch.zeros_like(x) if deterministic else torch.randn_like(x)
        nonzero_mask = (1 - (timesteps == 0).float()).reshape(batch_size, *((1,) * (x.ndim - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_sample_loop(
        self,
        obs: torch.Tensor,
        shape: Tuple[int, ...],
        deterministic: bool = False,
    ) -> torch.Tensor:
        x = torch.zeros(shape, device=self.betas.device) if deterministic else torch.randn(shape, device=self.betas.device)
        for step in reversed(range(self.n_timesteps)):
            timesteps = torch.full((shape[0],), step, device=self.betas.device, dtype=torch.long)
            x = self.p_sample(x=x, timesteps=timesteps, obs=obs, deterministic=deterministic)
        return x

    def sample(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        batch_size = obs.shape[0]
        action = self.p_sample_loop(
            obs=obs,
            shape=(batch_size, self.action_dim),
            deterministic=deterministic,
        )
        return self.clamp_action(action)

    def q_sample(
        self,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape) * noise
        )

    def p_losses(
        self,
        x_start: torch.Tensor,
        obs: torch.Tensor,
        timesteps: torch.Tensor,
        weights: Union[float, torch.Tensor] = 1.0,
    ) -> torch.Tensor:
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, timesteps=timesteps, noise=noise)
        x_recon = self.model(x_noisy, timesteps, obs)
        target = noise if self.predict_epsilon else x_start
        return self.loss_fn(x_recon, target, weights)

    def loss(
        self,
        actions: torch.Tensor,
        obs: torch.Tensor,
        weights: Union[float, torch.Tensor] = 1.0,
    ) -> torch.Tensor:
        batch_size = actions.shape[0]
        timesteps = torch.randint(0, self.n_timesteps, (batch_size,), device=actions.device).long()
        return self.p_losses(x_start=actions, obs=obs, timesteps=timesteps, weights=weights)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self.sample(obs, deterministic=deterministic)
