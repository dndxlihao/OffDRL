import numpy as np
import torch

from OffDRL.backbone import MLP
from OffDRL.kernel import DiffusionKernel
from OffDRL.modules import Critic
from OffDRL.policy.generative_rl import DiffusionQLPolicy


def _build_critic(obs_dim: int, act_dim: int) -> Critic:
    backbone = MLP(input_dim=obs_dim + act_dim, hidden_dims=[32, 32])
    return Critic(backbone, device="cpu")


def test_diffusion_ql_policy_smoke() -> None:
    torch.manual_seed(0)

    obs_dim = 4
    act_dim = 2
    actor = DiffusionKernel(
        obs_dim=obs_dim,
        action_dim=act_dim,
        max_action=1.0,
        hidden_dim=32,
        n_timesteps=8,
        action_low=np.array([-0.2, -1.0], dtype=np.float32),
        action_high=np.array([0.3, 0.5], dtype=np.float32),
    )
    critic1 = _build_critic(obs_dim, act_dim)
    critic2 = _build_critic(obs_dim, act_dim)

    policy = DiffusionQLPolicy(
        actor=actor,
        critic1=critic1,
        critic2=critic2,
        actor_optim=torch.optim.Adam(actor.parameters(), lr=1e-3),
        critic1_optim=torch.optim.Adam(critic1.parameters(), lr=1e-3),
        critic2_optim=torch.optim.Adam(critic2.parameters(), lr=1e-3),
        eta=0.5,
        ema_start_step=0,
        ema_update_every=1,
        num_backup_samples=4,
        num_action_samples=6,
        grad_norm=1.0,
    )

    batch = {
        "observations": torch.randn(16, obs_dim),
        "actions": torch.tanh(torch.randn(16, act_dim)),
        "next_observations": torch.randn(16, obs_dim),
        "rewards": torch.randn(16),
        "terminateds": torch.zeros(16, dtype=torch.bool),
        "truncateds": torch.zeros(16, dtype=torch.bool),
    }

    loss = policy.learn(batch)
    assert "loss/actor" in loss
    assert "loss/bc" in loss
    assert np.isfinite(loss["loss/critic1"])

    deterministic_action = policy.select_action(np.zeros((1, obs_dim), dtype=np.float32), deterministic=True)
    deterministic_action_2 = policy.select_action(np.zeros((1, obs_dim), dtype=np.float32), deterministic=True)
    stochastic_action = policy.select_action(np.zeros((1, obs_dim), dtype=np.float32), deterministic=False)

    assert deterministic_action.shape == (act_dim,)
    assert stochastic_action.shape == (act_dim,)
    assert np.allclose(deterministic_action, deterministic_action_2)
    assert deterministic_action[0] >= -0.2 - 1e-6 and deterministic_action[0] <= 0.3 + 1e-6
    assert deterministic_action[1] >= -1.0 - 1e-6 and deterministic_action[1] <= 0.5 + 1e-6
