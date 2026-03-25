import numpy as np
import torch
import torch.nn as nn

from copy import deepcopy
from typing import Dict, Optional
from OffDRL.policy import BasePolicy
from OffDRL.utils.noise import GaussianNoise


class TD3Policy(BasePolicy):

    def __init__(
        self,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
        tau: float = 0.005,
        gamma: float = 0.99,
        max_action: float = 1.0,
        exploration_noise: Optional[GaussianNoise] = None,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        update_actor_freq: int = 2,
    ) -> None:
        super().__init__()

        self.actor = actor
        self.actor_old = deepcopy(actor).eval()
        self.actor_optim = actor_optim

        self.critic1 = critic1
        self.critic1_old = deepcopy(critic1).eval()
        self.critic1_optim = critic1_optim

        self.critic2 = critic2
        self.critic2_old = deepcopy(critic2).eval()
        self.critic2_optim = critic2_optim

        self._tau = float(tau)
        self._gamma = float(gamma)

        self._max_action = float(max_action)
        self.exploration_noise = exploration_noise or GaussianNoise(mu=0.0, sigma=0.2)
        self._policy_noise = float(policy_noise)
        self._noise_clip = float(noise_clip)
        self._freq = int(update_actor_freq)

        self._cnt = 0
        self._last_actor_loss = 0.0

    def train(self) -> None:
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

    def eval(self) -> None:
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()

    def _soft_update(self, tgt: nn.Module, src: nn.Module) -> None:
        for o, n in zip(tgt.parameters(), src.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)

    def _sync_weight(self) -> None:
        self._soft_update(self.actor_old, self.actor)
        self._soft_update(self.critic1_old, self.critic1)
        self._soft_update(self.critic2_old, self.critic2)

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            device = next(self.actor.parameters()).device
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            if obs_t.ndim == 1:
                obs_t = obs_t.unsqueeze(0)

            out = self.actor(obs_t)
            action = out[0] if isinstance(out, (tuple, list)) else out
            action = action.squeeze(0).cpu().numpy()

        if not deterministic:
            noise = self.exploration_noise(action.shape)
            action = np.clip(action + noise, -self._max_action, self._max_action)

        return action

    def learn(self, batch: Dict) -> Dict[str, float]:
        obss          = batch["observations"]
        actions       = batch["actions"]
        next_obss     = batch["next_observations"]
        rewards       = batch["rewards"]
        terminateds   = batch["terminateds"]
        truncateds    = batch["truncateds"]

        if rewards.ndim == 1: rewards = rewards.unsqueeze(-1)
        if terminateds.ndim == 1: terminateds = terminateds.unsqueeze(-1)
        if truncateds.ndim == 1:  truncateds  = truncateds.unsqueeze(-1)

        dones = torch.logical_or(terminateds.bool(), truncateds.bool()).float()

        # ---------- Critic update ----------
        q1, q2 = self.critic1(obss, actions), self.critic2(obss, actions)
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self._policy_noise).clamp(-self._noise_clip, self._noise_clip)
            out_old = self.actor_old(next_obss)
            next_actions = out_old[0] if isinstance(out_old, (tuple, list)) else out_old
            next_actions = (next_actions + noise).clamp(-self._max_action, self._max_action)

            next_q = torch.min(self.critic1_old(next_obss, next_actions),
                               self.critic2_old(next_obss, next_actions))
            target_q = rewards + self._gamma * (1.0 - dones) * next_q

        critic1_loss = (q1 - target_q).pow(2).mean()
        critic2_loss = (q2 - target_q).pow(2).mean()

        self.critic1_optim.zero_grad(set_to_none=True)
        critic1_loss.backward()
        self.critic1_optim.step()

        self.critic2_optim.zero_grad(set_to_none=True)
        critic2_loss.backward()
        self.critic2_optim.step()

        # ---------- Actor update ----------
        if self._cnt % self._freq == 0:
            out = self.actor(obss)
            a = out[0] if isinstance(out, (tuple, list)) else out
            q = self.critic1(obss, a)
            actor_loss = -q.mean()
            self.actor_optim.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optim.step()
            self._last_actor_loss = float(actor_loss.detach().cpu())
            self._sync_weight()

        self._cnt += 1

        return {
            "loss/actor":   float(self._last_actor_loss),
            "loss/critic1": float(critic1_loss.detach().cpu()),
            "loss/critic2": float(critic2_loss.detach().cpu()),
        }
