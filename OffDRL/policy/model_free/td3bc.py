import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict
from OffDRL.policy import TD3Policy
from OffDRL.utils.noise import GaussianNoise


class TD3BCPolicy(TD3Policy):

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
        exploration_noise: GaussianNoise = GaussianNoise,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        update_actor_freq: int = 2,
        alpha: float = 2.5,
    ) -> None:
        super().__init__(
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            tau=tau,
            gamma=gamma,
            max_action=max_action,
            exploration_noise=exploration_noise,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            update_actor_freq=update_actor_freq,
        )
        self._alpha = float(alpha)
        self._last_actor_loss = 0.0

    # ---------------- basic modes ----------------
    def train(self) -> None:
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

    def eval(self) -> None:
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()

    # ---------------- target soft update ----------------
    def _sync_weight(self) -> None:
        for o, n in zip(self.actor_old.parameters(), self.actor.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic1_old.parameters(), self.critic1.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic2_old.parameters(), self.critic2.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)

    # ---------------- action selection (eval uses deterministic=True) ----------------
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        device = next(self.actor.parameters()).device
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action_t = self.actor(obs_t)              
        action = action_t.cpu().numpy().flatten()
        if not deterministic:
            action = action + self.exploration_noise(action.shape)
            action = np.clip(action, -self._max_action, self._max_action)
        return action

    # ---------------- single train step ----------------
    def learn(self, batch: Dict) -> Dict[str, float]:
        obss        = batch["observations"]
        actions     = batch["actions"]
        next_obss   = batch["next_observations"]
        rewards     = batch["rewards"]
        terminateds = batch["terminateds"].float()
        truncateds  = batch["truncateds"].float()

        if rewards.ndim == 1:     rewards     = rewards.unsqueeze(-1)
        if terminateds.ndim == 1: terminateds = terminateds.unsqueeze(-1)
        if truncateds.ndim == 1:  truncateds  = truncateds.unsqueeze(-1)

        terminals = (terminateds.bool() | truncateds.bool()).float()

        # -------- Critic update --------
        q1, q2 = self.critic1(obss, actions), self.critic2(obss, actions)
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self._policy_noise).clamp(
                -self._noise_clip, self._noise_clip
            )
            next_actions = (self.actor_old(next_obss) + noise).clamp(
                -self._max_action, self._max_action
            )
            next_q = torch.min(
                self.critic1_old(next_obss, next_actions),
                self.critic2_old(next_obss, next_actions),
            )
            target_q = rewards + self._gamma * (1.0 - terminals) * next_q

        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)

        self.critic1_optim.zero_grad(set_to_none=True)
        critic1_loss.backward()
        self.critic1_optim.step()

        self.critic2_optim.zero_grad(set_to_none=True)
        critic2_loss.backward()
        self.critic2_optim.step()

        # -------- Actor update (delayed) --------
        if self._cnt % self._freq == 0:
            a = self.actor(obss)                  
            q = self.critic1(obss, a)

            lmbda = self._alpha / q.abs().mean().detach().clamp_min(1e-8)
            actor_loss = -lmbda * q.mean() + F.mse_loss(a, actions)

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
