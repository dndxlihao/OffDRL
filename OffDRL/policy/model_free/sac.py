import numpy as np
import torch
import torch.nn as nn

from copy import deepcopy
from typing import Dict, Union, Tuple
from OffDRL.policy import BasePolicy


class SACPolicy(BasePolicy):

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
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        target_update_interval: int = 1, 
    ) -> None:
        super().__init__()

        self.actor = actor
        self.critic1, self.critic1_old = critic1, deepcopy(critic1)
        self.critic1_old.eval()
        self.critic2, self.critic2_old = critic2, deepcopy(critic2)
        self.critic2_old.eval()

        self.actor_optim = actor_optim
        self.critic1_optim = critic1_optim
        self.critic2_optim = critic2_optim

        self._tau = float(tau)
        self._gamma = float(gamma)
        self._target_update_interval = int(target_update_interval)
        self._grad_steps = 0

        self._is_auto_alpha = False
        if isinstance(alpha, tuple):
            self._is_auto_alpha = True
            self._target_entropy, self._log_alpha, self.alpha_optim = alpha
            self._alpha = self._log_alpha.detach().exp()
        else:
            self._alpha = torch.as_tensor(alpha, dtype=torch.float32)

    def train(self) -> None:
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

    def eval(self) -> None:
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()

    def _sync_weight(self) -> None:
        for o, n in zip(self.critic1_old.parameters(), self.critic1.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic2_old.parameters(), self.critic2.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)

    def actforward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.actor(obs) 
        if deterministic:
            squashed_action, raw_action = dist.mode()
        else:
            squashed_action, raw_action = dist.rsample()
        log_prob = dist.log_prob(squashed_action, raw_action)  
        return squashed_action, log_prob

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> np.ndarray:
        with torch.no_grad():
            device = next(self.actor.parameters()).device
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            if obs_t.ndim == 1:
                obs_t = obs_t.unsqueeze(0)
            action, _ = self.actforward(obs_t, deterministic)
            action = action.squeeze(0)
        return action.cpu().numpy()

    def learn(self, batch: Dict) -> Dict[str, float]:
        obss         = batch["observations"]
        actions      = batch["actions"]
        next_obss    = batch["next_observations"]
        rewards      = batch["rewards"]
        terminateds  = batch["terminateds"]
        truncateds   = batch["truncateds"]

        # ---------------- Critic update ----------------
        q1 = self.critic1(obss, actions)
        q2 = self.critic2(obss, actions)

        with torch.no_grad():
            next_actions, next_log_probs = self.actforward(next_obss, deterministic=False)
            if next_log_probs.ndim == 1:
                next_log_probs = next_log_probs.unsqueeze(-1)  

            next_q = torch.min(
                self.critic1_old(next_obss, next_actions),
                self.critic2_old(next_obss, next_actions)
            ) - self._alpha * next_log_probs

            if rewards.ndim == 1:
                rewards = rewards.unsqueeze(-1)
            if terminateds.ndim == 1:
                terminateds = terminateds.unsqueeze(-1)
            if truncateds.ndim == 1:
                truncateds = truncateds.unsqueeze(-1)

            cont_mask = (1.0 - terminateds.float()) * (1.0 - truncateds.float())

            target_q = rewards + self._gamma * cont_mask * next_q

        critic1_loss = (q1 - target_q).pow(2).mean()
        self.critic1_optim.zero_grad(set_to_none=True)
        critic1_loss.backward()
        self.critic1_optim.step()

        critic2_loss = (q2 - target_q).pow(2).mean()
        self.critic2_optim.zero_grad(set_to_none=True)
        critic2_loss.backward()
        self.critic2_optim.step()

        # ---------------- Actor update ----------------
        a, log_probs = self.actforward(obss, deterministic=False)
        if log_probs.ndim == 1:
            log_probs = log_probs.unsqueeze(-1)

        q1a, q2a = self.critic1(obss, a), self.critic2(obss, a)
        actor_loss = (self._alpha * log_probs - torch.min(q1a, q2a)).mean()

        self.actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optim.step()

        # ---------------- Temperature (alpha) update ----------------
        if self._is_auto_alpha:
            alpha_loss = -(self._log_alpha * (log_probs + self._target_entropy).detach()).mean()
            self.alpha_optim.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp().clamp_min(1e-8)

        # ---------------- Target network soft update ----------------
        self._grad_steps += 1
        if self._grad_steps % self._target_update_interval == 0:
            self._sync_weight()

        result = {
            "loss/actor":   float(actor_loss.detach().cpu()),
            "loss/critic1": float(critic1_loss.detach().cpu()),
            "loss/critic2": float(critic2_loss.detach().cpu()),
        }
        if self._is_auto_alpha:
            result["loss/alpha"] = float(alpha_loss.detach().cpu())
            result["alpha"] = float(self._alpha.detach().cpu())
        return result
