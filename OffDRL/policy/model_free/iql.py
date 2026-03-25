import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from typing import Dict, Optional
from OffDRL.policy import BasePolicy


class IQLPolicy(BasePolicy):

    def __init__(
        self,
        actor: nn.Module,              
        critic_q1: nn.Module,
        critic_q2: nn.Module,
        critic_v: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic_q1_optim: torch.optim.Optimizer,
        critic_q2_optim: torch.optim.Optimizer,
        critic_v_optim: torch.optim.Optimizer,
        action_space,
        tau: float = 0.005,
        gamma: float = 0.99,
        expectile: float = 0.7,
        temperature: float = 3.0,
    ) -> None:
        super().__init__()

        self.actor = actor
        self.critic_q1, self.critic_q1_old = critic_q1, deepcopy(critic_q1)
        self.critic_q2, self.critic_q2_old = critic_q2, deepcopy(critic_q2)
        self.critic_v = critic_v

        self.critic_q1_old.eval()
        self.critic_q2_old.eval()

        self.actor_optim = actor_optim
        self.critic_q1_optim = critic_q1_optim
        self.critic_q2_optim = critic_q2_optim
        self.critic_v_optim = critic_v_optim

        self.action_space = action_space
        self._tau = float(tau)
        self._gamma = float(gamma)
        self._expectile = float(expectile)
        self._temperature = float(temperature)

        self._last_actor_loss = 0.0

    # ---------------- modes ----------------
    def train(self) -> None:
        self.actor.train()
        self.critic_q1.train()
        self.critic_q2.train()
        self.critic_v.train()

    def eval(self) -> None:
        self.actor.eval()
        self.critic_q1.eval()
        self.critic_q2.eval()
        self.critic_v.eval()

    # ---------------- utilities ----------------
    def _sync_weight(self) -> None:
        for o, n in zip(self.critic_q1_old.parameters(), self.critic_q1.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic_q2_old.parameters(), self.critic_q2.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)

    def _expectile_regression(self, diff: torch.Tensor) -> torch.Tensor:
        w = torch.where(diff > 0, self._expectile, 1.0 - self._expectile)
        return w * (diff ** 2)

    # ---------------- acting ----------------
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        device = next(self.actor.parameters()).device
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            dist = self.actor(obs_t)
            if deterministic:
                act = dist.mode()
            else:
                act = dist.sample()
        act = act.clamp_(min=self.action_space.low[0], max=self.action_space.high[0])
        return act.cpu().numpy().flatten()

    # ---------------- learning ----------------
    def learn(self, batch: Dict) -> Dict[str, float]:
        obss        = batch["observations"]
        actions     = batch["actions"]
        next_obss   = batch["next_observations"]
        rewards     = batch["rewards"]
        terminateds = batch.get("terminateds")
        truncateds  = batch.get("truncateds")
       
        if rewards.ndim == 1:
            rewards = rewards.unsqueeze(-1)
        if terminateds.ndim == 1:
            terminateds = terminateds.unsqueeze(-1)
        if truncateds.ndim == 1:
            truncateds = truncateds.unsqueeze(-1)

        cont_mask = (1.0 - terminateds.float()) * (1.0 - truncateds.float())

        # ---------------- Value (V) update via expectile regression ----------------
        with torch.no_grad():
            q1_old = self.critic_q1_old(obss, actions)
            q2_old = self.critic_q2_old(obss, actions)
            q_old = torch.min(q1_old, q2_old)
        v = self.critic_v(obss)
        v_loss = self._expectile_regression(q_old - v).mean()

        self.critic_v_optim.zero_grad(set_to_none=True)
        v_loss.backward()
        self.critic_v_optim.step()

        # ---------------- Q update (TD backup with V target) ----------------
        q1 = self.critic_q1(obss, actions)
        q2 = self.critic_q2(obss, actions)
        with torch.no_grad():
            next_v = self.critic_v(next_obss)
            target_q = rewards + self._gamma * cont_mask * next_v

        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)

        self.critic_q1_optim.zero_grad(set_to_none=True)
        q1_loss.backward()
        self.critic_q1_optim.step()

        self.critic_q2_optim.zero_grad(set_to_none=True)
        q2_loss.backward()
        self.critic_q2_optim.step()

        # ---------------- Actor update (advantage-weighted BC) ----------------
        with torch.no_grad():
            q1_old = self.critic_q1_old(obss, actions)
            q2_old = self.critic_q2_old(obss, actions)
            q_old = torch.min(q1_old, q2_old)
            v = self.critic_v(obss)
            adv = (q_old - v) / max(self._temperature, 1e-8)
            w = torch.exp(adv).clamp_(max=100.0)

        dist = self.actor(obss)
        logp = dist.log_prob(actions)
        if logp.ndim > 1:
            logp = logp.sum(dim=-1, keepdim=True)
        actor_loss = -(w * logp).mean()

        self.actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optim.step()

        self._sync_weight()

        self._last_actor_loss = float(actor_loss.detach().cpu())
        return {
            "loss/actor": self._last_actor_loss,
            "loss/q1": float(q1_loss.detach().cpu()),
            "loss/q2": float(q2_loss.detach().cpu()),
            "loss/v": float(v_loss.detach().cpu()),
        }
