import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from typing import Dict, Tuple, Union
from torch.nn import functional as F

from OffDRL.policy import SACPolicy


class CQLPolicy(SACPolicy):

    def __init__(
        self,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
        action_space: gym.spaces.Box,
        tau: float = 0.005,
        gamma: float = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        # CQL-specific
        cql_weight: float = 1.0,          
        temperature: float = 1.0,       
        max_q_backup: bool = False,       
        deterministic_backup: bool = True, 
        with_lagrange: bool = True,      
        lagrange_threshold: float = 10.0, 
        cql_alpha_lr: float = 1e-4,        
        num_repeat_actions: int = 10,     
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
            alpha=alpha,
        )

        self.action_space = action_space
        self._cql_weight = float(cql_weight)
        self._temperature = float(temperature)
        self._max_q_backup = bool(max_q_backup)
        self._deterministic_backup = bool(deterministic_backup)
        self._with_lagrange = bool(with_lagrange)
        self._lagrange_threshold = float(lagrange_threshold)
        self._num_repeat_actions = int(num_repeat_actions)

        device = next(self.actor.parameters()).device
 
        self.cql_log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.cql_alpha_optim = torch.optim.Adam([self.cql_log_alpha], lr=cql_alpha_lr)

    # ----------------- utils -----------------
    @staticmethod
    def _sum_logprob(logp: torch.Tensor) -> torch.Tensor:
        if logp.ndim > 1:
            return logp.sum(dim=-1, keepdim=True)
        return logp

    def _uniform_logprob_const(self, batch_size: int, device) -> torch.Tensor:
        low = torch.as_tensor(self.action_space.low, device=device, dtype=torch.float32)
        high = torch.as_tensor(self.action_space.high, device=device, dtype=torch.float32)
        width = (high - low).clamp_min(1e-6)
        const_scalar = -torch.log(width).sum()  
        return const_scalar.expand(batch_size, 1)  

    @torch.no_grad()
    def _sample_actions_from_policy(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        actions, logp = self.actforward(obs, deterministic=False)
        logp = self._sum_logprob(logp)
        return actions, logp

    # ----------------- learning -----------------
    def learn(self, batch: Dict) -> Dict[str, float]:
        obss        = batch["observations"]
        actions     = batch["actions"]
        next_obss   = batch["next_observations"]
        rewards     = batch["rewards"]
        terminateds = batch["terminateds"]
        truncateds  = batch["truncateds"]

        terminals = (terminateds.bool() | truncateds.bool()).float()

        if rewards.ndim == 1:
            rewards = rewards.unsqueeze(-1)
        if terminals.ndim == 1:
            terminals = terminals.unsqueeze(-1)

        batch_size = obss.shape[0]
        act_dim = actions.shape[-1]
        device = next(self.actor.parameters()).device

        # ---------------- actor update ----------------
        a_pi, logp_pi = self.actforward(obss)        
        logp_pi = self._sum_logprob(logp_pi)          
        q1a, q2a = self.critic1(obss, a_pi), self.critic2(obss, a_pi)
        actor_loss = (self._alpha * logp_pi - torch.min(q1a, q2a)).mean()

        self.actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            alpha_loss = -(self._log_alpha * (logp_pi.detach() + self._target_entropy)).mean()
            self.alpha_optim.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp().clamp_min(1e-8)

        # ---------------- critic TD target ----------------
        if self._max_q_backup:
            with torch.no_grad():
                rep_next = next_obss.unsqueeze(1).repeat(1, self._num_repeat_actions, 1) \
                    .view(batch_size * self._num_repeat_actions, next_obss.shape[-1])
                next_a, _ = self._sample_actions_from_policy(rep_next) 
                q1n = self.critic1_old(rep_next, next_a).view(batch_size, self._num_repeat_actions, 1).max(dim=1)[0]
                q2n = self.critic2_old(rep_next, next_a).view(batch_size, self._num_repeat_actions, 1).max(dim=1)[0]
                next_q = torch.min(q1n, q2n)
        else:
            with torch.no_grad():
                if self._deterministic_backup:
                    next_a, _ = self.actforward(next_obss, deterministic=True)
                    next_q = torch.min(
                        self.critic1_old(next_obss, next_a),
                        self.critic2_old(next_obss, next_a)
                    )
                else:
                    next_a, next_logp = self._sample_actions_from_policy(next_obss)
                    next_q = torch.min(
                        self.critic1_old(next_obss, next_a),
                        self.critic2_old(next_obss, next_a)
                    ) - self._alpha * next_logp

        target_q = rewards + self._gamma * (1.0 - terminals) * next_q

        q1, q2 = self.critic1(obss, actions), self.critic2(obss, actions)
        critic1_td = F.mse_loss(q1, target_q)
        critic2_td = F.mse_loss(q2, target_q)

        rand_act = torch.empty(batch_size * self._num_repeat_actions, act_dim, device=device) \
            .uniform_(float(self.action_space.low[0]), float(self.action_space.high[0]))

        rep_obss = obss.unsqueeze(1).repeat(1, self._num_repeat_actions, 1) \
            .view(batch_size * self._num_repeat_actions, obss.shape[-1])
        rep_next_obss = next_obss.unsqueeze(1).repeat(1, self._num_repeat_actions, 1) \
            .view(batch_size * self._num_repeat_actions, next_obss.shape[-1])

        with torch.no_grad():
            pi_a_s, pi_logp_s = self._sample_actions_from_policy(rep_obss)       
            pi_a_sp, pi_logp_sp = self._sample_actions_from_policy(rep_next_obss)

        q1_pi_s = self.critic1(rep_obss, pi_a_s)
        q2_pi_s = self.critic2(rep_obss, pi_a_s)
        v1_pi_s = q1_pi_s - pi_logp_s
        v2_pi_s = q2_pi_s - pi_logp_s

        q1_pi_sp = self.critic1(rep_next_obss, pi_a_sp)
        q2_pi_sp = self.critic2(rep_next_obss, pi_a_sp)
        v1_pi_sp = q1_pi_sp - pi_logp_sp
        v2_pi_sp = q2_pi_sp - pi_logp_sp

        q1_rand = self.critic1(rep_obss, rand_act)
        q2_rand = self.critic2(rep_obss, rand_act)
        const = self._uniform_logprob_const(q1_rand.shape[0], device)  
        v1_rand = q1_rand - const
        v2_rand = q2_rand - const

        def _view_br1(x): return x.view(batch_size, self._num_repeat_actions, 1)
        v1_pi_s  = _view_br1(v1_pi_s)
        v2_pi_s  = _view_br1(v2_pi_s)
        v1_pi_sp = _view_br1(v1_pi_sp)
        v2_pi_sp = _view_br1(v2_pi_sp)
        v1_rand  = _view_br1(v1_rand)
        v2_rand  = _view_br1(v2_rand)

        cat_q1 = torch.cat([v1_pi_s, v1_pi_sp, v1_rand], dim=1) 
        cat_q2 = torch.cat([v2_pi_s, v2_pi_sp, v2_rand], dim=1)

        conservative1 = (
            torch.logsumexp(cat_q1 / self._temperature, dim=1).mean() * self._cql_weight * self._temperature
            - q1.mean() * self._cql_weight
        )
        conservative2 = (
            torch.logsumexp(cat_q2 / self._temperature, dim=1).mean() * self._cql_weight * self._temperature
            - q2.mean() * self._cql_weight
        )

        if self._with_lagrange:
            cql_alpha = torch.clamp(self.cql_log_alpha.exp(), 0.0, 1e6)
            conservative1 = cql_alpha * (conservative1 - self._lagrange_threshold)
            conservative2 = cql_alpha * (conservative2 - self._lagrange_threshold)

            self.cql_alpha_optim.zero_grad(set_to_none=True)
            cql_alpha_loss = -0.5 * (conservative1 + conservative2)
            cql_alpha_loss.backward(retain_graph=True)
            self.cql_alpha_optim.step()

        critic1_loss = critic1_td + conservative1
        critic2_loss = critic2_td + conservative2

        self.critic1_optim.zero_grad(set_to_none=True)
        critic1_loss.backward(retain_graph=True)
        self.critic1_optim.step()

        self.critic2_optim.zero_grad(set_to_none=True)
        critic2_loss.backward()
        self.critic2_optim.step()

        self._sync_weight()

        # ---------------- logs ----------------
        result = {
            "loss/actor":   float(actor_loss.detach().cpu()),
            "loss/critic1": float(critic1_loss.detach().cpu()),
            "loss/critic2": float(critic2_loss.detach().cpu()),
        }
        if self._is_auto_alpha:
            result["loss/alpha"] = float(alpha_loss.detach().cpu())
            result["alpha"] = float(self._alpha.detach().cpu())
        if self._with_lagrange:
            result["loss/cql_alpha"] = float(cql_alpha_loss.detach().cpu())
            result["cql_alpha"] = float(torch.clamp(self.cql_log_alpha.exp(), 0.0, 1e6).detach().cpu())
        return result
