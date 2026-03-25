import copy
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from OffDRL.policy.model_free.td3 import TD3Policy  

class PerturbMLP(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([s, a], dim=-1)
        return self.net(sa)  


class BCQPolicy(TD3Policy):

    def __init__(
        self,
        critic1: nn.Module,
        critic2: nn.Module,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
        behavior_policy: nn.Module,
        behavior_policy_optim: torch.optim.Optimizer,
        obs_dim: int,
        act_dim: int,
        actor: Optional[nn.Module] = None,                       
        actor_optim: Optional[torch.optim.Optimizer] = None,    
        num_sampled_actions: int = 10,                         
        perturbation_limit: float = 0.1,                                                    
        l2_delta: float = 0.0,                              
        tau: float = 0.005,
        gamma: float = 0.99,
        max_action: float = 1.0,
        update_actor_freq: int = 2,                         
    ) -> None:

        if actor is None:
            actor = PerturbMLP(obs_dim, act_dim)
        if actor_optim is None:
            actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-3)

        super().__init__(
            actor=actor,
            critic1=critic1,
            critic2=critic2,
            actor_optim=actor_optim,
            critic1_optim=critic1_optim,
            critic2_optim=critic2_optim,
            tau=tau,
            gamma=gamma,
            max_action=max_action,
            update_actor_freq=update_actor_freq,
        )

        self.behavior_policy = behavior_policy
        self.behavior_policy_optim = behavior_policy_optim

        self._obs_dim = int(obs_dim)
        self._act_dim = int(act_dim)
        self._num_sampled_actions = int(num_sampled_actions)
        self._phi = float(perturbation_limit)
        self._l2_delta = float(l2_delta)

    def train(self) -> None:
        super().train()
        self.behavior_policy.train()

    def eval(self) -> None:
        super().eval()
        self.behavior_policy.eval()

    def _clamp_action(self, a: torch.Tensor) -> torch.Tensor:
        return a.clamp(min=-self._max_action, max=self._max_action)

    def _perturb(self, s: torch.Tensor, a: torch.Tensor, use_old_actor: bool = False) -> torch.Tensor:
        net = self.actor_old if use_old_actor else self.actor 
        if hasattr(net, "forward") and net.forward.__code__.co_argcount >= 3:
            delta = net(s, a) 
        else:
            sa = torch.cat([s, a], dim=-1)
            delta = net(sa)
        delta = torch.tanh(delta) * (self._phi * self._max_action)
        a_pert = self._clamp_action(a + delta)
        return a_pert

    @torch.no_grad()
    def select_action(self, obs, deterministic: bool = True):
        device = next(self.actor.parameters()).device
        s = obs if isinstance(obs, torch.Tensor) else torch.as_tensor(obs, dtype=torch.float32, device=device)
        if s.ndim == 1:
            s = s.unsqueeze(0)

        s_rep = s.repeat(self._num_sampled_actions, 1)
        a_cand = self.behavior_policy.decode(s_rep)                   
        a_pert = self._perturb(s_rep, a_cand, use_old_actor=False)     

        q1 = self.critic1(s_rep, a_pert)
        q2 = self.critic2(s_rep, a_pert)
        q = torch.min(q1, q2).reshape(1, self._num_sampled_actions, -1).squeeze(-1)
        idx = q.argmax(dim=1)
        a_best = a_pert.reshape(1, self._num_sampled_actions, -1)[0, idx.item(), :]
        return a_best.cpu().numpy()

    def learn(self, batch: Dict):
        obss        = batch["observations"]
        actions     = batch["actions"]
        next_obss   = batch["next_observations"]
        rewards     = batch["rewards"]
        terminateds = batch["terminateds"]
        truncateds  = batch["truncateds"]

        if rewards.ndim == 1:     rewards     = rewards.unsqueeze(-1)
        if terminateds.ndim == 1: terminateds = terminateds.unsqueeze(-1)
        if truncateds.ndim == 1:  truncateds  = truncateds.unsqueeze(-1)

        dones = (terminateds.bool() | truncateds.bool()).float()
        not_done = 1.0 - dones
        B = obss.shape[0]
        N = self._num_sampled_actions

        # ---------------- VAE（behavior policy） ----------------
        recon, mean, std = self.behavior_policy(obss, actions)
        recon_loss = F.mse_loss(recon, actions)
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + KL_loss
        self.behavior_policy_optim.zero_grad(set_to_none=True)
        vae_loss.backward()
        self.behavior_policy_optim.step()

        # ---------------- Update Critic ----------------
        with torch.no_grad():
            srep_next = torch.repeat_interleave(next_obss, repeats=N, dim=0)     
            a_cand_next = self.behavior_policy.decode(srep_next)              
            a_pert_next = self._perturb(srep_next, a_cand_next, use_old_actor=True)

            q1n = self.critic1_old(srep_next, a_pert_next).reshape(B, N, 1)
            q2n = self.critic2_old(srep_next, a_pert_next).reshape(B, N, 1)
            next_q = torch.min(q1n, q2n).max(dim=1, keepdim=False)[0]       

            target_q = rewards + self._gamma * not_done * next_q      

        q1 = self.critic1(obss, actions)
        q2 = self.critic2(obss, actions)
        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)

        self.critic1_optim.zero_grad(set_to_none=True)
        critic1_loss.backward()
        self.critic1_optim.step()

        self.critic2_optim.zero_grad(set_to_none=True)
        critic2_loss.backward()
        self.critic2_optim.step()

        # ---------------- Update Actor (Perturbation Model) ----------------
        srep = torch.repeat_interleave(obss, repeats=N, dim=0)                  
        with torch.no_grad():
            a_base = self.behavior_policy.decode(srep)                         
        a_pert = self._perturb(srep, a_base, use_old_actor=False)             

        q1p = self.critic1(srep, a_pert).reshape(B, N, 1)
        q2p = self.critic2(srep, a_pert).reshape(B, N, 1)
        q_min = torch.min(q1p, q2p)                                        
        max_q = q_min.max(dim=1, keepdim=False)[0]                      

        actor_loss = -max_q.mean()
        if self._l2_delta > 0.0:
            actor_loss = actor_loss + self._l2_delta * (a_pert - a_base).pow(2).mean()

        self.actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optim.step()

        if self._cnt % self._freq == 0:
            self._sync_weight()  
        self._cnt += 1

        # ---------------- log ----------------
        return {
            "loss/actor":           float(actor_loss.detach().cpu()),
            "loss/critic1":         float(critic1_loss.detach().cpu()),
            "loss/critic2":         float(critic2_loss.detach().cpu()),
            "loss/behavior_policy": float(vae_loss.detach().cpu()),
        }
