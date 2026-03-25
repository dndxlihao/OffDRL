import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from sklearn.neighbors import KDTree
from OffDRL.policy import TD3Policy


class _PRDCIndex:
 
    def __init__(
        self,
        states: np.ndarray,  
        actions: np.ndarray, 
        beta: float = 2.0,
        leaf_size: int = 40,
    ):
        assert states.ndim == 2 and actions.ndim == 2
        self.beta = float(beta)
        self.X = np.concatenate([self.beta * states, actions], axis=1).astype(np.float32)  
        self.A = actions.astype(np.float32)
        self.kdt = KDTree(self.X, leaf_size=leaf_size)

    def _prep_query(self, s: np.ndarray, a: np.ndarray) -> np.ndarray:
        return np.concatenate([self.beta * s, a], axis=1).astype(np.float32)

    def query_actions(self, s: np.ndarray, a: np.ndarray, k: int = 1) -> np.ndarray:
  
        xq = self._prep_query(s, a)
        k = min(k, len(self.A))
        d, idx = self.kdt.query(xq, k=k)  
        if k == 1:
            return self.A[idx[:, 0]]
        w = 1.0 / (d + 1e-6)
        w = w / w.sum(axis=1, keepdims=True)
        neigh = self.A[idx]
        return (w[..., None] * neigh).sum(axis=1)


class PRDCPolicy(TD3Policy):

    def __init__(
        self,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
        # TD3 
        tau: float = 0.005,
        gamma: float = 0.99,
        max_action: float = 1.0,
        policy_noise: float = 0.1,
        noise_clip: float = 0.2,
        update_actor_freq: int = 2,
        # PRDC 
        alpha: float = 2.5,     
        beta: float = 2.0,      
        dc_coef: float = 1.0,    
        knn_k: int = 1,         

        prdc_index: Optional[_PRDCIndex] = None,
        dataset_states: Optional[np.ndarray] = None,
        dataset_actions: Optional[np.ndarray] = None,
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
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            update_actor_freq=update_actor_freq,
        )
        self._alpha = float(alpha)
        self._beta = float(beta)
        self._dc_coef = float(dc_coef)
        self._knn_k = int(knn_k)
        self._last_actor_loss = 0.0

        if prdc_index is not None:
            self._index = prdc_index
        else:
            assert (dataset_states is not None) and (dataset_actions is not None), \
                "PRDCPolicy: provide prdc_index or dataset_states/actions to build one."
            self._index = _PRDCIndex(dataset_states, dataset_actions, beta=self._beta)

    def train(self) -> None:
        self.actor.train(); self.critic1.train(); self.critic2.train()

    def eval(self) -> None:
        self.actor.eval(); self.critic1.eval(); self.critic2.eval()

    def _sync_weight(self) -> None:
        for o, n in zip(self.actor_old.parameters(), self.actor.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic1_old.parameters(), self.critic1.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic2_old.parameters(), self.critic2.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        device = next(self.actor.parameters()).device
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        a = self.actor(obs_t).clamp(-self._max_action, self._max_action)
        return a.cpu().numpy().flatten()

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

        # -------- Critic update (TD3) --------
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
            a_pi = self.actor(obss)            
            q = self.critic1(obss, a_pi)

            lmbda = self._alpha / q.abs().mean().detach().clamp_min(1e-8)

            with torch.no_grad():
                S_np = obss.detach().cpu().numpy()
                A_np = a_pi.detach().cpu().numpy()
                a_tilde = self._index.query_actions(S_np, A_np, k=self._knn_k)   
                a_tilde = torch.as_tensor(a_tilde, dtype=torch.float32, device=obss.device)

            # PRDC
            loss_q  = -lmbda * q.mean()
            loss_dc = F.mse_loss(a_pi, a_tilde)
            actor_loss = loss_q + self._dc_coef * loss_dc

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
            "loss/prdc_dc": float(0.0 if self._cnt % self._freq != 1 else self._last_actor_loss),
        }
