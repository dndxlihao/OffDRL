import copy
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from OffDRL.kernel.diffusion import EMA
from OffDRL.policy.base import BasePolicy


class DiffusionQLPolicy(BasePolicy):
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
        eta: float = 1.0,
        max_action: float = 1.0,
        max_q_backup: bool = False,
        num_backup_samples: int = 10,
        num_action_samples: int = 50,
        ema_decay: float = 0.995,
        ema_start_step: int = 1000,
        ema_update_every: int = 5,
        grad_norm: float = 0.0,
        action_selection_temperature: float = 1.0,
        actor_lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        critic1_lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        critic2_lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> None:
        super().__init__()

        self.actor = actor
        self.actor_target = copy.deepcopy(actor).eval()
        self.actor_optim = actor_optim

        self.critic1 = critic1
        self.critic1_target = copy.deepcopy(critic1).eval()
        self.critic1_optim = critic1_optim

        self.critic2 = critic2
        self.critic2_target = copy.deepcopy(critic2).eval()
        self.critic2_optim = critic2_optim

        self._tau = float(tau)
        self._gamma = float(gamma)
        self._eta = float(eta)
        self._max_action = float(max_action)
        self._max_q_backup = bool(max_q_backup)
        self._num_backup_samples = int(num_backup_samples)
        self._num_action_samples = int(num_action_samples)
        self._ema = EMA(ema_decay)
        self._ema_start_step = int(ema_start_step)
        self._ema_update_every = int(ema_update_every)
        self._grad_norm = float(grad_norm)
        self._action_selection_temperature = float(action_selection_temperature)
        self._step = 0

        self.actor_lr_scheduler = actor_lr_scheduler
        self.critic1_lr_scheduler = critic1_lr_scheduler
        self.critic2_lr_scheduler = critic2_lr_scheduler

    def train(self) -> None:
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

    def eval(self) -> None:
        self.actor.eval()
        self.actor_target.eval()
        self.critic1.eval()
        self.critic1_target.eval()
        self.critic2.eval()
        self.critic2_target.eval()

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                (1.0 - self._tau) * target_param.data + self._tau * source_param.data
            )

    def _step_actor_ema(self) -> None:
        if self._step < self._ema_start_step:
            self.actor_target.load_state_dict(self.actor.state_dict())
            return
        self._ema.update_model_average(self.actor_target, self.actor)

    def _candidate_actions(
        self,
        obs: torch.Tensor,
        num_candidates: int,
        actor: nn.Module,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = obs.shape[0]
        obs_rep = torch.repeat_interleave(obs, repeats=num_candidates, dim=0)
        actions = actor.sample(obs_rep, deterministic=deterministic)
        q_values = torch.min(
            self.critic1_target(obs_rep, actions),
            self.critic2_target(obs_rep, actions),
        ).view(batch_size, num_candidates)
        actions = actions.view(batch_size, num_candidates, -1)
        return actions, q_values

    def _clamp_action(self, actions: torch.Tensor) -> torch.Tensor:
        if hasattr(self.actor, "clamp_action"):
            return self.actor.clamp_action(actions)
        return actions.clamp(-self._max_action, self._max_action)

    @torch.no_grad()
    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        device = next(self.actor.parameters()).device
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)

        if deterministic:
            chosen = self.actor_target.sample(obs_t, deterministic=True)
        else:
            actions, q_values = self._candidate_actions(
                obs_t,
                self._num_action_samples,
                actor=self.actor,
                deterministic=False,
            )
            temperature = max(self._action_selection_temperature, 1e-6)
            probs = F.softmax(q_values / temperature, dim=1)
            indices = torch.multinomial(probs, 1).squeeze(-1)
            batch_idx = torch.arange(obs_t.shape[0], device=device)
            chosen = actions[batch_idx, indices]

        chosen = self._clamp_action(chosen)
        chosen_np = chosen.cpu().numpy()
        return chosen_np[0] if chosen_np.shape[0] == 1 else chosen_np

    def learn(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obss = batch["observations"]
        actions = batch["actions"]
        next_obss = batch["next_observations"]
        rewards = batch["rewards"]
        terminateds = batch["terminateds"]
        truncateds = batch["truncateds"]

        if rewards.ndim == 1:
            rewards = rewards.unsqueeze(-1)
        if terminateds.ndim == 1:
            terminateds = terminateds.unsqueeze(-1)
        if truncateds.ndim == 1:
            truncateds = truncateds.unsqueeze(-1)

        dones = torch.logical_or(terminateds.bool(), truncateds.bool()).float()
        not_done = 1.0 - dones
        batch_size = obss.shape[0]

        current_q1 = self.critic1(obss, actions)
        current_q2 = self.critic2(obss, actions)

        with torch.no_grad():
            if self._max_q_backup:
                next_obss_rep = torch.repeat_interleave(
                    next_obss,
                    repeats=self._num_backup_samples,
                    dim=0,
                )
                next_actions_rep = self.actor_target.sample(next_obss_rep)
                target_q1 = self.critic1_target(next_obss_rep, next_actions_rep).view(
                    batch_size, self._num_backup_samples, 1
                )
                target_q2 = self.critic2_target(next_obss_rep, next_actions_rep).view(
                    batch_size, self._num_backup_samples, 1
                )
                target_q1 = target_q1.max(dim=1)[0]
                target_q2 = target_q2.max(dim=1)[0]
                target_q = torch.min(target_q1, target_q2)
            else:
                next_actions = self.actor_target.sample(next_obss)
                target_q = torch.min(
                    self.critic1_target(next_obss, next_actions),
                    self.critic2_target(next_obss, next_actions),
                )

            target_q = rewards + not_done * self._gamma * target_q

        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        self.critic1_optim.zero_grad(set_to_none=True)
        critic1_loss.backward()
        if self._grad_norm > 0.0:
            nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=self._grad_norm, norm_type=2)
        self.critic1_optim.step()

        self.critic2_optim.zero_grad(set_to_none=True)
        critic2_loss.backward()
        if self._grad_norm > 0.0:
            nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=self._grad_norm, norm_type=2)
        self.critic2_optim.step()

        bc_loss = self.actor.loss(actions, obss)
        new_actions = self.actor.sample(obss)
        q1_new_action = self.critic1(obss, new_actions)
        q2_new_action = self.critic2(obss, new_actions)

        if torch.rand((), device=obss.device) > 0.5:
            q_loss = -q1_new_action.mean() / q2_new_action.abs().mean().detach().clamp_min(1e-6)
        else:
            q_loss = -q2_new_action.mean() / q1_new_action.abs().mean().detach().clamp_min(1e-6)

        actor_loss = bc_loss + self._eta * q_loss

        self.actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self._grad_norm > 0.0:
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self._grad_norm, norm_type=2)
        self.actor_optim.step()

        if self._step % self._ema_update_every == 0:
            self._step_actor_ema()
        self._soft_update(self.critic1_target, self.critic1)
        self._soft_update(self.critic2_target, self.critic2)
        self._step += 1

        if self.actor_lr_scheduler is not None:
            self.actor_lr_scheduler.step()
        if self.critic1_lr_scheduler is not None:
            self.critic1_lr_scheduler.step()
        if self.critic2_lr_scheduler is not None:
            self.critic2_lr_scheduler.step()

        return {
            "loss/actor": float(actor_loss.detach().cpu()),
            "loss/critic1": float(critic1_loss.detach().cpu()),
            "loss/critic2": float(critic2_loss.detach().cpu()),
            "loss/bc": float(bc_loss.detach().cpu()),
            "loss/q_guidance": float(q_loss.detach().cpu()),
            "misc/target_q": float(target_q.mean().detach().cpu()),
        }
