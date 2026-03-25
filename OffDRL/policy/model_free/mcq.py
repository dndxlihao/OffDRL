import torch
import torch.nn.functional as F
from OffDRL.policy.model_free.sac import SACPolicy

class MCQPolicy(SACPolicy):
    def __init__(
        self,
        *args,
        behavior_policy,
        behavior_policy_optim,
        num_sampled_actions: int = 10,
        lmbda: float = 0.75,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.behavior_policy = behavior_policy
        self.behavior_policy_optim = behavior_policy_optim
        self._num_sampled_actions = int(num_sampled_actions)
        self._lmbda = float(lmbda)

    def learn(self, batch):
        obss        = batch["observations"]
        actions     = batch["actions"]
        next_obss   = batch["next_observations"]
        rewards     = batch["rewards"]
        terminateds = batch["terminateds"]
        truncateds  = batch["truncateds"]

        # ---------------- Behavior policy update (VAE) ----------------
        recon, mean, std = self.behavior_policy(obss, actions)
        recon_loss = F.mse_loss(recon, actions)
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + KL_loss
        self.behavior_policy_optim.zero_grad(set_to_none=True)
        vae_loss.backward()
        self.behavior_policy_optim.step()

        # ---------------- Critic targets  ----------------
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
            target_q_in = rewards + self._gamma * cont_mask * next_q

        q1_in, q2_in = self.critic1(obss, actions), self.critic2(obss, actions)
        loss_c1_in = F.mse_loss(q1_in, target_q_in)
        loss_c2_in = F.mse_loss(q2_in, target_q_in)

        # ---------------- Critic targets (OOD via VAE samples) ----------------
        s_in = torch.cat([obss, next_obss], dim=0)
        with torch.no_grad():
            s_rep = torch.repeat_interleave(s_in, self._num_sampled_actions, dim=0)
            sampled_actions = self.behavior_policy.decode(s_rep)
            tq1 = self.critic1_old(s_rep, sampled_actions).reshape(s_in.shape[0], -1).max(1, keepdim=True)[0]
            tq2 = self.critic2_old(s_rep, sampled_actions).reshape(s_in.shape[0], -1).max(1, keepdim=True)[0]
            target_q_ood = torch.min(tq1, tq2)

        q1_ood_all = self.critic1(s_rep, sampled_actions).reshape(s_in.shape[0], -1)
        q2_ood_all = self.critic2(s_rep, sampled_actions).reshape(s_in.shape[0], -1)
        q1_ood = q1_ood_all.max(1, keepdim=True)[0]
        q2_ood = q2_ood_all.max(1, keepdim=True)[0]

        loss_c1_ood = F.mse_loss(q1_ood, target_q_ood)
        loss_c2_ood = F.mse_loss(q2_ood, target_q_ood)

        critic1_loss = self._lmbda * loss_c1_in + (1 - self._lmbda) * loss_c1_ood
        critic2_loss = self._lmbda * loss_c2_in + (1 - self._lmbda) * loss_c2_ood

        self.critic1_optim.zero_grad(set_to_none=True)
        critic1_loss.backward()
        self.critic1_optim.step()

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

        # ---------------- Alpha update ----------------
        if self._is_auto_alpha:
            alpha_loss = -(self._log_alpha * (log_probs + self._target_entropy).detach()).mean()
            self.alpha_optim.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp().clamp_min(1e-8)

        # ---------------- Soft update target networks ----------------
        self._grad_steps += 1
        if self._grad_steps % self._target_update_interval == 0:
            self._sync_weight()

        # ---------------- Return logs ----------------
        result = {
            "loss/actor":   float(actor_loss.detach().cpu()),
            "loss/critic1": float(critic1_loss.detach().cpu()),
            "loss/critic2": float(critic2_loss.detach().cpu()),
            "loss/behavior_policy": float(vae_loss.detach().cpu())
        }
        if self._is_auto_alpha:
            result["loss/alpha"] = float(alpha_loss.detach().cpu())
            result["alpha"] = float(self._alpha.detach().cpu())
        return result
