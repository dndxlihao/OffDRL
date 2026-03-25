import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Optional
from OffDRL.policy import BasePolicy


class BCPolicy(BasePolicy):

    def __init__(
        self,
        actor: nn.Module,
        actor_optim: torch.optim.Optimizer,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.actor_optim = actor_optim

    def train(self) -> None:
        self.actor.train()

    def eval(self) -> None:
        self.actor.eval()

    def select_action(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
 
        device = next(self.actor.parameters()).device
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            act_t = self.actor(obs_t)
        return act_t.cpu().numpy().flatten()

    def learn(self, batch: Dict) -> Dict[str, float]:

        obss   = batch["observations"]
        acts   = batch["actions"]

        pred = self.actor(obss)
        actor_loss = F.mse_loss(pred, acts)

        self.actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optim.step()

        return {"loss/actor": float(actor_loss.detach().cpu())}
