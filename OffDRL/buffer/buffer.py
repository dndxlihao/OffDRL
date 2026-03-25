import numpy as np
import torch
from typing import Tuple, Dict, Union

"""
ReplayBuffer (gymnasium style)

API:
- add(obs, next_obs, action, reward, terminated, truncated)
- add_batch(obss, next_obss, actions, rewards, terminateds, truncateds)
- load_dataset(dataset) requires keys 'terminated' and 'truncated'
- sample(batch_size) -> returns torch tensors (terminated/truncated as bool)
- sample_all() -> returns numpy arrays
"""

class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple[int, ...],
        obs_dtype: np.dtype,
        action_shape: Tuple[int, ...],
        action_dtype: np.dtype,
        device: str = "cpu"
    ) -> None:
        self._max_size = int(buffer_size)
        if isinstance(obs_shape, int):
            self.obs_shape = (obs_shape,)
        else:
            self.obs_shape = tuple(obs_shape)
        self.obs_dtype = obs_dtype

        if isinstance(action_shape, int):
            self.action_shape = (action_shape,)
        else:
            self.action_shape = tuple(action_shape)
        self.action_dtype = action_dtype

        self._ptr = 0
        self._size = 0

        self.observations = np.zeros((self._max_size, *self.obs_shape), dtype=self.obs_dtype)
        self.next_observations = np.zeros((self._max_size, *self.obs_shape), dtype=self.obs_dtype)
        self.actions = np.zeros((self._max_size, *self.action_shape), dtype=self.action_dtype)
        self.rewards = np.zeros((self._max_size,), dtype=np.float32)

        self.terminateds = np.zeros((self._max_size,), dtype=np.bool_)
        self.truncateds = np.zeros((self._max_size,), dtype=np.bool_)

        self.device = torch.device(device)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: Union[float, int, np.ndarray],
        terminated: Union[bool, np.ndarray],
        truncated: Union[bool, np.ndarray]
    ) -> None:

        self.observations[self._ptr] = np.array(obs).astype(self.obs_dtype).copy()
        self.next_observations[self._ptr] = np.array(next_obs).astype(self.obs_dtype).copy()
        self.actions[self._ptr] = np.array(action).astype(self.action_dtype).copy()
        self.rewards[self._ptr] = float(np.array(reward).reshape(-1)[0])

        term = bool(np.asarray(terminated).reshape(-1)[0])
        trunc = bool(np.asarray(truncated).reshape(-1)[0])

        self.terminateds[self._ptr] = term
        self.truncateds[self._ptr] = trunc

        self._ptr = (self._ptr + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)

    def add_batch(
        self,
        obss: np.ndarray,
        next_obss: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminateds: np.ndarray,
        truncateds: np.ndarray
    ) -> None:
        """
        Supported shapes for terminateds/truncateds: scalar, (batch,), (batch,1), (1,batch).
        """
        batch_size = len(obss)
        indexes = np.arange(self._ptr, self._ptr + batch_size) % self._max_size

        self.observations[indexes] = np.array(obss).astype(self.obs_dtype).copy()
        self.next_observations[indexes] = np.array(next_obss).astype(self.obs_dtype).copy()
        self.actions[indexes] = np.array(actions).astype(self.action_dtype).copy()
        self.rewards[indexes] = np.array(rewards).astype(np.float32).reshape(-1)

        def _to_1d_bool(x, name):
            arr = np.asarray(x)
            if arr.ndim == 0:
                return np.full((batch_size,), bool(arr), dtype=bool)
            if arr.ndim == 1:
                if arr.shape[0] == batch_size:
                    return arr.astype(bool)
                if arr.shape[0] == 1:
                    return np.full((batch_size,), bool(arr.reshape(-1)[0]), dtype=bool)
                raise ValueError(f"{name} length {arr.shape[0]} does not match batch_size {batch_size}")
            if arr.ndim == 2:
                if arr.shape[0] == batch_size and arr.shape[1] == 1:
                    return arr[:, 0].astype(bool)
                if arr.shape[1] == batch_size and arr.shape[0] == 1:
                    return arr.reshape(-1).astype(bool)
                raise ValueError(f"Unsupported {name} shape {arr.shape} for batch_size {batch_size}")
            raise ValueError(f"Unsupported {name} ndim {arr.ndim}")

        term = _to_1d_bool(terminateds, "terminateds")
        trunc = _to_1d_bool(truncateds, "truncateds")

        self.terminateds[indexes] = term
        self.truncateds[indexes] = trunc

        self._ptr = (self._ptr + batch_size) % self._max_size
        self._size = min(self._size + batch_size, self._max_size)

    def load_dataset(self, dataset: Dict[str, np.ndarray]) -> None:
    
        observations = np.array(dataset["observations"], dtype=self.obs_dtype)
        next_observations = np.array(dataset["next_observations"], dtype=self.obs_dtype)
        actions = np.array(dataset["actions"], dtype=self.action_dtype)
        rewards = np.array(dataset["rewards"], dtype=np.float32).reshape(-1)
        terminateds = np.array(dataset["terminateds"]).astype(bool).reshape(-1)
        truncateds = np.array(dataset["truncateds"]).astype(bool).reshape(-1)

        lengths = [
            observations.shape[0],
            next_observations.shape[0],
            actions.shape[0],
            rewards.shape[0],
            terminateds.shape[0],
            truncateds.shape[0]
        ]
        n = min(lengths)

        self.observations = observations[:n].copy()
        self.next_observations = next_observations[:n].copy()
        self.actions = actions[:n].copy()
        self.rewards = rewards[:n].copy()
        self.terminateds = terminateds[:n].copy()
        self.truncateds = truncateds[:n].copy()

        self._ptr = n
        self._size = n

    def normalize_obs(self, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
        mean = self.observations.mean(axis=0, keepdims=True)
        std = self.observations.std(axis=0, keepdims=True) + eps
        self.observations = (self.observations - mean) / std
        self.next_observations = (self.next_observations - mean) / std
        return mean, std

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        
        batch_indexes = np.random.randint(0, self._size, size=batch_size)

        obs = torch.from_numpy(self.observations[batch_indexes]).to(self.device)
        acts = torch.from_numpy(self.actions[batch_indexes]).to(self.device)
        next_obs = torch.from_numpy(self.next_observations[batch_indexes]).to(self.device)
        rews = torch.from_numpy(self.rewards[batch_indexes]).to(self.device)

        term_bool = self.terminateds[batch_indexes].astype(np.bool_)
        trunc_bool = self.truncateds[batch_indexes].astype(np.bool_)

        return {
            "observations": obs,
            "actions": acts,
            "next_observations": next_obs,
            "terminateds": torch.from_numpy(term_bool).to(self.device),
            "truncateds": torch.from_numpy(trunc_bool).to(self.device),
            "rewards": rews
        }

    def sample_all(self) -> Dict[str, np.ndarray]:
        return {
            "observations": self.observations[:self._size].copy(),
            "actions": self.actions[:self._size].copy(),
            "next_observations": self.next_observations[:self._size].copy(),
            "terminateds": self.terminateds[:self._size].copy(),
            "truncateds": self.truncateds[:self._size].copy(),
            "rewards": self.rewards[:self._size].copy()
        }