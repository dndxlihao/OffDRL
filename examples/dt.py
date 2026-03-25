# dt.py —— Decision Transformer offline trainer (td3bc-style + tqdm progress bars)
import argparse
import os
import random
from typing import Dict, List
from itertools import cycle

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import gymnasium as gym
import h5py
from tqdm import tqdm  # 进度条

from utils import setup_project_paths
setup_project_paths()
import chargym  # 确保环境已注册

from utils import (
    setup_logging_dirs, get_output_config, save_final_model, plot_curves
)
from OffDRL.utils.logger import Logger

from OffDRL.policy.transformer_rl.dt import DecisionTransformer


# ---------------------------
# Args
# ---------------------------
def get_args():
    p = argparse.ArgumentParser(description="Decision Transformer")
    # env & dataset
    p.add_argument("--env_id", type=str, default="ChargingEnv-v0")
    p.add_argument("--dataset", type=str, default="data/dt_dataset.h5")
    # seed & device
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # model
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--context-length", type=int, default=20, help="K")
    p.add_argument("--action-tanh", action="store_true", default=True)
    # optim
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--grad-clip", type=float, default=1.0)
    # td3bc-style schedule
    p.add_argument("--steps-per-episode", type=int, default=200, help="每个训练episode内的梯度步数")
    p.add_argument("--total-episodes", type=int, default=200, help="训练多少个episode（外层循环次数）")
    # eval
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--target-return", type=float, default=-21.0, help="评估时的固定目标回报")
    # misc
    p.add_argument("--algo-name", type=str, default="DT")
    return p.parse_args()


# ---------------------------
# Utils
# ---------------------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_h5(path: str) -> Dict[str, np.ndarray]:
    assert os.path.exists(path), f"Dataset not found: {path}"
    with h5py.File(path, "r") as f:
        data = {k: f[k][:] for k in f.keys()}
    required = ["observations", "actions", "returns_to_go", "episode_id", "timestep"]
    missing = [k for k in required if k not in data]
    assert not missing, f"dataset 缺少字段: {missing}"
    return data


class DTSequenceDataset(Dataset):
    """
    从连续 episode 中抽取固定长度 K 的窗口；不跨 episode。
    需要：observations, actions, returns_to_go, episode_id, timestep
    """
    def __init__(self, data: Dict[str, np.ndarray], K: int):
        self.obs = data["observations"].astype(np.float32)
        self.acts = data["actions"].astype(np.float32)
        rtg = data["returns_to_go"].astype(np.float32)
        if rtg.ndim == 1:
            rtg = rtg[:, None]
        self.rtg = rtg
        self.eid = data["episode_id"].astype(np.int32)
        self.t   = data["timestep"].astype(np.int32)

        assert self.obs.shape[0] == self.acts.shape[0] == self.rtg.shape[0] == self.eid.shape[0] == self.t.shape[0]
        self.K = int(K)

        self.indices: List[int] = []
        N = len(self.eid)
        for i in range(0, N - self.K + 1):
            if self.eid[i] == self.eid[i + self.K - 1] and (self.t[i + self.K - 1] - self.t[i] + 1) == self.K:
                self.indices.append(i)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        i = self.indices[idx]
        j = i + self.K
        states = torch.from_numpy(self.obs[i:j])                  # (K, state_dim)
        actions = torch.from_numpy(self.acts[i:j])                # (K, action_dim)
        returns = torch.from_numpy(self.rtg[i:j]).reshape(-1, 1)  # (K, 1)
        timesteps = torch.from_numpy(self.t[i:j]).long()          # (K,)
        return {"states": states, "actions": actions, "returns_to_go": returns, "timesteps": timesteps}


@torch.no_grad()
def evaluate_rollouts(env, model: DecisionTransformer, K: int, target_return: float,
                      device: torch.device, episodes: int = 5) -> Dict[str, float]:
    model.eval()
    ep_returns = []
    max_action = float(env.action_space.high[0]) if hasattr(env.action_space, "high") else 1.0

    for _ in tqdm(range(episodes), desc="Eval Episodes", leave=False):
        obs, _ = env.reset()
        obs = obs.astype(np.float32)
        done = False

        states_hist  = [obs]
        actions_hist = [np.zeros_like(env.action_space.sample(), dtype=np.float32)]
        rtg_hist     = [float(target_return)]
        t_hist       = [0]
        ep_ret = 0.0

        while not done:
            # 组装 K 长度窗口（episode 开头左侧用 0 填充）
            def _pad(seq, pad_val, shape):
                out = [pad_val] * max(0, K - len(seq)) + seq[-K:]
                return np.stack(out, axis=0).reshape((K,) + shape) if shape is not None else np.array(out, dtype=np.float32).reshape(K)

            states_win  = _pad(states_hist, np.zeros_like(states_hist[0], dtype=np.float32), shape=states_hist[0].shape)
            actions_win = _pad(actions_hist, np.zeros_like(actions_hist[0], dtype=np.float32), shape=actions_hist[0].shape)
            rtg_win     = _pad(rtg_hist, 0.0, shape=())[:, None]  # (K,1)
            t_win       = _pad(t_hist, 0, shape=None).astype(np.int64)

            states_t  = torch.from_numpy(states_win).unsqueeze(0).to(device)   # (1,K,sd)
            actions_t = torch.from_numpy(actions_win).unsqueeze(0).to(device)  # (1,K,ad)
            rtg_t     = torch.from_numpy(rtg_win).unsqueeze(0).to(device)      # (1,K,1)
            t_t       = torch.from_numpy(t_win).unsqueeze(0).to(device)        # (1,K)

            act = model.predict_next_action(states_t, actions_t, rtg_t, t_t, max_action=max_action)[0].cpu().numpy()
            next_obs, reward, terminated, truncated, _ = env.step(act.astype(np.float32))
            done = bool(terminated) or bool(truncated)

            ep_ret += float(reward)
            states_hist.append(next_obs.astype(np.float32))
            actions_hist.append(act.astype(np.float32))
            t_hist.append(t_hist[-1] + 1)
            rtg_hist.append(rtg_hist[-1] - float(reward))

        ep_returns.append(ep_ret)

    return {
        "eval/episodes": float(episodes),
        "eval/return_mean": float(np.mean(ep_returns)),
        "eval/return_std": float(np.std(ep_returns)),
        "eval/return_min": float(np.min(ep_returns)),
        "eval/return_max": float(np.max(ep_returns)),
    }@torch.no_grad()
def evaluate_rollouts(env, model: DecisionTransformer, K: int, target_return: float,
                      device: torch.device, episodes: int = 5) -> Dict[str, float]:
    model.eval()
    ep_returns = []
    max_action = float(env.action_space.high[0]) if hasattr(env.action_space, "high") else 1.0

    for _ in range(episodes):
        obs, _ = env.reset()
        obs = obs.astype(np.float32)
        done = False

        states_hist  = [obs]
        actions_hist = [np.zeros_like(env.action_space.sample(), dtype=np.float32)]
        rtg_hist     = [float(target_return)]
        t_hist       = [0]
        ep_ret = 0.0

        while not done:
            def _pad(seq, pad_val, shape):
                out = [pad_val] * max(0, K - len(seq)) + seq[-K:]
                return np.stack(out, axis=0).reshape((K,) + shape) if shape is not None else np.array(out, dtype=np.float32).reshape(K)

            states_win  = _pad(states_hist, np.zeros_like(states_hist[0], dtype=np.float32), shape=states_hist[0].shape)
            actions_win = _pad(actions_hist, np.zeros_like(actions_hist[0], dtype=np.float32), shape=actions_hist[0].shape)
            rtg_win     = _pad(rtg_hist, 0.0, shape=())[:, None]  # (K,1)
            t_win       = _pad(t_hist, 0, shape=None).astype(np.int64)

            # ✨ 关键：强制用 float32
            states_t  = torch.from_numpy(states_win).unsqueeze(0).to(device=device, dtype=torch.float32)   # (1,K,sd)
            actions_t = torch.from_numpy(actions_win).unsqueeze(0).to(device=device, dtype=torch.float32)  # (1,K,ad)
            rtg_t     = torch.from_numpy(rtg_win).unsqueeze(0).to(device=device, dtype=torch.float32)      # (1,K,1)
            t_t       = torch.from_numpy(t_win).unsqueeze(0).to(device)                                    # (1,K) long/int64 OK

            act = model.predict_next_action(states_t, actions_t, rtg_t, t_t, max_action=max_action)[0].cpu().numpy()
            next_obs, reward, terminated, truncated, _ = env.step(act.astype(np.float32))
            done = bool(terminated) or bool(truncated)

            ep_ret += float(reward)
            states_hist.append(next_obs.astype(np.float32))
            actions_hist.append(act.astype(np.float32))
            t_hist.append(t_hist[-1] + 1)
            rtg_hist.append(rtg_hist[-1] - float(reward))

        ep_returns.append(ep_ret)

    return {
        "eval/episodes": float(episodes),
        "eval/return_mean": float(np.mean(ep_returns)),
        "eval/return_std": float(np.std(ep_returns)),
        "eval/return_min": float(np.min(ep_returns)),
        "eval/return_max": float(np.max(ep_returns)),
    }

def train(args: argparse.Namespace):
    # env
    env = gym.make(args.env_id)
    env.reset(seed=args.seed)

    # dataset
    print(">>> Using dataset:", os.path.abspath(args.dataset))
    dataset = load_h5(args.dataset)
    print(">>> Keys:", sorted(list(dataset.keys())))
    print(">>> returns_to_go shape:", dataset["returns_to_go"].shape)

    obs_shape = env.observation_space.shape
    action_dim = int(np.prod(env.action_space.shape))

    # seeds/device
    seed_everything(args.seed)
    device = torch.device(args.device)

    # dataloader（固定长度 K，不跨 episode）
    K = int(args.context_length)
    ds = DTSequenceDataset(dataset, K=K)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    dl_iter = cycle(dl)  

    # model
    model = DecisionTransformer(
        state_dim=int(np.prod(obs_shape)),
        action_dim=action_dim,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        max_timestep=4096,
        action_tanh=args.action_tanh,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model_dir, logs_dir = setup_logging_dirs(args.algo_name)
    logger = Logger(str(logs_dir), get_output_config())

    global_step = 0
    print(f"[DT] device={device} windows={len(ds)} batches/epoch≈{len(dl)} steps_per_episode={args.steps_per_episode}")

    outer_bar = tqdm(range(1, args.total_episodes + 1), desc="Train Episodes")
    for ep in outer_bar:
        model.train()
        train_losses = []

        # 内层训练步进度条
        inner_bar = tqdm(range(args.steps_per_episode), desc=f"Train Steps (Ep {ep})", leave=False)
        for _ in inner_bar:
            batch = next(dl_iter)
            states  = batch["states"].to(device)            # (B,K,sd)
            actions = batch["actions"].to(device)           # (B,K,ad)
            returns = batch["returns_to_go"].to(device)     # (B,K,1)
            timests = batch["timesteps"].to(device)         # (B,K)

            pred_actions, _ = model(states, actions, returns, timests)
            loss = ((pred_actions - actions) ** 2).mean()

            optim.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()

            global_step += 1
            cur_loss = float(loss.detach().cpu())
            train_losses.append(cur_loss)
            # 在进度条上显示当前 loss
            inner_bar.set_postfix(loss=f"{cur_loss:.4f}")

        # 训练一个“episode”结束：汇总并打印/记录训练loss
        train_mse = float(np.mean(train_losses))
        logger.logkv("episode", ep)
        logger.logkv("timestep", global_step)
        logger.logkv("loss/train_mse", train_mse)
        logger.dumpkvs()
        print(f"[DT][Episode {ep:03d}] steps={args.steps_per_episode} train_mse={train_mse:.6f}")

        # 紧接着做评估并打印/记录（带进度条）
        eval_stats = evaluate_rollouts(env, model, K, args.target_return, device, episodes=args.eval_episodes)
        for k, v in eval_stats.items():
            logger.logkv(k, v)
        logger.logkv("episode", ep)
        logger.logkv("timestep", global_step)
        logger.dumpkvs()
        print(f"[DT][Eval @ Episode {ep:03d}] {eval_stats}")

        # 在外层进度条上也显示最近一次训练/评估结果
        outer_bar.set_postfix(train_mse=f"{train_mse:.4f}", ret_mean=f"{eval_stats['eval/return_mean']:.3f}")

    # save & plot
    save_final_model(model, model_dir, filename="decision_transformer.pt")
    plot_curves(logs_dir, args.algo_name)
    # logger.close()  # 如你的 Logger 需要的话再启用


if __name__ == "__main__":
    args = get_args()
    train(args)
