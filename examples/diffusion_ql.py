import argparse
import random

import gymnasium as gym
import numpy as np
import torch

from utils import setup_project_paths

setup_project_paths()
import chargym

from OffDRL.backbone import MLP
from OffDRL.buffer import ReplayBuffer
from OffDRL.kernel import DiffusionKernel
from OffDRL.modules import Critic
from OffDRL.policy.generative_rl import DiffusionQLPolicy
from OffDRL.trainer import MFPolicyTrainer
from OffDRL.utils.logger import Logger

from utils import (
    get_output_config,
    load_offline_dataset,
    plot_curves,
    save_final_model,
    setup_logging_dirs,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="DiffusionQL")

    parser.add_argument("--env_id", type=str, default="ChargingEnv-v0")
    parser.add_argument("--dataset", type=str, default="data/mixed_dataset.h5")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--hidden-dims", type=int, nargs="*", default=[256, 256, 256])
    parser.add_argument("--diffusion-hidden", type=int, default=256)
    parser.add_argument("--diffusion-time-dim", type=int, default=16)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--beta-schedule", type=str, default="linear", choices=["linear", "cosine", "vp"])

    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--grad-norm", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)

    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--ema-start-step", type=int, default=1000)
    parser.add_argument("--ema-update-every", type=int, default=5)
    parser.add_argument("--max-q-backup", action="store_true")
    parser.add_argument("--num-backup-samples", type=int, default=10)
    parser.add_argument("--num-action-samples", type=int, default=50)

    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)

    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def build_critics(args, obs_dim: int, act_dim: int):
    critic1 = Critic(
        MLP(input_dim=obs_dim + act_dim, hidden_dims=args.hidden_dims),
        args.device,
    )
    critic2 = Critic(
        MLP(input_dim=obs_dim + act_dim, hidden_dims=args.hidden_dims),
        args.device,
    )
    return critic1, critic2


def train(args=get_args()):
    env = gym.make(args.env_id)
    env.reset(seed=args.seed)

    dataset = load_offline_dataset(args.dataset)

    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    action_low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    action_high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    max_action = float(max(np.max(np.abs(action_low)), np.max(np.abs(action_high))))

    seed_everything(args.seed)

    critic1, critic2 = build_critics(args, obs_dim, act_dim)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    actor = DiffusionKernel(
        obs_dim=obs_dim,
        action_dim=act_dim,
        max_action=max_action,
        hidden_dim=args.diffusion_hidden,
        time_dim=args.diffusion_time_dim,
        beta_schedule=args.beta_schedule,
        n_timesteps=args.diffusion_steps,
        action_low=action_low,
        action_high=action_high,
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    policy = DiffusionQLPolicy(
        actor=actor,
        critic1=critic1,
        critic2=critic2,
        actor_optim=actor_optim,
        critic1_optim=critic1_optim,
        critic2_optim=critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        eta=args.eta,
        max_action=max_action,
        max_q_backup=args.max_q_backup,
        num_backup_samples=args.num_backup_samples,
        num_action_samples=args.num_action_samples,
        ema_decay=args.ema_decay,
        ema_start_step=args.ema_start_step,
        ema_update_every=args.ema_update_every,
        grad_norm=args.grad_norm,
    )

    buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=env.observation_space.shape,
        obs_dtype=np.float32,
        action_shape=(act_dim,),
        action_dtype=np.float32,
        device=args.device,
    )
    buffer.load_dataset(dataset)

    model_dir, logs_dir = setup_logging_dirs(args.algo_name)
    logger = Logger(str(logs_dir), get_output_config())
    logger.log_hyperparameters(vars(args))

    trainer = MFPolicyTrainer(
        policy=policy,
        eval_env=env,
        buffer=buffer,
        logger=logger,
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        batch_size=args.batch_size,
        eval_episodes=args.eval_episodes,
    )
    trainer.train()

    save_final_model(policy, model_dir)
    plot_curves(logs_dir, args.algo_name)
    logger.close()


if __name__ == "__main__":
    train()
