import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

from utils import setup_project_paths
setup_project_paths()
import chargym  

from OffDRL.backbone import MLP
from OffDRL.modules import Actor
from OffDRL.buffer import ReplayBuffer
from OffDRL.utils.logger import Logger
from OffDRL.trainer import MFPolicyTrainer
from OffDRL.policy.model_free.bc import BCPolicy

from utils import (
    load_offline_dataset, plot_curves,
    setup_logging_dirs, get_output_config, save_final_model
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="BC")

    # env & data
    parser.add_argument("--env_id", type=str, default="ChargingEnv-v0")
    parser.add_argument("--dataset", type=str, default="data/mixed_dataset.h5")

    # seeds & device
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # model
    parser.add_argument("--hidden-dims", type=int, nargs='*', default=[256, 256, 256])
    parser.add_argument("--dropout_rate", type=float, default=0.1)

    # optim
    parser.add_argument("--actor-lr", type=float, default=2e-4)
    parser.add_argument("--lr-decay", action="store_true", default=True)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    # schedule
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=20)

    # data loading
    parser.add_argument("--batch-size", type=int, default=256)

    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train(args=get_args()):
    # Create environment
    env = gym.make(args.env_id)
    env.reset(seed=args.seed)

    # Load dataset
    dataset = load_offline_dataset(args.dataset)

    # Setup env info
    obs_shape = env.observation_space.shape
    action_dim = int(np.prod(env.action_space.shape))
    max_action = float(env.action_space.high[0])

    # Seeds
    seed_everything(args.seed)

    # Model
    actor_backbone = MLP(
        input_dim=int(np.prod(obs_shape)),
        hidden_dims=args.hidden_dims,
        dropout_rate=args.dropout_rate
    )
    actor = Actor(
        backbone=actor_backbone,
        action_dim=action_dim,
        max_action=max_action,
        device=args.device
    )
    for m in list(actor.modules()):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.zeros_(m.bias)

    # Optimizer 
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr,
                                   weight_decay=max(0.0, float(args.weight_decay)))

    # LR scheduler
    if args.lr_decay:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor_optim, args.epoch)
    else:
        lr_scheduler = None

    # Policy
    policy = BCPolicy(actor=actor, actor_optim=actor_optim)

    # Buffer
    buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=obs_shape,
        obs_dtype=np.float32,
        action_shape=(action_dim,),
        action_dtype=np.float32,
        device=args.device
    )
    buffer.load_dataset(dataset)

    # Logging
    model_dir, logs_dir = setup_logging_dirs("BC")
    logger = Logger(str(logs_dir), get_output_config())
    logger.log_hyperparameters(vars(args))

    # Trainer
    policy_trainer = MFPolicyTrainer(
        policy=policy,
        eval_env=env,
        buffer=buffer,
        logger=logger,
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        batch_size=args.batch_size,
        eval_episodes=args.eval_episodes,
        lr_scheduler=lr_scheduler
    )

    policy_trainer.train()

    # Save & plot
    save_final_model(policy, model_dir)
    plot_curves(logs_dir, "BC")
    logger.close()


if __name__ == "__main__":
    train()
