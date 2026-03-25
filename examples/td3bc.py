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
from OffDRL.modules import Actor, Critic   
from OffDRL.buffer import ReplayBuffer
from OffDRL.utils.logger import Logger
from OffDRL.trainer import MFPolicyTrainer
from OffDRL.policy.model_free.td3bc import TD3BCPolicy

from utils import (
    load_offline_dataset, plot_curves,
    setup_logging_dirs, get_output_config, save_final_model
)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo-name", type=str, default="TD3BC")

    # env & dataset
    p.add_argument("--env_id", type=str, default="ChargingEnv-v0")
    p.add_argument("--dataset", type=str, default="data/mixed_dataset.h5")

    # seed & device
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # model
    p.add_argument("--hidden-dims", type=int, nargs="*", default=[256, 256])

    # optim
    p.add_argument("--actor-lr", type=float, default=3e-4)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--lr-decay", action="store_true", default=False)

    # td3bc hyper-params
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--alpha", type=float, default=2.5)
    p.add_argument("--policy-noise", type=float, default=0.1) 
    p.add_argument("--noise-clip", type=float, default=0.2)
    p.add_argument("--policy-freq", type=int, default=2)

    # training schedule
    p.add_argument("--epoch", type=int, default=200)
    p.add_argument("--step-per-epoch", type=int, default=1000)
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    return p.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train(args=get_args()):
    # env
    env = gym.make(args.env_id)
    env.reset(seed=args.seed)

    # dataset
    dataset = load_offline_dataset(args.dataset)

    # info
    obs_shape = env.observation_space.shape
    action_dim = int(np.prod(env.action_space.shape))
    max_action = float(env.action_space.high[0])

    # seeds
    seed_everything(args.seed)

    # models
    actor_backbone = MLP(int(np.prod(obs_shape)), args.hidden_dims)
    actor = Actor(actor_backbone, action_dim, max_action, args.device)

    critic1_backbone = MLP(int(np.prod(obs_shape)) + action_dim, args.hidden_dims)
    critic2_backbone = MLP(int(np.prod(obs_shape)) + action_dim, args.hidden_dims)
    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)

    # init 
    for m in list(actor.modules()) + list(critic1.modules()) + list(critic2.modules()):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.zeros_(m.bias)

    # opt
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    # lr scheduler
    if args.lr_decay:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor_optim, args.epoch)
    else:
        lr_scheduler = None

    # policy
    policy = TD3BCPolicy(
        actor=actor,
        critic1=critic1,
        critic2=critic2,
        actor_optim=actor_optim,
        critic1_optim=critic1_optim,
        critic2_optim=critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        max_action=max_action,
        alpha=args.alpha,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        update_actor_freq=args.policy_freq,               
    )

    # buffer
    buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=obs_shape,
        obs_dtype=np.float32,
        action_shape=(action_dim,),
        action_dtype=np.float32,
        device=args.device,
    )
    buffer.load_dataset(dataset)  

    # logging
    model_dir, logs_dir = setup_logging_dirs("TD3BC")
    logger = Logger(str(logs_dir), get_output_config())
    logger.log_hyperparameters(vars(args))

    # trainer
    trainer = MFPolicyTrainer(
        policy=policy,
        eval_env=env,
        buffer=buffer,
        logger=logger,
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        batch_size=args.batch_size,
        eval_episodes=args.eval_episodes,
        lr_scheduler=lr_scheduler,
    )

    trainer.train()

    # save & plot
    save_final_model(policy, model_dir)
    plot_curves(logs_dir, "TD3BC")
    logger.close()


if __name__ == "__main__":
    train()
