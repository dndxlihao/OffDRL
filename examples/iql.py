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
from OffDRL.modules import ActorProb, Critic, DiagGaussian
from OffDRL.buffer import ReplayBuffer
from OffDRL.utils.logger import Logger
from OffDRL.trainer import MFPolicyTrainer
from OffDRL.policy.model_free.iql import IQLPolicy

from utils import (
    load_offline_dataset, plot_curves,
    setup_logging_dirs, get_output_config, save_final_model
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="IQL")

    # env & data
    parser.add_argument("--env_id", type=str, default="ChargingEnv-v0")
    parser.add_argument("--dataset", type=str, default="data/mixed_dataset.h5")
    
    # seeds & device
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # model
    parser.add_argument("--hidden-dims", type=int, nargs='*', default=[256, 256, 256])
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    
    # IQL
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-q-lr", type=float, default=3e-4)
    parser.add_argument("--critic-v-lr", type=float, default=3e-4)
    parser.add_argument("--lr-decay", action="store_true", default=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--expectile", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=3.0)

    # schedule / eval
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)

    return parser.parse_args()


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
    actor_backbone = MLP(input_dim=int(np.prod(obs_shape)), hidden_dims=args.hidden_dims, dropout_rate=args.dropout_rate)
    critic_q1_backbone = MLP(input_dim=int(np.prod(obs_shape)) + action_dim, hidden_dims=args.hidden_dims)
    critic_q2_backbone = MLP(input_dim=int(np.prod(obs_shape)) + action_dim, hidden_dims=args.hidden_dims)
    critic_v_backbone  = MLP(input_dim=int(np.prod(obs_shape)), hidden_dims=args.hidden_dims)

    dist = DiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=action_dim,
        unbounded=False,
        conditioned_sigma=False,
        max_mu=max_action
    )
    actor    = ActorProb(actor_backbone, dist, args.device)
    critic_q1 = Critic(critic_q1_backbone, args.device)
    critic_q2 = Critic(critic_q2_backbone, args.device)
    critic_v  = Critic(critic_v_backbone,  args.device)

    # init
    for m in list(actor.modules()) + list(critic_q1.modules()) + list(critic_q2.modules()) + list(critic_v.modules()):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.zeros_(m.bias)

    # opt
    actor_optim     = torch.optim.Adam(actor.parameters(),     lr=args.actor_lr)
    critic_q1_optim = torch.optim.Adam(critic_q1.parameters(), lr=args.critic_q_lr)
    critic_q2_optim = torch.optim.Adam(critic_q2.parameters(), lr=args.critic_q_lr)
    critic_v_optim  = torch.optim.Adam(critic_v.parameters(),  lr=args.critic_v_lr)

    # lr scheduler 
    if args.lr_decay:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor_optim, args.epoch)
    else:
        lr_scheduler = None

    # policy
    policy = IQLPolicy(
        actor=actor,
        critic_q1=critic_q1,
        critic_q2=critic_q2,
        critic_v=critic_v,
        actor_optim=actor_optim,
        critic_q1_optim=critic_q1_optim,
        critic_q2_optim=critic_q2_optim,
        critic_v_optim=critic_v_optim,
        action_space=env.action_space,
        tau=args.tau,
        gamma=args.gamma,
        expectile=args.expectile,
        temperature=args.temperature,
    )

    # buffer
    buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=obs_shape,
        obs_dtype=np.float32,
        action_shape=(action_dim,),
        action_dtype=np.float32,
        device=args.device
    )
    buffer.load_dataset(dataset) 

    # logging
    model_dir, logs_dir = setup_logging_dirs("IQL")
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
        lr_scheduler=lr_scheduler
    )

    trainer.train()

    # save & plot
    save_final_model(policy, model_dir)
    plot_curves(logs_dir, "IQL")
    logger.close()


if __name__ == "__main__":
    train()
