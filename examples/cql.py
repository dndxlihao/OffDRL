import argparse
import random
import numpy as np
import torch
import gymnasium as gym

from utils import setup_project_paths
PROJECT_ROOT = setup_project_paths()

import chargym  

from OffDRL.backbone import MLP
from OffDRL.modules import ActorProb, Critic, TanhDiagGaussian
from OffDRL.buffer import ReplayBuffer
from OffDRL.utils.logger import Logger
from OffDRL.trainer import MFPolicyTrainer
from OffDRL.policy.model_free.cql import CQLPolicy

from utils import (
    load_offline_dataset, plot_curves,
    setup_logging_dirs, get_output_config, save_final_model
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo_name", type=str, default="CQL")

    # env & data
    parser.add_argument("--env_id", type=str, default="ChargingEnv-v0")
    parser.add_argument("--dataset", type=str, default="data/mixed_dataset.h5")
    
    # seeds & device
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # model
    parser.add_argument("--hidden_dims", type=int, nargs='*', default=[256, 256, 256])
    parser.add_argument("--actor_lr", type=float, default=1e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)

    # SAC
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--target_entropy", type=float, default=None)
    parser.add_argument("--auto_alpha", action="store_true", default=True)
    parser.add_argument("--alpha_lr", type=float, default=1e-4)

    # CQL
    parser.add_argument("--cql_weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_q_backup", action="store_true", default=False)
    parser.add_argument("--deterministic_backup", type=lambda s: s.lower() in ("true", "1", "yes"), default=True)
    parser.add_argument("--with_lagrange", type=lambda s: s.lower() in ("true", "1", "yes"), default=False)
    parser.add_argument("--lagrange_threshold", type=float, default=2.0)
    parser.add_argument("--cql_alpha_lr", type=float, default=3e-4)
    parser.add_argument("--num_repeat_actions", type=int, default=10)

    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--step_per_epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument("--lr_decay", action="store_true", default=False)

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
    obs_dim = int(np.prod(obs_shape))
    actor_backbone = MLP(input_dim=obs_dim, hidden_dims=args.hidden_dims)
    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=action_dim,
        unbounded=True,
        conditioned_sigma=True,
        max_mu=max_action
    )
    actor = ActorProb(actor_backbone, dist, args.device)

    critic1_backbone = MLP(input_dim=obs_dim + action_dim, hidden_dims=args.hidden_dims)
    critic2_backbone = MLP(input_dim=obs_dim + action_dim, hidden_dims=args.hidden_dims)
    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)

    # init  
    for net in [actor_backbone, critic1_backbone, critic2_backbone]:
        for m in net.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                torch.nn.init.zeros_(m.bias)

    # opt
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    # entropy coeff
    if args.auto_alpha:
        target_entropy = args.target_entropy if args.target_entropy is not None else -float(action_dim)
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (float(target_entropy), log_alpha, alpha_optim)
    else:
        alpha = float(args.alpha)

    # policy
    policy = CQLPolicy(
        actor=actor,
        critic1=critic1,
        critic2=critic2,
        actor_optim=actor_optim,
        critic1_optim=critic1_optim,
        critic2_optim=critic2_optim,
        action_space=env.action_space,
        tau=args.tau,
        gamma=args.gamma,
        alpha=alpha,
        cql_weight=args.cql_weight,
        temperature=args.temperature,
        max_q_backup=args.max_q_backup,
        deterministic_backup=bool(args.deterministic_backup),
        with_lagrange=bool(args.with_lagrange),
        lagrange_threshold=args.lagrange_threshold,
        cql_alpha_lr=args.cql_alpha_lr,
        num_repeat_actions=args.num_repeat_actions,
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
    model_dir, logs_dir = setup_logging_dirs(args.algo_name)
    logger = Logger(str(logs_dir), get_output_config())
    logger.log_hyperparameters(vars(args))

    # trainer
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor_optim, args.epoch) if args.lr_decay else None
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
    plot_curves(logs_dir, args.algo_name)
    logger.close()


if __name__ == "__main__":
    train()
