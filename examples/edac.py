import argparse
import random
import numpy as np
import torch
import gymnasium as gym

from utils import setup_project_paths
PROJECT_ROOT = setup_project_paths()

import chargym  

from OffDRL.backbone import MLP
from OffDRL.modules import ActorProb, EnsembleCritic, TanhDiagGaussian
from OffDRL.buffer import ReplayBuffer
from OffDRL.utils.logger import Logger
from OffDRL.trainer import MFPolicyTrainer
from OffDRL.policy import EDACPolicy

from utils import (
    load_offline_dataset, plot_curves,
    setup_logging_dirs, get_output_config, save_final_model
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo_name", type=str, default="EDAC")

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

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)

    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--target_entropy", type=float, default=None)
    parser.add_argument("--auto_alpha", action="store_true", default=True)
    parser.add_argument("--alpha_lr", type=float, default=1e-4)

    # EDAC
    parser.add_argument("--num_critics", type=int, default=50)          
    parser.add_argument("--max_q_backup", action="store_true", default=False)
    parser.add_argument("--deterministic_backup", action="store_true", default=False)
    parser.add_argument("--eta", type=float, default=1.0)               

    parser.add_argument("--epoch", type=int, default=200)          
    parser.add_argument("--step_per_epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)

    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train(args=get_args()):

    env = gym.make(args.env_id)
    env.reset(seed=args.seed)

    dataset = load_offline_dataset(args.dataset)   

    obs_shape = env.observation_space.shape
    action_dim = int(np.prod(env.action_space.shape))
    max_action = float(env.action_space.high[0])

    seed_everything(args.seed)

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

    # Ensemble Critics
    critics = EnsembleCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=args.hidden_dims,
        num_ensemble=args.num_critics,
        device=args.device
    )
    for layer in critics.model[::2]:
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.constant_(layer.bias, 0.1)
    if isinstance(critics.model[-1], torch.nn.Linear):
        torch.nn.init.uniform_(critics.model[-1].weight, -3e-3, 3e-3)
        torch.nn.init.uniform_(critics.model[-1].bias,   -3e-3, 3e-3)

    actor_optim   = torch.optim.Adam(actor.parameters(),   lr=args.actor_lr)
    critics_optim = torch.optim.Adam(critics.parameters(), lr=args.critic_lr)

    # Temperature α
    if args.auto_alpha:
        target_entropy = args.target_entropy if args.target_entropy is not None else -float(action_dim)
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = float(args.alpha)
 
    policy = EDACPolicy(
        actor=actor,
        critics=critics,
        actor_optim=actor_optim,
        critics_optim=critics_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=alpha,
        max_q_backup=args.max_q_backup,
        deterministic_backup=args.deterministic_backup,
        eta=args.eta
    )

    buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=obs_shape,
        obs_dtype=np.float32,
        action_shape=(action_dim,),
        action_dtype=np.float32,
        device=args.device
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
        eval_episodes=args.eval_episodes
    )

    trainer.train()

    # save & plot
    save_final_model(policy, model_dir)
    plot_curves(logs_dir, args.algo_name)
    logger.close()


if __name__ == "__main__":
    train()
