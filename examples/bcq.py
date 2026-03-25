import argparse
import random
import numpy as np
import torch
import gymnasium as gym

from utils import setup_project_paths
setup_project_paths()
import chargym 

from OffDRL.backbone import MLP
from OffDRL.modules import Critic
from OffDRL.buffer import ReplayBuffer
from OffDRL.utils.logger import Logger
from OffDRL.trainer import MFPolicyTrainer

from OffDRL.policy.model_free.bcq import BCQPolicy, PerturbMLP
from OffDRL.generation.vae import VAE

from utils import (
    load_offline_dataset, plot_curves,
    setup_logging_dirs, get_output_config, save_final_model
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="BCQ")

    # env & data
    parser.add_argument("--env_id", type=str, default="ChargingEnv-v0")
    parser.add_argument("--dataset", type=str, default="data/mixed_dataset.h5")

    # seeds & device
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # model (critics & perturb)
    parser.add_argument("--hidden-dims", type=int, nargs='*', default=[256, 256, 256]) 
    parser.add_argument("--perturb-hidden", type=int, default=256)                 

    # optim
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--perturb-lr", type=float, default=1e-3)
    parser.add_argument("--behavior-policy-lr", type=float, default=1e-3)

    # BCQ
    parser.add_argument("--num-sampled-actions", type=int, default=10)  
    parser.add_argument("--perturbation-limit", type=float, default=0.1)
    parser.add_argument("--l2-delta", type=float, default=0.0)        
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--update-actor-freq", type=int, default=2)

    # VAE (behavior policy)
    parser.add_argument("--vae-hidden-dim", type=int, default=750)
    parser.add_argument("--vae-latent-mul", type=int, default=2)  

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


def build_critics(args, obs_dim, act_dim):
    c1_backbone = MLP(input_dim=obs_dim + act_dim, hidden_dims=args.hidden_dims)
    c2_backbone = MLP(input_dim=obs_dim + act_dim, hidden_dims=args.hidden_dims)
    critic1 = Critic(c1_backbone, args.device)
    critic2 = Critic(c2_backbone, args.device)
    return critic1, critic2


def train(args=get_args()):
    # Env
    env = gym.make(args.env_id)
    env.reset(seed=args.seed)

    # Dataset
    dataset = load_offline_dataset(args.dataset)

    # Shapes
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    max_action = float(env.action_space.high[0])

    # Seeds
    seed_everything(args.seed)

    # Critics
    critic1, critic2 = build_critics(args, obs_dim, act_dim)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    # Behavior policy (state-conditional VAE)
    behavior_policy = VAE(
        input_dim=obs_dim,
        output_dim=act_dim,
        hidden_dim=args.vae_hidden_dim,
        latent_dim=act_dim * args.vae_latent_mul,
        max_action=max_action,
        device=args.device
    )
    behavior_policy_optim = torch.optim.Adam(behavior_policy.parameters(), lr=args.behavior_policy_lr)

    # Perturbation network (actor in BCQ)
    perturb = PerturbMLP(obs_dim=obs_dim, act_dim=act_dim, hidden=args.perturb_hidden).to(args.device)
    perturb_optim = torch.optim.Adam(perturb.parameters(), lr=args.perturb_lr)

    # Policy
    policy = BCQPolicy(
        critic1=critic1,
        critic2=critic2,
        critic1_optim=critic1_optim,
        critic2_optim=critic2_optim,
        behavior_policy=behavior_policy,
        behavior_policy_optim=behavior_policy_optim,
        obs_dim=obs_dim,
        act_dim=act_dim,
        actor=perturb,                
        actor_optim=perturb_optim,
        num_sampled_actions=args.num_sampled_actions,
        perturbation_limit=args.perturbation_limit,
        l2_delta=args.l2_delta,
        tau=args.tau,
        gamma=args.gamma,
        max_action=max_action,
        update_actor_freq=args.update_actor_freq,
    )

    # Buffer
    buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=env.observation_space.shape,
        obs_dtype=np.float32,
        action_shape=(act_dim,),
        action_dtype=np.float32,
        device=args.device
    )
    buffer.load_dataset(dataset)

    # Logging
    model_dir, logs_dir = setup_logging_dirs("BCQ")
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
        eval_episodes=args.eval_episodes
    )
    
    policy_trainer.train()

    # Save & plot
    save_final_model(policy, model_dir)
    plot_curves(logs_dir, "BCQ")
    logger.close()


if __name__ == "__main__":
    train()
