import argparse
import random
import numpy as np
import torch
import gymnasium as gym

from utils import setup_project_paths
setup_project_paths()
import chargym

from OffDRL.backbone import MLP
from OffDRL.modules import ActorProb, Critic, TanhDiagGaussian
from OffDRL.buffer import ReplayBuffer
from OffDRL.utils.logger import Logger
from OffDRL.trainer import MFPolicyTrainer

from OffDRL.policy.model_free.mcq import MCQPolicy
from OffDRL.generation.vae import VAE

from utils import (
    load_offline_dataset, plot_curves,
    setup_logging_dirs, get_output_config, save_final_model
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="MCQ")

    # env & data
    parser.add_argument("--env_id", type=str, default="ChargingEnv-v0")
    parser.add_argument("--dataset", type=str, default="data/mixed_dataset.h5")

    # seeds & device
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # model
    parser.add_argument("--hidden-dims", type=int, nargs='*', default=[256, 256, 256])
    parser.add_argument("--dropout_rate", type=float, default=0.1)

    # SAC/optim
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-alpha", action="store_true", default=True)
    parser.add_argument("--target-entropy", type=float, default=None)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)

    # MCQ
    parser.add_argument("--lmbda", type=float, default=0.9)
    parser.add_argument("--num-sampled-actions", type=int, default=10)

    # VAE (behavior policy)
    parser.add_argument("--vae-hidden-dim", type=int, default=750)
    parser.add_argument("--vae-latent-mul", type=int, default=2)  
    parser.add_argument("--behavior-policy-lr", type=float, default=1e-3)

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


def build_actor_critic(args, obs_dim, act_dim, max_action):
    actor_backbone = MLP(
        input_dim=obs_dim,
        hidden_dims=args.hidden_dims,
        dropout_rate=args.dropout_rate
    )
    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=act_dim,
        unbounded=True,
        conditioned_sigma=True,
        max_mu=max_action
    )
    actor = ActorProb(actor_backbone, dist, args.device)

    # Critics
    critic1_backbone = MLP(input_dim=obs_dim + act_dim, hidden_dims=args.hidden_dims)
    critic2_backbone = MLP(input_dim=obs_dim + act_dim, hidden_dims=args.hidden_dims)
    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)

    return actor, critic1, critic2


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

    # Actor/Critic
    actor, critic1, critic2 = build_actor_critic(args, obs_dim, act_dim, max_action)

    # Optims
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic-lr if hasattr(args, "critic-lr") else args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic-lr if hasattr(args, "critic-lr") else args.critic_lr)

    # Entropy temperature
    if args.auto_alpha:
        target_entropy = args.target_entropy if args.target_entropy is not None else -float(act_dim)
        args.target_entropy = target_entropy
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = args.alpha

    # Behavior policy
    behavior_policy = VAE(
        input_dim=obs_dim,
        output_dim=act_dim,
        hidden_dim=args.vae_hidden_dim,
        latent_dim=act_dim * args.vae_latent_mul,
        max_action=max_action,
        device=args.device
    )
    behavior_policy_optim = torch.optim.Adam(behavior_policy.parameters(), lr=args.behavior_policy_lr)

    # Policy
    policy = MCQPolicy(
        actor=actor,
        critic1=critic1,
        critic2=critic2,
        actor_optim=actor_optim,
        critic1_optim=critic1_optim,
        critic2_optim=critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=alpha,
        behavior_policy=behavior_policy,
        behavior_policy_optim=behavior_policy_optim,
        lmbda=args.lmbda,
        num_sampled_actions=args.num_sampled_actions,
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
    model_dir, logs_dir = setup_logging_dirs("MCQ")
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

    # Train
    policy_trainer.train()

    # Save & plot
    save_final_model(policy, model_dir)
    plot_curves(logs_dir, "MCQ")
    logger.close()


if __name__ == "__main__":
    train()
