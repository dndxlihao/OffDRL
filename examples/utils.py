import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import h5py


def find_project_root():
    current = Path(__file__).parent
    while current != current.parent:
        if (current / "OffDRL").exists() and (current / "env").exists():
            return current
        current = current.parent
    return Path(__file__).parent.parent


def setup_project_paths():
    project_root = find_project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    env_dir = project_root / "env"
    if str(env_dir) not in sys.path:
        sys.path.insert(0, str(env_dir))
    return project_root


def load_offline_dataset(dataset_path: str):
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    if dataset_path.suffix.lower() in ['.h5', '.hdf5']:
        with h5py.File(dataset_path, 'r') as f:
            dataset = {
                'observations': f['observations'][:],
                'next_observations': f['next_observations'][:],
                'actions': f['actions'][:],
                'rewards': f['rewards'][:],
                'terminateds': f['terminateds'][:],
                'truncateds': f['truncateds'][:],
            }
    elif dataset_path.suffix.lower() == '.npz':
        data = np.load(dataset_path)
        dataset = {
            'observations': data['observations'],
            'next_observations': data['next_observations'],
            'actions': data['actions'],
            'rewards': data['rewards'],
            'terminateds': data['terminateds'],
            'truncateds': data['truncateds'],
        }
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")
    
    return dataset


def read_training_csv(log_dir: Path):
    csv_path = log_dir / "record" / "policy_training_progress.csv"
    if not csv_path.exists():
        print(f"Warning: Training CSV not found at {csv_path}")
        return None
    return pd.read_csv(csv_path)


def plot_curves(log_dir: Path, algo_name: str):
    df = read_training_csv(log_dir)
    if df is None:
        return
    
    plt.style.use('seaborn-v0_8')
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'legend.fontsize': 11,
        'lines.linewidth': 2.0,
    })
    
    if 'eval/normalized_episode_reward' in df.columns:
        plt.figure(figsize=(8, 5))
        
        rewards = df['eval/normalized_episode_reward'].dropna()
        timesteps = np.arange(len(rewards)) * 1000
        
        plt.plot(timesteps, rewards, color="#1f77b4", label=f"{algo_name}")
        
        if 'eval/normalized_episode_reward_std' in df.columns:
            stds = df['eval/normalized_episode_reward_std'].dropna()
            if len(stds) == len(rewards):
                lo = rewards - stds
                hi = rewards + stds
                plt.fill_between(timesteps, lo, hi, color="#1f77b4", alpha=0.2, linewidth=0)
        
        plt.xlabel("Timestep")
        plt.ylabel("Normalized Reward")
        plt.title(f"{algo_name} Evaluation Reward")
        plt.grid(alpha=0.3)
        plt.legend()
        
        reward_plot_path = log_dir / f"{algo_name}_reward.png"
        plt.savefig(reward_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"saved: {reward_plot_path}")
    
    loss_columns = [col for col in df.columns if col.startswith('loss/')]
    if loss_columns:
        plt.figure(figsize=(8, 5))
        
        colors = ["#d62728", "#2ca02c", "#9467bd", "#8c564b"]
        
        for i, col in enumerate(loss_columns):
            losses = df[col].dropna()
            timesteps = np.arange(len(losses)) * 1000
            color = colors[i % len(colors)]
            
            if 'actor' in col.lower():
                label = "Actor Loss"
            elif 'critic1' in col.lower():
                label = "Critic1 Loss"
            elif 'critic2' in col.lower():
                label = "Critic2 Loss"
            elif 'alpha' in col.lower():
                label = "Alpha Loss"
            else:
                label = col.replace('loss/', '')
            
            plt.plot(timesteps, losses, color=color, label=label)
        
        plt.xlabel("Timestep")
        plt.ylabel("Loss")
        plt.title(f"{algo_name} Training Loss")
        plt.grid(alpha=0.3)
        plt.legend()
        
        loss_plot_path = log_dir / f"{algo_name}_loss.png"
        plt.savefig(loss_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"saved: {loss_plot_path}")


def setup_logging_dirs(algo_name: str):
    project_root = find_project_root()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_dir = project_root / "data" / "models" / algo_name.upper() / timestamp
    logs_dir = project_root / "data" / "logs" / algo_name.upper() / timestamp
    
    model_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    return model_dir, logs_dir


def get_output_config():
    return {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "tb": "tensorboard"
    }


def save_final_model(policy, model_dir: Path):
    import torch
    final_model_path = model_dir / "policy.pth"
    torch.save(policy.state_dict(), final_model_path)
    return final_model_path
