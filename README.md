# OffDRL

A modular offline reinforcement learning toolkit built with PyTorch.

This repository packages the reusable algorithm layer separately from any specific environment or dataset. The intended workflow is:

- `OffDRL`: reusable algorithm package
- `Grid2AI`: your environment, dataset generation, and experiment scripts

That separation lets you keep the RL algorithms as a clean dependency while evolving `Grid2AI` independently.

## Included algorithms

- BC
- SAC
- TD3
- CQL
- IQL
- MCQ
- TD3+BC
- EDAC
- BCQ
- PRDC
- Decision Transformer backbone module

## Project layout

```text
offdrl/
├── src/OffDRL/
│   ├── backbone/
│   ├── buffer/
│   ├── generation/
│   ├── modules/
│   ├── policy/
│   ├── trainer/
│   └── utils/
├── examples/
├── tests/
├── pyproject.toml
└── README.md
```

## Installation

### Local editable install

```bash
pip install -e .
```

### Install with example dependencies

```bash
pip install -e .[examples]
```

### Install from GitHub in another project

```bash
pip install git+https://github.com/<your-name>/offdrl.git
```

Or pin it in `Grid2AI/pyproject.toml`:

```toml
[project]
dependencies = [
  "offdrl @ git+https://github.com/<your-name>/offdrl.git"
]
```

## Using OffDRL inside Grid2AI

Your `Grid2AI` repository should own:

- environment registration
- offline dataset collection
- training entry scripts
- experiment configs

Example sketch:

```python
import gymnasium as gym
import numpy as np
import torch

from OffDRL.backbone import MLP
from OffDRL.buffer import ReplayBuffer
from OffDRL.modules import ActorProb, Critic, TanhDiagGaussian
from OffDRL.policy.model_free.cql import CQLPolicy
from OffDRL.trainer import MFPolicyTrainer

env = gym.make("Grid2AI-v0")
obs_shape = env.observation_space.shape
action_dim = int(np.prod(env.action_space.shape))
max_action = float(env.action_space.high[0])

actor_backbone = MLP(input_dim=int(np.prod(obs_shape)), hidden_dims=[256, 256, 256])
dist = TanhDiagGaussian(
    latent_dim=actor_backbone.output_dim,
    output_dim=action_dim,
    unbounded=True,
    conditioned_sigma=True,
    max_mu=max_action,
)
actor = ActorProb(actor_backbone, dist, device="cuda")
critic1 = Critic(MLP(int(np.prod(obs_shape)) + action_dim, [256, 256, 256]), device="cuda")
critic2 = Critic(MLP(int(np.prod(obs_shape)) + action_dim, [256, 256, 256]), device="cuda")

policy = CQLPolicy(
    actor=actor,
    critic1=critic1,
    critic2=critic2,
    actor_optim=torch.optim.Adam(actor.parameters(), lr=1e-4),
    critic1_optim=torch.optim.Adam(critic1.parameters(), lr=3e-4),
    critic2_optim=torch.optim.Adam(critic2.parameters(), lr=3e-4),
    action_space=env.action_space,
)
```

## Notes

- `generation/diffusion.py` is kept as experimental work and may require extra local dependencies before use.
- `examples/` contains original script-style experiment entry points and is best treated as reference code.
- The package name on PyPI/GitHub is `offdrl`, while the import path remains `OffDRL` to preserve your existing code.

## Recommended next step for Grid2AI

Keep your new project structured like this:

```text
grid2ai/
├── env/
├── datasets/
├── scripts/
│   ├── train_cql.py
│   ├── train_iql.py
│   └── train_td3bc.py
└── pyproject.toml
```

Then install this repo as a dependency instead of copying the source tree into `Grid2AI`.

## License

MIT
