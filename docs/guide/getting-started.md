# Getting Started

## Installation

vibe-rl requires **Python 3.12+** and [uv](https://docs.astral.sh/uv/).

### Basic install

```bash
# CPU (development)
uv sync

# GPU (CUDA 12)
uv sync && uv pip install "jax[cuda12]"

# TPU
uv sync && uv pip install --upgrade "jax[tpu]" \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

### Optional dependency groups

Install extras for specific features:

```bash
# JAX environments (gymnax, brax, jumanji)
uv sync --extra envs

# Orbax checkpointing
uv sync --extra checkpoint

# Offline data pipeline (lerobot, datasets, torch)
uv sync --extra data

# WebSocket inference serving
uv sync --extra serving

# Reward curve plotting (matplotlib, seaborn)
uv sync --extra plotting

# Development tools (pytest, ruff)
uv sync --extra dev

# Everything
uv sync --all-extras
```

Or with pip:

```bash
pip install -e ".[envs,checkpoint,plotting]"
```

## Your first training run

Train PPO on CartPole using the built-in preset system:

```bash
python scripts/train.py cartpole_ppo
```

This selects the `cartpole_ppo` preset — a pre-tuned configuration that bundles the environment, algorithm hyperparameters, and runner settings. You'll see output like:

```
Run directory: runs/CartPole-v1_ppo_20260216_143022
Training complete | final_loss=0.0234 | final_entropy=0.5821
Metrics: runs/CartPole-v1_ppo_20260216_143022/logs/metrics.jsonl
Reward curve: runs/CartPole-v1_ppo_20260216_143022/artifacts/reward_curve.png
```

### Available presets

| Preset | Algorithm | Environment | Description |
|--------|-----------|-------------|-------------|
| `cartpole_ppo` | PPO | CartPole-v1 | Fast sanity-check |
| `cartpole_dqn` | DQN | CartPole-v1 | Discrete off-policy |
| `pendulum_ppo` | PPO | Pendulum-v1 | Continuous control |
| `pendulum_sac` | SAC | Pendulum-v1 | Continuous off-policy |
| `gridworld_dqn` | DQN | GridWorld-v0 | Tabular-scale discrete |

Override any field from the command line:

```bash
python scripts/train.py cartpole_ppo --algo.lr 1e-3
python scripts/train.py pendulum_sac --runner.total_timesteps 500000
```

## Run directory layout

Every training run creates a structured output directory under `runs/`:

```
runs/
└── CartPole-v1_ppo_20260216_143022/
    ├── config.json              # Frozen config snapshot
    ├── checkpoints/
    │   ├── step_10000/
    │   ├── step_20000/
    │   └── best -> step_20000   # Symlink to best checkpoint
    ├── logs/
    │   └── metrics.jsonl        # Per-step training metrics
    ├── videos/
    │   └── eval_step_10000.mp4
    └── artifacts/
        └── reward_curve.png     # Auto-generated reward plot
```

The `RunDir` class manages this layout — see the [Configuration](./configuration.md) guide for details on `checkpoint_dir`, `checkpoint_interval`, and other runner settings.

## Viewing results

If you have the `plotting` extras installed, a reward curve is auto-generated after training. You can also plot manually:

```bash
# Plot a single run
python scripts/plot_rewards.py --run-dirs runs/CartPole-v1_ppo_20260216_143022

# Compare multiple runs
python scripts/plot_rewards.py \
  --run-dirs runs/CartPole-v1_ppo_* \
  --compare \
  --output comparison.png
```

See the [Plotting](./plotting.md) guide for smoothing options, custom axes, and styling.

## Next steps

- [Configuration](./configuration.md) — understand the tyro CLI, `TrainConfig`, and presets
- [Training](./training.md) — deep dive into training loops and the runner system
- [Algorithms](/algorithms/ppo) — algorithm-specific details (PPO, DQN, SAC)
