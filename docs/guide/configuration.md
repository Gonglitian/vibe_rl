# Configuration

vibe-rl uses [tyro](https://brentyi.github.io/tyro/) for type-safe CLI configuration. All settings are frozen dataclasses — immutable, serializable, and safe to pass as `jit` static arguments.

## Overview

The configuration system has three layers:

1. **Algorithm configs** — `PPOConfig`, `DQNConfig`, `SACConfig` with algorithm-specific hyperparameters
2. **Runner config** — `RunnerConfig` controlling the training loop, evaluation, and checkpointing
3. **Train config** — `TrainConfig` bundling environment, algorithm, and runner into a single object

## TrainConfig

The top-level config that the training script works with:

```python
@dataclass(frozen=True)
class TrainConfig:
    env_id: str = "CartPole-v1"
    algo: PPOConfig | DQNConfig | SACConfig = PPOConfig()
    runner: RunnerConfig = RunnerConfig()
```

The `algo` field is a tagged union — tyro automatically generates subcommands for each algorithm type.

## Preset system

Presets are pre-tuned `TrainConfig` instances registered in a dictionary. The CLI uses `tyro.extras.overridable_config_cli` to let you pick a preset and override any field:

```bash
# Use a preset as-is
python scripts/train.py cartpole_ppo

# Override specific fields
python scripts/train.py cartpole_ppo --algo.lr 1e-3
python scripts/train.py pendulum_sac --runner.total_timesteps 500000

# See all available options
python scripts/train.py cartpole_ppo --help
```

### Built-in presets

```python
PRESETS = {
    "cartpole_ppo": TrainConfig(
        env_id="CartPole-v1",
        algo=PPOConfig(
            hidden_sizes=(64, 64), lr=2.5e-4,
            n_steps=128, n_minibatches=4, n_epochs=4, num_envs=4,
        ),
        runner=RunnerConfig(total_timesteps=100_000, eval_every=5_000),
    ),
    "cartpole_dqn": TrainConfig(
        env_id="CartPole-v1",
        algo=DQNConfig(
            hidden_sizes=(128, 128), lr=1e-3,
            batch_size=64, target_update_freq=1_000, epsilon_decay_steps=50_000,
        ),
        runner=RunnerConfig(
            total_timesteps=100_000, eval_every=5_000,
            buffer_size=100_000, warmup_steps=1_000,
        ),
    ),
    "pendulum_ppo": TrainConfig(
        env_id="Pendulum-v1",
        algo=PPOConfig(
            hidden_sizes=(64, 64), lr=3e-4,
            n_steps=2048, n_minibatches=32, n_epochs=10,
            num_envs=1, ent_coef=0.0,
        ),
        runner=RunnerConfig(total_timesteps=200_000, eval_every=10_000),
    ),
    "pendulum_sac": TrainConfig(
        env_id="Pendulum-v1",
        algo=SACConfig(
            hidden_sizes=(256, 256),
            actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4,
            batch_size=256, tau=0.005,
        ),
        runner=RunnerConfig(
            total_timesteps=200_000, eval_every=5_000,
            buffer_size=100_000, warmup_steps=1_000,
        ),
    ),
    "gridworld_dqn": TrainConfig(
        env_id="GridWorld-v0",
        algo=DQNConfig(
            hidden_sizes=(64, 64), lr=5e-4,
            batch_size=32, target_update_freq=500, epsilon_decay_steps=20_000,
        ),
        runner=RunnerConfig(
            total_timesteps=50_000, eval_every=5_000,
            buffer_size=50_000, warmup_steps=500,
        ),
    ),
}
```

### Creating custom presets

Add entries to the `PRESETS` dictionary in `src/vibe_rl/configs/presets.py`:

```python
PRESETS["my_experiment"] = (
    "Description of my experiment",
    TrainConfig(
        env_id="CartPole-v1",
        algo=PPOConfig(hidden_sizes=(128, 128), lr=1e-4),
        runner=RunnerConfig(total_timesteps=500_000),
    ),
)
```

Then use it:

```bash
python scripts/train.py my_experiment
```

## RunnerConfig reference

`RunnerConfig` controls everything outside the algorithm — the training budget, evaluation schedule, checkpointing, and device setup.

```python
@dataclass(frozen=True)
class RunnerConfig:
    # Training budget
    total_timesteps: int = 100_000

    # Evaluation
    eval_every: int = 5_000        # evaluate every N steps
    eval_episodes: int = 10        # episodes per evaluation

    # Logging
    log_interval: int = 1_000      # log metrics every N steps

    # Off-policy specific
    buffer_size: int = 100_000     # replay buffer capacity
    warmup_steps: int = 1_000      # random actions before training

    # Multi-device
    num_devices: int | None = None # auto-detect if None
    num_envs: int = 1              # parallel envs per device
    fsdp_devices: int = 1          # FSDP axis size (1 = pure data-parallel)

    # Seeding
    seed: int = 0

    # Checkpointing
    checkpoint_dir: str | None = None   # None = no checkpointing
    checkpoint_interval: int = 5_000    # save every N steps
    max_checkpoints: int = 5            # recent checkpoints to retain
    keep_period: int | None = None      # permanently keep every N steps
    resume: bool = False                # resume from existing checkpoint
    overwrite: bool = False             # wipe existing checkpoints

    # Auto-plotting
    plot: bool = True              # generate reward curve after training
```

### Field details

| Field | Default | Description |
|-------|---------|-------------|
| `total_timesteps` | `100_000` | Total environment steps before stopping |
| `eval_every` | `5_000` | Run evaluation every N timesteps |
| `eval_episodes` | `10` | Number of episodes per evaluation round |
| `log_interval` | `1_000` | Write metrics to JSONL every N steps |
| `buffer_size` | `100_000` | Replay buffer capacity (DQN/SAC only) |
| `warmup_steps` | `1_000` | Random exploration steps before training (DQN/SAC only) |
| `num_devices` | `None` | Number of devices; `None` = auto-detect all |
| `num_envs` | `1` | Vectorized environments per device |
| `fsdp_devices` | `1` | FSDP mesh axis size; `1` = pure data-parallel |
| `seed` | `0` | Global PRNG seed |
| `checkpoint_dir` | `None` | Override checkpoint location; `None` = use RunDir default |
| `checkpoint_interval` | `5_000` | Save checkpoint every N steps |
| `max_checkpoints` | `5` | Maximum recent checkpoints to keep |
| `keep_period` | `None` | Permanently retain every N-th step checkpoint |
| `resume` | `False` | Resume training from latest checkpoint |
| `overwrite` | `False` | Delete existing checkpoints before starting |
| `plot` | `True` | Auto-generate reward curve after training |

## Command-line override examples

```bash
# Longer training with more frequent evaluation
python scripts/train.py cartpole_ppo \
  --runner.total_timesteps 500000 \
  --runner.eval_every 2000

# Disable auto-plotting
python scripts/train.py cartpole_ppo --runner.plot false

# Resume from checkpoint
python scripts/train.py cartpole_ppo --runner.resume true

# Multi-GPU with 4 envs per device
python scripts/train.py cartpole_ppo \
  --runner.num_envs 4 \
  --algo.num_envs 4

# Change algorithm hyperparameters
python scripts/train.py cartpole_ppo \
  --algo.lr 1e-3 \
  --algo.n_epochs 8 \
  --algo.hidden_sizes '(128, 128)'
```
