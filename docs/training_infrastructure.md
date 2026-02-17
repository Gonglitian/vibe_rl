# Training Infrastructure

This document covers the full training stack in **vibe\_rl**: how training
loops are structured, how checkpoints work, how to scale to multiple GPUs,
how metrics are recorded, and how reward curves are plotted.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Training Loops](#training-loops)
   - [On-Policy: PPO (PureJaxRL)](#on-policy-ppo-purejaxrl)
   - [Off-Policy: DQN / SAC (Hybrid)](#off-policy-dqn--sac-hybrid)
   - [Unified Training Script](#unified-training-script)
3. [RunnerConfig Reference](#runnerconfig-reference)
4. [Run Directory Layout](#run-directory-layout)
5. [Checkpointing](#checkpointing)
   - [One-Shot Save / Load](#one-shot-save--load)
   - [CheckpointManager (Periodic)](#checkpointmanager-periodic)
   - [Resuming Training](#resuming-training)
6. [Multi-GPU / FSDP Training](#multi-gpu--fsdp-training)
   - [Device Mesh](#device-mesh)
   - [Data Sharding](#data-sharding)
   - [FSDP Parameter Sharding](#fsdp-parameter-sharding)
   - [Device Utilities](#device-utilities)
7. [Metrics & Logging](#metrics--logging)
   - [MetricsLogger (JSONL)](#metricslogger-jsonl)
   - [WandB Backend](#wandb-backend)
   - [TensorBoard Backend](#tensorboard-backend)
   - [Console Logging](#console-logging)
8. [Plotting](#plotting)
   - [plot\_reward\_curve](#plot_reward_curve)
   - [Smoothing Methods](#smoothing-methods)
   - [Multi-Seed Aggregation](#multi-seed-aggregation)
   - [Color Palettes](#color-palettes)
   - [PlotConfig Reference](#plotconfig-reference)
   - [CLI Script](#cli-script)
9. [Evaluation](#evaluation)
10. [Schedules & Seeding](#schedules--seeding)
11. [Presets & CLI](#presets--cli)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│  scripts/train.py   (unified CLI entry point)        │
│    ↓ selects preset → TrainConfig(algo, runner, env) │
├──────────────────────────────────────────────────────┤
│  runner/             (training loops)                │
│  ├── train_ppo.py           on-policy  / lax.scan   │
│  ├── train_ppo_multigpu.py  multi-GPU  / FSDP       │
│  ├── train_dqn.py           off-policy / hybrid     │
│  ├── train_sac.py           off-policy / hybrid     │
│  ├── evaluator.py           vmap + while_loop eval  │
│  ├── config.py              RunnerConfig             │
│  └── device_utils.py        replicate / unreplicate │
├──────────────────────────────────────────────────────┤
│  checkpoint.py       Orbax + Equinox serialization   │
│  metrics.py          JSONL logger + WandB/TBoard     │
│  run_dir.py          Output directory management     │
│  sharding.py         Mesh / NamedSharding / FSDP     │
│  schedule.py         JIT-compatible value schedules  │
│  seeding.py          PRNG key management             │
├──────────────────────────────────────────────────────┤
│  plotting/           Reward curve visualization      │
│  ├── plot.py         Core plot_reward_curve()        │
│  ├── config.py       PlotConfig dataclass            │
│  └── colors.py       DeepMind color palette          │
└──────────────────────────────────────────────────────┘
```

vibe\_rl uses two distinct training-loop patterns depending on whether the
algorithm is on-policy or off-policy:

| Algorithm | Pattern | Loop Style | Replay Buffer |
|-----------|---------|------------|---------------|
| PPO       | PureJaxRL | `jax.lax.scan` (single JIT) | None |
| PPO multi-GPU | GSPMD | `jax.lax.scan` + `NamedSharding` | None |
| DQN       | Hybrid | Python `for` + jitted inner steps | NumPy ring buffer |
| SAC       | Hybrid | Python `for` + jitted inner steps | NumPy ring buffer |

---

## Training Loops

### On-Policy: PPO (PureJaxRL)

**File:** `src/vibe_rl/runner/train_ppo.py`

The PPO loop compiles the **entire** collect-rollout → compute-GAE →
mini-batch-SGD pipeline into a single `jax.lax.scan`, yielding:

- One JIT compilation up-front, then zero Python overhead.
- Hardware runs at full speed (GPU/TPU utilization near 100%).
- No I/O possible mid-training (metrics are batch-written after).

```python
from vibe_rl.algorithms.ppo import PPOConfig
from vibe_rl.env import make
from vibe_rl.env.wrappers import AutoResetWrapper
from vibe_rl.runner import RunnerConfig, train_ppo

env, env_params = make("CartPole-v1")
env = AutoResetWrapper(env)  # required for lax.scan loop

ppo_config = PPOConfig(n_steps=128, hidden_sizes=(64, 64))
runner_config = RunnerConfig(total_timesteps=100_000)

train_state, metrics = train_ppo(
    env, env_params,
    ppo_config=ppo_config,
    runner_config=runner_config,
)
```

**Vectorized environments**: When `ppo_config.num_envs > 1`, the runner
uses `jax.vmap` to run N environments in parallel. Total timesteps per
update becomes `n_steps * num_envs`.

```python
ppo_config = PPOConfig(n_steps=128, num_envs=8, hidden_sizes=(64, 64))
train_state, metrics = train_ppo(
    env, env_params, ppo_config=ppo_config, runner_config=runner_config,
)
```

**Return types:**

- `PPOTrainState` — final agent state, env state, obs, and RNG.
- `PPOMetricsHistory` — per-update arrays of shape `(n_updates,)`:
  `total_loss`, `actor_loss`, `critic_loss`, `entropy`, `approx_kl`.

**Scan body** (one iteration = one update):

1. `PPO.collect_rollout()` — step the env `n_steps` times, collecting
   `(obs, action, reward, done, log_prob, value)` transitions.
2. `PPO.update()` — compute GAE advantages, then run `n_epochs` of
   mini-batch SGD over `n_minibatches` random slices.

**Metrics I/O**: Since `lax.scan` forbids side effects, PPO batch-writes
all metrics to JSONL **after** the scan completes, when `run_dir` is provided.

### Off-Policy: DQN / SAC (Hybrid)

**Files:** `src/vibe_rl/runner/train_dqn.py`, `src/vibe_rl/runner/train_sac.py`

Off-policy algorithms need a mutable replay buffer that cannot live inside
`lax.scan`. The design is:

- **Outer loop**: Python `for step in range(...)` — pushes transitions into
  the buffer, handles eval/logging callbacks, manages checkpoints.
- **Inner step**: `@jax.jit`-compiled `_act_and_step()` — selects action +
  steps the environment with no Python overhead on the hot path.

```python
from vibe_rl.algorithms.dqn import DQNConfig
from vibe_rl.runner import RunnerConfig, train_dqn

result = train_dqn(
    env, env_params,
    dqn_config=DQNConfig(),
    runner_config=RunnerConfig(total_timesteps=50_000),
)
# result.agent_state, result.episode_returns, result.metrics_log
```

SAC is structurally identical but uses continuous actions and a separate
`_ContinuousReplayBuffer` with float32 action storage:

```python
from vibe_rl.algorithms.sac import SACConfig
from vibe_rl.runner import train_sac

result = train_sac(
    env, env_params,
    sac_config=SACConfig(),
    runner_config=RunnerConfig(total_timesteps=100_000),
    action_dim=1,
)
```

**DQN training loop structure:**

```
for step in 1..total_timesteps:
    1. _act_and_step()          (jitted: epsilon-greedy + env.step)
    2. buffer.push(transition)  (numpy)
    3. Track episode returns    (on done)
    4. if buffer >= warmup:
       a. batch = buffer.sample(batch_size)
       b. DQN.update(batch)    (jitted: TD loss + target network)
    5. Periodic checkpoint      (ckpt_mgr.save)
    6. Periodic metrics log     (MetricsLogger.write)
```

**SAC metrics logged**: `actor_loss`, `critic_loss`, `alpha`, `entropy`, `q_mean`.
**DQN metrics logged**: `loss`, `q_mean`, `epsilon`.
Both log `episode_return` and `episode_length` on each episode completion.

### Unified Training Script

**File:** `scripts/train.py`

A single CLI entry point that selects a preset configuration and dispatches
to the appropriate training function:

```bash
python scripts/train.py cartpole_ppo
python scripts/train.py cartpole_ppo --algo.lr 1e-3
python scripts/train.py pendulum_sac --runner.total_timesteps 500000
```

The script:
1. Creates a `RunDir` for the experiment.
2. Saves the config snapshot to `config.json`.
3. Dispatches to `train_ppo`, `train_dqn`, or `train_sac`.
4. Auto-plots a reward curve to `artifacts/reward_curve.png` (when
   `runner.plot=True` and `matplotlib` is installed).

---

## RunnerConfig Reference

**File:** `src/vibe_rl/runner/config.py`

`RunnerConfig` is a frozen dataclass controlling the outer training loop.
Algorithm-specific settings live in `PPOConfig`, `DQNConfig`, or `SACConfig`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `total_timesteps` | `int` | `100_000` | Training budget in env steps |
| `eval_every` | `int` | `5_000` | Evaluate every N steps |
| `eval_episodes` | `int` | `10` | Number of eval episodes |
| `log_interval` | `int` | `1_000` | Log metrics every N steps |
| `buffer_size` | `int` | `100_000` | Replay buffer capacity (off-policy) |
| `warmup_steps` | `int` | `1_000` | Min buffer size before training |
| `num_devices` | `int\|None` | `None` | Device count (auto-detect if None) |
| `num_envs` | `int` | `1` | Parallel envs per device (multi-GPU) |
| `fsdp_devices` | `int` | `1` | FSDP axis size (1 = pure data-parallel) |
| `seed` | `int` | `0` | Random seed |
| `checkpoint_dir` | `str\|None` | `None` | Checkpoint directory (None = off) |
| `checkpoint_interval` | `int` | `5_000` | Save every N steps |
| `max_checkpoints` | `int` | `5` | Recent checkpoints to retain |
| `keep_period` | `int\|None` | `None` | Permanently keep every N steps |
| `resume` | `bool` | `False` | Resume from existing checkpoint |
| `overwrite` | `bool` | `False` | Wipe existing checkpoints |
| `plot` | `bool` | `True` | Generate reward curve after training |

---

## Run Directory Layout

**File:** `src/vibe_rl/run_dir.py`

`RunDir` creates and manages a standardized output directory for each
experiment run:

```
runs/
└── dqn_cartpole_20260215_143022/
    ├── config.json          # frozen experiment config snapshot
    ├── checkpoints/
    │   ├── step_10000/      # Orbax checkpoint at step 10k
    │   ├── step_20000/
    │   └── best -> step_20000   # symlink to best checkpoint
    ├── logs/
    │   ├── metrics.jsonl    # structured JSONL metrics
    │   └── events.out.*     # TensorBoard events (optional)
    ├── videos/
    │   ├── eval_step_10000.mp4
    │   └── eval_step_20000.mp4
    └── artifacts/
        └── reward_curve.png
```

**Key API:**

```python
run = RunDir("dqn_cartpole", base_dir="runs")

# Path accessors
run.root           # runs/dqn_cartpole_20260215_143022
run.checkpoints    # .../checkpoints/
run.logs           # .../logs/
run.videos         # .../videos/
run.artifacts      # .../artifacts/
run.log_path()     # .../logs/metrics.jsonl

# Config
run.save_config(config)   # serialize frozen dataclass to JSON
run.load_config()         # read it back

# Checkpoints
run.checkpoint_dir(step=10000)    # .../checkpoints/step_10000/
run.list_checkpoints()            # [(10000, Path), (20000, Path)]
run.latest_checkpoint             # Path to highest step
run.mark_best(step=20000)         # create symlink
run.best_checkpoint               # resolve symlink
run.cleanup_checkpoints(keep=5)   # prune old checkpoints

# Discovery
from vibe_rl.run_dir import find_runs
runs = find_runs("runs", experiment_name="dqn_cartpole")
```

---

## Checkpointing

**File:** `src/vibe_rl/checkpoint.py`

Requires the optional dependency: `pip install vibe-rl[checkpoint]`
(installs `orbax-checkpoint`).

### One-Shot Save / Load

For saving/loading a single checkpoint outside a training loop:

```python
from vibe_rl.checkpoint import save_checkpoint, load_checkpoint

# Save (creates directory structure automatically)
save_checkpoint("checkpoints/", train_state, step=1000)

# With metadata
save_checkpoint(
    "checkpoints/", train_state,
    step=1000,
    metadata={"env": "CartPole-v1", "algorithm": "dqn"},
)

# Load (needs a template pytree for structure)
restored = load_checkpoint("checkpoints/", like=train_state, step=1000)

# Load and replicate for multi-device
restored = load_checkpoint(
    "checkpoints/", like=train_state, step=1000, replicate_to=4,
)
```

**Low-level Equinox serialization** (no Orbax dependency):

```python
from vibe_rl.checkpoint import save_eqx, load_eqx

save_eqx("model.eqx", pytree)
restored = load_eqx("model.eqx", like=pytree)
```

### CheckpointManager (Periodic)

For managed checkpointing during training loops — handles retention
policies, async writes, and best-model tracking:

```python
from vibe_rl.checkpoint import CheckpointManager

with CheckpointManager(
    "checkpoints/",
    max_to_keep=5,
    keep_period=500,                # permanently keep every 500 steps
    save_interval_steps=100,        # only save on multiples of 100
    async_timeout_secs=7200,        # 2-hour timeout for async writes
    best_fn=lambda m: m["loss"],    # track best model by loss
    best_mode="min",
) as mgr:
    for step in range(num_steps):
        state = train_step(state)
        mgr.save(step, state, metrics={"loss": float(loss)})
    mgr.wait()  # block until all async saves complete
```

**Retention logic:**

- `max_to_keep=5` — retain only the 5 most recent checkpoints.
- `keep_period=500` — checkpoints at steps 500, 1000, 1500, ... are
  **never** pruned (permanent snapshots for experiment archival).
- `best_fn` + `best_mode` — track the best checkpoint by a custom metric.

**Manager API:**

| Method | Description |
|--------|-------------|
| `save(step, pytree, metrics=...)` | Save if step is on the interval |
| `restore(step, like)` | Restore a checkpoint into a template pytree |
| `latest_step()` | Latest available step number |
| `all_steps()` | All checkpoint step numbers (sorted) |
| `wait()` | Block until async writes complete |
| `close()` | Wait + close the Orbax manager |

### Resuming Training

The `initialize_checkpoint_dir()` function implements openpi-style
resume/overwrite semantics:

```python
from vibe_rl.checkpoint import initialize_checkpoint_dir

mgr, resuming = initialize_checkpoint_dir(
    "checkpoints/",
    keep_period=500,
    overwrite=False,
    resume=True,
    max_to_keep=5,
    save_interval_steps=100,
)

if resuming:
    step = mgr.latest_step()
    agent_state = mgr.restore(step, agent_state)
    start_step = step + 1
```

**Behavior matrix:**

| Existing checkpoints? | `resume=True` | `overwrite=True` | Neither | Result |
|----------------------|---------------|------------------|---------|--------|
| No | - | - | - | Fresh start |
| Yes | Yes | - | - | Resume from latest |
| Yes | - | Yes | - | Wipe directory, fresh start |
| Yes | - | - | Yes | `FileExistsError` |

The DQN and SAC runners integrate this automatically via `RunnerConfig`:

```python
runner_config = RunnerConfig(
    checkpoint_dir="checkpoints/dqn_run",
    checkpoint_interval=5_000,
    max_checkpoints=5,
    keep_period=10_000,
    resume=True,         # resume from existing checkpoint
)
result = train_dqn(env, env_params, dqn_config=config, runner_config=runner_config)
```

---

## Multi-GPU / FSDP Training

**Files:** `src/vibe_rl/runner/train_ppo_multigpu.py`, `src/vibe_rl/sharding.py`

Data-parallel PPO across multiple devices using JAX's GSPMD partitioning
with explicit `in_shardings`/`out_shardings` on `jax.jit`. No manual
`pmean` is needed — gradient reduction is handled implicitly.

```python
from vibe_rl.runner import RunnerConfig, train_ppo_multigpu

train_state, metrics = train_ppo_multigpu(
    env, env_params,
    ppo_config=PPOConfig(n_steps=128, hidden_sizes=(64, 64)),
    runner_config=RunnerConfig(
        total_timesteps=100_000,
        num_envs=4,          # envs per device
        num_devices=None,    # auto-detect (use all GPUs)
        fsdp_devices=1,      # 1 = pure data-parallel
    ),
)
```

### Device Mesh

**File:** `src/vibe_rl/sharding.py`

The mesh is 2-D with axes `("batch", "fsdp")`:

```python
from vibe_rl.sharding import make_mesh

mesh = make_mesh(num_fsdp_devices=1)   # pure data-parallel
mesh = make_mesh(num_fsdp_devices=2)   # 2-way FSDP + data-parallel
```

When `fsdp_devices=1`, the FSDP axis is trivial and every parameter is
replicated — equivalent to pure data-parallelism.

### Data Sharding

The data shape convention is `(n_devices, num_envs, *feature_dims)`:

- Each device shard runs `num_envs` parallel environments.
- The sharded outer axis distributes across devices.
- Inside each shard, `vmap` vectorizes across environments.

```python
from vibe_rl.sharding import data_sharding, replicate_sharding

data_spec = data_sharding(mesh)       # leading axis over (batch, fsdp)
param_spec = replicate_sharding(mesh) # fully replicated
```

**Total timesteps per update** = `n_devices * num_envs * n_steps`.

### FSDP Parameter Sharding

Large parameters can be sharded across the FSDP axis to reduce per-device
memory:

```python
from vibe_rl.sharding import fsdp_sharding

param_shardings = fsdp_sharding(params_abstract, mesh, min_size_mbytes=4)
```

**Sharding rules:**

| Condition | Sharding |
|-----------|----------|
| Size < 4 MB | Replicated |
| Scalar or 1-D | Replicated |
| 2-D+, >= 4 MB, divisible dim exists | Sharded along largest divisible dim |
| 2-D+, >= 4 MB, no divisible dim | Replicated (fallback) |

### Device Utilities

**File:** `src/vibe_rl/runner/device_utils.py`

```python
from vibe_rl.runner import (
    get_num_devices,       # auto-detect or validate requested count
    replicate,             # add leading (n_devices,) axis to pytree
    unreplicate,           # take first replica: pytree[0]
    split_key_across_devices,  # split PRNG key into per-device keys
    shard_pytree,          # place pytree on mesh with data sharding
    replicate_on_mesh,     # place pytree on mesh, fully replicated
)
```

---

## Metrics & Logging

**File:** `src/vibe_rl/metrics.py`

### MetricsLogger (JSONL)

The core logger writes one JSON object per line to a JSONL file. Each
line is self-describing — fields can vary between entries.

```python
from vibe_rl.run_dir import RunDir
from vibe_rl.metrics import MetricsLogger

run = RunDir("dqn_cartpole")
logger = MetricsLogger(run.log_path())  # logs/metrics.jsonl

logger.write({"step": 1000, "loss": 0.42, "reward": 195.0})
logger.write({"step": 2000, "loss": 0.31, "episode_return": 200.0})
logger.close()
```

**Automatic fields:**
- `wall_time` — seconds since logger creation (added if not present).

**JAX/numpy conversion:** Scalars are auto-converted to Python floats
for JSON serialization.

**Reading back:**

```python
from vibe_rl.metrics import read_metrics

records = read_metrics("runs/.../logs/metrics.jsonl")
# [{"step": 1000, "loss": 0.42, ...}, ...]
```

**Context manager:**

```python
with MetricsLogger(run.log_path()) as logger:
    logger.write({"step": 1, "loss": 0.5})
# auto-closed
```

### WandB Backend

Fan out every `write()` call to Weights & Biases:

```python
from vibe_rl.metrics import MetricsLogger, WandbBackend

backend = WandbBackend(
    project="rl",
    entity="my-team",
    config={"algo": "dqn", "lr": 1e-3},
    run_dir=run.root,   # persists run ID for resumption
)
logger = MetricsLogger(run.log_path(), backends=[backend])
```

**Resume a WandB run** after interruption:

```python
from vibe_rl.metrics import resume_wandb

backend = resume_wandb(run.root, project="rl")
logger = MetricsLogger(run.log_path(), backends=[backend])
```

The run ID is stored in `<run_dir>/wandb_id.txt` and automatically
read on next launch.

### TensorBoard Backend

```python
from vibe_rl.metrics import TensorBoardBackend

backend = TensorBoardBackend(log_dir=run.logs)
logger = MetricsLogger(run.log_path(), backends=[backend])
```

Requires `pip install tensorboardX` (or `torch.utils.tensorboard`).
All numeric values in each `write()` call are recorded as TensorBoard
scalars.

### Console Logging

Compact structured console output:

```python
from vibe_rl.metrics import setup_logging, log_step_progress

setup_logging()  # install compact formatter on "vibe_rl" logger

log_step_progress(5000, 100_000, metrics={"loss": 0.42, "reward": 195.0})
# Output: I 2026-02-15 14:30:22.123 [vibe_rl] step 5000/100000 (5.0%) | loss=0.42 reward=195
```

**Formatter style:** `{level_abbrev} {timestamp}.{ms} [{logger_name}] {message}`

Level abbreviations: `D`=DEBUG, `I`=INFO, `W`=WARNING, `E`=ERROR, `C`=CRITICAL.

---

## Plotting

**Files:** `src/vibe_rl/plotting/`

Requires: `pip install 'vibe-rl[plotting]'` (installs matplotlib + seaborn).

### plot\_reward\_curve

The main function reads JSONL metrics files and renders publication-quality
reward curves:

```python
from vibe_rl.plotting import plot_reward_curve

# Single run
fig = plot_reward_curve("runs/ppo_seed0")

# Multiple seeds (auto-aggregated with shading)
fig = plot_reward_curve(
    ["runs/ppo_seed0", "runs/ppo_seed1", "runs/ppo_seed2"],
    y_key="episode_return",
    smooth=10,
)

# Compare algorithms
fig = plot_reward_curve(
    ["runs/ppo_run", "runs/dqn_run", "runs/sac_run"],
    group_by="auto",  # groups by detected algorithm name
)

fig.savefig("reward.png")
```

**Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `run_dirs` | path(s) | required | Run directories or JSONL files |
| `x_key` | `str` | `"step"` | JSON key for x-axis |
| `y_key` | `str` | `"episode_return"` | JSON key for y-axis |
| `smooth` | `int\|None` | `None` | Override smooth\_radius |
| `style` | `str\|None` | `None` | Override matplotlib style |
| `group_by` | `str` | `"auto"` | `"auto"` (by algo name) or `"none"` |
| `config` | `PlotConfig\|None` | `None` | Full config (overrides applied on top) |
| `save_path` | path\|None | `None` | Save figure to this path |

**Algorithm detection:** When `group_by="auto"`, the function reads
`config.json` from the run directory (looking for `algorithm`, `algo`,
or `algo_name` keys), or falls back to parsing the directory name
(e.g. `ppo_cartpole_20260215_143022` -> `ppo_cartpole`).

### Smoothing Methods

Two smoothing methods are available:

**Window averaging** (default, SpinningUp-style):

```python
from vibe_rl.plotting import smooth_window
smoothed = smooth_window(y, radius=10)
# y'[i] = mean(y[max(0, i-10) : i+11])
```

**Exponential moving average** (rl-plotter style):

```python
from vibe_rl.plotting import smooth_ema
smoothed = smooth_ema(y, span=10)
# alpha = 2 / (span + 1), matches pandas EWM semantics
```

Select via `PlotConfig.smooth_mode = "window"` (default) or `"ema"`.

### Multi-Seed Aggregation

When multiple run directories map to the same algorithm name:

1. Each seed's `(x, y)` series is loaded and smoothed independently.
2. All series are interpolated onto a common x-grid.
3. The mean curve is plotted with a shaded region:
   - `"std"` — mean +/- 1 standard deviation (default)
   - `"stderr"` — mean +/- standard error
   - `"none"` — no shading

### Color Palettes

Default palette is 10-color, colorblind-friendly, adapted from DeepMind
publications:

```python
from vibe_rl.plotting import get_colors, set_palette, reset_palette, DEEPMIND

print(get_colors())
# ['#0077BB', '#EE7733', '#009988', '#CC3311', ...]

set_palette(["#FF0000", "#00FF00", "#0000FF"])   # custom
reset_palette()                                    # back to DeepMind
```

Colors cycle automatically when there are more groups than palette entries.

### PlotConfig Reference

**File:** `src/vibe_rl/plotting/config.py`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `smooth_radius` | `int` | `10` | Half-window size or EMA span |
| `smooth_mode` | `str` | `"window"` | `"window"` or `"ema"` |
| `shaded` | `str` | `"std"` | `"std"`, `"stderr"`, or `"none"` |
| `figsize` | `tuple` | `(8, 6)` | Figure size in inches |
| `dpi` | `int` | `150` | Resolution for raster output |
| `style` | `str` | `"seaborn-v0_8-darkgrid"` | Matplotlib style |
| `save_format` | `str` | `"png"` | Default save format |
| `title` | `str\|None` | `None` | Figure title |
| `xlabel` | `str` | `"Step"` | X-axis label |
| `ylabel` | `str` | `"Episode Return"` | Y-axis label |

### CLI Script

**File:** `scripts/plot_rewards.py`

```bash
# Plot by experiment name (auto-discovers matching runs)
python scripts/plot_rewards.py --experiment-name cartpole_ppo

# Plot specific run directories
python scripts/plot_rewards.py --run-dirs runs/ppo_seed0 runs/ppo_seed1

# Compare algorithms
python scripts/plot_rewards.py --compare \
    --run-dirs runs/ppo_run runs/dqn_run runs/sac_run

# Custom y-axis and smoothing
python scripts/plot_rewards.py --experiment-name cartpole_ppo \
    --y-key total_loss --smooth-radius 20 --output loss_curve.png
```

---

## Evaluation

**File:** `src/vibe_rl/runner/evaluator.py`

Fully JIT-compiled evaluation using `jax.vmap` + `jax.lax.while_loop`:

```python
from vibe_rl.runner import evaluate, jit_evaluate

eval_metrics = evaluate(
    act_fn,         # (agent_state, obs) -> action (greedy, no state update)
    agent_state,
    env, env_params,
    n_episodes=10,
    max_steps=500,
    rng=jax.random.PRNGKey(99),
)

print(eval_metrics.mean_return)   # scalar
print(eval_metrics.std_return)    # scalar
print(eval_metrics.mean_length)   # scalar
```

All `n_episodes` episodes run in parallel via `vmap`. Each episode uses
a `while_loop` bounded by `max_steps`. A pre-jitted convenience
`jit_evaluate` is also exported.

---

## Schedules & Seeding

### Schedules

**File:** `src/vibe_rl/schedule.py`

Pure-function schedules compatible with `jax.jit`:

```python
from vibe_rl.schedule import linear_schedule

schedule = linear_schedule(start=1.0, end=0.01, steps=10_000)
eps = schedule(step)  # works inside @jax.jit
```

### Seeding

**File:** `src/vibe_rl/seeding.py`

Explicit PRNG key management (no global state):

```python
from vibe_rl.seeding import make_rng, split_key, split_keys, fold_in

rng = make_rng(42)
rng, subkey = split_key(rng)
rng, agent_key, env_key = split_keys(rng, n=2)
worker_key = fold_in(rng, worker_id)
```

---

## Presets & CLI

**File:** `src/vibe_rl/configs/presets.py`

Pre-tuned experiment configurations bundling environment, algorithm, and
runner settings:

| Preset | Environment | Algorithm | Description |
|--------|-------------|-----------|-------------|
| `cartpole_ppo` | CartPole-v1 | PPO | Fast sanity-check |
| `cartpole_dqn` | CartPole-v1 | DQN | Discrete off-policy |
| `pendulum_ppo` | Pendulum-v1 | PPO | Continuous control |
| `pendulum_sac` | Pendulum-v1 | SAC | Continuous off-policy |
| `gridworld_dqn` | GridWorld-v0 | DQN | Tabular-scale discrete |

**CLI usage:**

```bash
python scripts/train.py cartpole_ppo
python scripts/train.py cartpole_ppo --algo.lr 1e-3
python scripts/train.py pendulum_sac --runner.total_timesteps 500000
python scripts/train.py cartpole_ppo --help
```

**Programmatic usage:**

```python
from vibe_rl.configs import TrainConfig, cli, PRESETS

config = cli()                                          # parse sys.argv
config = cli(["cartpole_ppo", "--algo.lr", "1e-3"])     # explicit args
desc, config = PRESETS["cartpole_ppo"]                  # direct access
```

`TrainConfig` is a frozen dataclass with three fields:
- `env_id: str` — environment name
- `algo: PPOConfig | DQNConfig | SACConfig` — algorithm hyperparameters
- `runner: RunnerConfig` — outer loop settings
