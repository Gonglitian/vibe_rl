# Runner API

Training runners for **vibe_rl** algorithms.

```python
from vibe_rl.runner import (
    RunnerConfig,
    train_ppo, train_ppo_multigpu, train_dqn, train_sac,
    evaluate, jit_evaluate, EvalMetrics,
    PPOTrainState, PPOMetricsHistory,
    DQNTrainResult, SACTrainResult,
    get_num_devices, replicate, unreplicate,
    shard_pytree, replicate_on_mesh, split_key_across_devices,
)
```

Two runner styles are provided:

- **PureJaxRL** (PPO) — the entire training loop lives inside `jax.lax.scan`. One JIT compilation, then zero Python overhead.
- **Hybrid** (DQN, SAC) — Python outer loop for replay-buffer management and logging, with `jax.jit`-compiled inner steps for env interaction and gradient updates.

---

## RunnerConfig

`vibe_rl.runner.config.RunnerConfig`

Shared hyperparameters for training runners. Controls the outer training loop, evaluation schedule, and logging. Algorithm-specific settings live in each algorithm's own config (e.g. `PPOConfig`, `DQNConfig`).

```python
@dataclass(frozen=True)
class RunnerConfig:
    # Training budget
    total_timesteps: int = 100_000

    # Evaluation
    eval_every: int = 5_000
    eval_episodes: int = 10

    # Logging
    log_interval: int = 1_000

    # Off-policy specific
    buffer_size: int = 100_000
    warmup_steps: int = 1_000

    # Multi-device (jit + NamedSharding)
    num_devices: int | None = None   # auto-detect if None
    num_envs: int = 1                # parallel envs per device
    fsdp_devices: int = 1            # FSDP axis size (1 = pure data-parallel)

    # Seeding
    seed: int = 0

    # Checkpointing
    checkpoint_dir: str | None = None    # None = no checkpointing
    checkpoint_interval: int = 5_000     # save every N steps
    max_checkpoints: int = 5             # recent checkpoints to retain
    keep_period: int | None = None       # permanently keep every N steps
    resume: bool = False                 # resume from existing checkpoint
    overwrite: bool = False              # wipe existing checkpoints

    # Auto-plotting
    plot: bool = True                    # generate reward curve after training
```

| Field | Type | Default | Description |
|---|---|---|---|
| `total_timesteps` | `int` | `100_000` | Total environment steps for training. |
| `eval_every` | `int` | `5_000` | Evaluate every N timesteps. |
| `eval_episodes` | `int` | `10` | Number of episodes per evaluation. |
| `log_interval` | `int` | `1_000` | Log metrics every N timesteps. |
| `buffer_size` | `int` | `100_000` | Replay buffer capacity (DQN/SAC only). |
| `warmup_steps` | `int` | `1_000` | Fill buffer with random actions before training (DQN/SAC only). |
| `num_devices` | `int \| None` | `None` | Number of devices; `None` = auto-detect all available. |
| `num_envs` | `int` | `1` | Parallel environments per device. |
| `fsdp_devices` | `int` | `1` | FSDP axis size. `1` = pure data-parallel, `>1` = shard large params. |
| `seed` | `int` | `0` | PRNG seed. |
| `checkpoint_dir` | `str \| None` | `None` | Directory for Orbax checkpoints. `None` disables checkpointing. |
| `checkpoint_interval` | `int` | `5_000` | Save a checkpoint every N steps. |
| `max_checkpoints` | `int` | `5` | Maximum number of recent checkpoints to retain. |
| `keep_period` | `int \| None` | `None` | Permanently keep a checkpoint every N steps. |
| `resume` | `bool` | `False` | Resume training from the latest checkpoint. |
| `overwrite` | `bool` | `False` | Delete existing checkpoints before starting. |
| `plot` | `bool` | `True` | Generate a reward curve plot after training completes. |

---

## train_ppo

```python
def train_ppo(
    env: Environment,
    env_params: EnvParams,
    *,
    ppo_config: PPOConfig,
    runner_config: RunnerConfig,
    obs_shape: tuple[int, ...] | None = None,
    n_actions: int | None = None,
    run_dir: RunDir | None = None,
) -> tuple[PPOTrainState, PPOMetricsHistory]
```

Train PPO from scratch using a single-JIT `lax.scan` loop.

The environment **must** be wrapped with `AutoResetWrapper` (or equivalent) so that episodes auto-reset inside the scan. When `ppo_config.num_envs > 1`, the loop uses `jax.vmap`-vectorized environments for parallel rollout collection.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `env` | `Environment` | Pure-JAX environment (must auto-reset on done). |
| `env_params` | `EnvParams` | Environment parameters. |
| `ppo_config` | `PPOConfig` | PPO algorithm hyperparameters. |
| `runner_config` | `RunnerConfig` | Outer-loop settings (total_timesteps, seed, ...). |
| `obs_shape` | `tuple[int, ...] \| None` | Observation shape. Inferred from env if `None`. |
| `n_actions` | `int \| None` | Number of discrete actions. Inferred from env if `None`. |
| `run_dir` | `RunDir \| None` | Optional `RunDir` for JSONL metrics logging. Metrics are batch-written after training completes (since PPO runs entirely inside `lax.scan`). |

**Returns:** `(PPOTrainState, PPOMetricsHistory)`

- `PPOTrainState` — final training state (agent state + env state + RNG).
- `PPOMetricsHistory` — per-update metrics, each field shape `(n_updates,)`.

**Example:**

```python
from vibe_rl.algorithms.ppo import PPOConfig
from vibe_rl.env import make
from vibe_rl.env.wrappers import AutoResetWrapper
from vibe_rl.runner import RunnerConfig, train_ppo

env, env_params = make("CartPole-v1")
env = AutoResetWrapper(env)
ppo_config = PPOConfig(n_steps=128, num_envs=8, hidden_sizes=(64, 64))
runner_config = RunnerConfig(total_timesteps=100_000)

train_state, metrics = train_ppo(
    env, env_params,
    ppo_config=ppo_config,
    runner_config=runner_config,
)
```

---

## train_ppo_multigpu

```python
def train_ppo_multigpu(
    env: Environment,
    env_params: EnvParams,
    *,
    ppo_config: PPOConfig,
    runner_config: RunnerConfig,
    obs_shape: tuple[int, ...] | None = None,
    n_actions: int | None = None,
) -> tuple[PPOTrainState, PPOMetricsHistory]
```

Train PPO with data-parallel `jit` + FSDP `NamedSharding`.

Data is shaped as `(n_devices, num_envs, *feature_dims)`. Gradient reduction is handled implicitly by JAX's GSPMD through the declared `in_shardings`/`out_shardings` on `jax.jit`. When `runner_config.fsdp_devices > 1`, large model parameters (>= 4 MB, 2-D+) are sharded across the FSDP axis.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `env` | `Environment` | Pure-JAX environment (must auto-reset on done). |
| `env_params` | `EnvParams` | Environment parameters. |
| `ppo_config` | `PPOConfig` | PPO algorithm hyperparameters. |
| `runner_config` | `RunnerConfig` | Outer-loop settings. `num_devices`, `num_envs`, and `fsdp_devices` control device distribution. |
| `obs_shape` | `tuple[int, ...] \| None` | Observation shape. Inferred from env if `None`. |
| `n_actions` | `int \| None` | Number of discrete actions. Inferred from env if `None`. |

**Returns:** `(PPOTrainState, PPOMetricsHistory)`

- `PPOTrainState` — final state with leading device dimension. Use `unreplicate()` to get a single copy.
- `PPOMetricsHistory` — fields have shape `(n_devices, n_updates)`.

**Example:**

```python
from vibe_rl.runner import RunnerConfig, train_ppo_multigpu

runner_config = RunnerConfig(total_timesteps=100_000, num_envs=4)
train_state, metrics = train_ppo_multigpu(
    env, env_params,
    ppo_config=ppo_config,
    runner_config=runner_config,
)
```

---

## train_dqn

```python
def train_dqn(
    env: Environment,
    env_params: EnvParams,
    *,
    dqn_config: DQNConfig,
    runner_config: RunnerConfig,
    obs_shape: tuple[int, ...] | None = None,
    n_actions: int | None = None,
    callback: callable | None = None,
    run_dir: RunDir | None = None,
) -> DQNTrainResult
```

Train DQN with a hybrid Python/JAX loop. The replay buffer lives in Python (numpy); everything else is jitted.

Supports checkpoint-based resume: when `runner_config.resume` is `True` and a checkpoint exists at `runner_config.checkpoint_dir`, training resumes from the saved step.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `env` | `Environment` | Pure-JAX environment (should auto-reset on done). |
| `env_params` | `EnvParams` | Environment parameters. |
| `dqn_config` | `DQNConfig` | DQN algorithm hyperparameters. |
| `runner_config` | `RunnerConfig` | Outer-loop settings. |
| `obs_shape` | `tuple[int, ...] \| None` | Observation shape. Inferred from env if `None`. |
| `n_actions` | `int \| None` | Number of discrete actions. Inferred from env if `None`. |
| `callback` | `callable \| None` | Optional `callback(step, agent_state, metrics_dict)` called every `log_interval` steps. |
| `run_dir` | `RunDir \| None` | Optional `RunDir` for JSONL metrics logging. |

**Returns:** `DQNTrainResult`

**Example:**

```python
from vibe_rl.algorithms.dqn import DQNConfig
from vibe_rl.runner import RunnerConfig, train_dqn

dqn_config = DQNConfig()
runner_config = RunnerConfig(total_timesteps=50_000)

result = train_dqn(
    env, env_params,
    dqn_config=dqn_config,
    runner_config=runner_config,
)
# result.agent_state, result.episode_returns, result.metrics_log
```

---

## train_sac

```python
def train_sac(
    env: Environment,
    env_params: EnvParams,
    *,
    sac_config: SACConfig,
    runner_config: RunnerConfig,
    obs_shape: tuple[int, ...] | None = None,
    action_dim: int | None = None,
    callback: callable | None = None,
    run_dir: RunDir | None = None,
) -> SACTrainResult
```

Train SAC with a hybrid Python/JAX loop for continuous action spaces.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `env` | `Environment` | Pure-JAX environment (should auto-reset on done). |
| `env_params` | `EnvParams` | Environment parameters. |
| `sac_config` | `SACConfig` | SAC algorithm hyperparameters. |
| `runner_config` | `RunnerConfig` | Outer-loop settings. |
| `obs_shape` | `tuple[int, ...] \| None` | Observation shape. Inferred from env if `None`. |
| `action_dim` | `int \| None` | Action dimensionality. Inferred from env if `None`. |
| `callback` | `callable \| None` | Optional `callback(step, agent_state, metrics_dict)`. |
| `run_dir` | `RunDir \| None` | Optional `RunDir` for JSONL metrics logging. |

**Returns:** `SACTrainResult`

**Example:**

```python
from vibe_rl.algorithms.sac import SACConfig
from vibe_rl.runner import RunnerConfig, train_sac

env, env_params = make("Pendulum-v1")
env = AutoResetWrapper(env)
sac_config = SACConfig()
runner_config = RunnerConfig(total_timesteps=100_000)

result = train_sac(
    env, env_params,
    sac_config=sac_config,
    runner_config=runner_config,
    action_dim=1,
)
```

---

## Return Types

### PPOTrainState

```python
class PPOTrainState(NamedTuple):
    agent_state: PPOState      # PPO agent state (params, opt_state, step, rng)
    env_obs: chex.Array        # Last observation
    env_state: EnvState        # Last environment state
    rng: chex.PRNGKey          # Outer-loop PRNG key
```

### PPOMetricsHistory

Per-update metrics collected across the entire training run. Each field has shape `(n_updates,)`.

```python
class PPOMetricsHistory(NamedTuple):
    total_loss: chex.Array
    actor_loss: chex.Array
    critic_loss: chex.Array
    entropy: chex.Array
    approx_kl: chex.Array
```

### DQNTrainResult

```python
class DQNTrainResult(NamedTuple):
    agent_state: DQNState            # Final DQN agent state
    episode_returns: list[float]     # Per-episode returns
    metrics_log: list[dict[str, float]]  # Training metrics dicts
```

### SACTrainResult

```python
class SACTrainResult(NamedTuple):
    agent_state: SACState            # Final SAC agent state
    episode_returns: list[float]     # Per-episode returns
    metrics_log: list[dict[str, float]]  # Training metrics dicts
```

---

## Evaluation

### EvalMetrics

```python
class EvalMetrics(NamedTuple):
    mean_return: chex.Array    # Mean episode return
    std_return: chex.Array     # Std of episode returns
    mean_length: chex.Array    # Mean episode length
```

### evaluate

```python
def evaluate(
    act_fn: callable,
    agent_state: chex.ArrayTree,
    env: Environment,
    env_params: EnvParams,
    *,
    n_episodes: int,
    max_steps: int,
    rng: chex.PRNGKey,
) -> EvalMetrics
```

Evaluate a greedy policy over multiple episodes in parallel.

Fully JIT-compiled. Each episode runs inside a `lax.while_loop` (bounded by `max_steps`), and the `n_episodes` episodes are vmapped in parallel.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `act_fn` | `callable` | `(agent_state, obs) -> action` — greedy action selector. Must be JIT-compatible and pure inference (no state update). |
| `agent_state` | `ArrayTree` | Agent state pytree (broadcast across episodes). |
| `env` | `Environment` | Pure-JAX environment. |
| `env_params` | `EnvParams` | Environment parameters. |
| `n_episodes` | `int` | Number of parallel evaluation episodes. |
| `max_steps` | `int` | Maximum steps per episode (must be static for XLA). |
| `rng` | `PRNGKey` | PRNG key; split across episodes. |

**Returns:** `EvalMetrics`

### jit_evaluate

```python
jit_evaluate = jax.jit(
    evaluate,
    static_argnames=("act_fn", "env", "n_episodes", "max_steps"),
)
```

Pre-jitted convenience wrapper around `evaluate`.

---

## Device Utilities

### get_num_devices

```python
def get_num_devices(requested: int | None = None) -> int
```

Return the number of devices to use. If `requested` is `None`, returns the count of all available JAX devices (GPUs/TPUs/CPUs). Raises `ValueError` if `requested` exceeds available devices.

### replicate

```python
def replicate(pytree, n_devices: int)
```

Replicate a pytree across devices by adding a leading axis. Each leaf is broadcast to shape `(n_devices, *original_shape)`.

### unreplicate

```python
def unreplicate(pytree)
```

Take the first replica from a replicated pytree. Inverse of `replicate` — removes the leading device dimension by taking `tree[0]`.

### split_key_across_devices

```python
def split_key_across_devices(rng, n_devices: int)
```

Split a PRNG key into per-device keys. Returns array of shape `(n_devices, 2)`.

### shard_pytree

```python
def shard_pytree(pytree, mesh: Mesh)
```

Place a pytree on devices with data-sharding on the leading axis. The leading dimension of each leaf is split across the mesh's data axes `("batch", "fsdp")`. Use for batched data (observations, actions, etc.).

### replicate_on_mesh

```python
def replicate_on_mesh(pytree, mesh: Mesh)
```

Replicate a pytree across all devices in a mesh. Every device gets a full copy. Use for model parameters and optimizer state.
