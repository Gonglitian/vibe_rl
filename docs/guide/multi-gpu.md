# Multi-GPU Training

vibe-rl scales PPO to multiple GPUs (or TPUs) using JAX's GSPMD partitioning with `jax.jit` + `NamedSharding`. This replaces the older `pmap`-based approach — no manual `pmean` is needed, gradient reduction is handled implicitly by declaring `in_shardings` / `out_shardings` on the jitted function.

## Quick start

```python
from vibe_rl.algorithms.ppo import PPOConfig
from vibe_rl.env import make
from vibe_rl.env.wrappers import AutoResetWrapper
from vibe_rl.runner import RunnerConfig, train_ppo_multigpu

env, env_params = make("CartPole-v1")
env = AutoResetWrapper(env)

ppo_config = PPOConfig(n_steps=128, hidden_sizes=(64, 64))
runner_config = RunnerConfig(
    total_timesteps=100_000,
    num_envs=4,          # parallel envs per device
    num_devices=None,    # auto-detect all GPUs
    fsdp_devices=1,      # 1 = pure data-parallel
)

train_state, metrics = train_ppo_multigpu(
    env, env_params,
    ppo_config=ppo_config,
    runner_config=runner_config,
)
```

The environment **must** be wrapped with `AutoResetWrapper` (or equivalent) so that episodes auto-reset inside the `lax.scan` loop.

## How it works

The training runner (`train_ppo_multigpu`) compiles the entire rollout-collection + GAE + mini-batch SGD pipeline into a single `jax.jit` call with explicit sharding annotations. JAX's GSPMD compiler then:

1. Distributes data across devices according to the declared shardings.
2. Inserts cross-device communication (all-reduce for gradients) automatically.
3. Executes the full training loop with zero Python overhead.

Data shape convention: `(n_devices, num_envs, *feature_dims)`. Each device shard runs `num_envs` parallel environments. Inside each shard, `vmap` vectorizes across environments.

**Total timesteps per update** = `n_devices * num_envs * n_steps`.

## Device mesh

The core abstraction is a 2-D device mesh with axes `("batch", "fsdp")`:

```python
from vibe_rl.sharding import make_mesh

# Pure data-parallel (fsdp axis is trivial)
mesh = make_mesh(num_fsdp_devices=1)

# 2-way FSDP + data-parallel
mesh = make_mesh(num_fsdp_devices=2)

# Explicit device count
mesh = make_mesh(num_fsdp_devices=1, num_devices=4)
```

`make_mesh` reshapes `jax.devices()[:num_devices]` into a `(batch, fsdp)` grid. When `num_fsdp_devices=1`, the FSDP axis is size 1 and every parameter is replicated — equivalent to pure data-parallelism.

::: info
`num_devices` must be evenly divisible by `num_fsdp_devices`. For example, 8 GPUs with `num_fsdp_devices=2` creates a `(4, 2)` mesh: 4-way data-parallel, 2-way FSDP.
:::

### `set_mesh` context manager

For manual use outside the training runner:

```python
from vibe_rl.sharding import set_mesh

with set_mesh(num_fsdp_devices=1) as mesh:
    jitted_fn = jax.jit(fn, in_shardings=..., out_shardings=...)
    result = jitted_fn(data)
```

## Data sharding

Two sharding specs cover most use cases:

```python
from vibe_rl.sharding import data_sharding, replicate_sharding

data_spec = data_sharding(mesh)       # PartitionSpec(("batch", "fsdp"))
param_spec = replicate_sharding(mesh) # PartitionSpec()  — full copy everywhere
```

| Spec | Use case | PartitionSpec |
|------|----------|---------------|
| `data_sharding` | Observations, actions, rewards — anything with a leading batch dimension | `(("batch", "fsdp"),)` |
| `replicate_sharding` | Model parameters, optimizer state (when not using FSDP) | `()` |

## FSDP parameter sharding

When models are large, fully replicating parameters across all devices wastes memory. FSDP (Fully Sharded Data Parallelism) shards large parameters across the `fsdp` mesh axis.

```python
from vibe_rl.sharding import fsdp_sharding

param_shardings = fsdp_sharding(params_abstract, mesh, min_size_mbytes=4)
```

`fsdp_sharding` walks a pytree of arrays (or `jax.ShapeDtypeStruct` from `jax.eval_shape`) and decides per-parameter:

| Condition | Result |
|-----------|--------|
| Size < 4 MB (configurable via `min_size_mbytes`) | Replicated — communication overhead exceeds savings |
| Scalar or 1-D array | Replicated — cannot meaningfully shard |
| 2-D+ array >= threshold, divisible dimension exists | Sharded along the **largest** dimension that is evenly divisible by the FSDP axis size |
| 2-D+ array >= threshold, no divisible dimension | Replicated (fallback) |

When `mesh.shape["fsdp"] == 1`, every parameter is replicated regardless of size — the mesh collapses to pure data-parallelism.

### Example: mixed sharding in a pytree

```python
import jax
import jax.numpy as jnp
from vibe_rl.sharding import make_mesh, fsdp_sharding

mesh = make_mesh(num_fsdp_devices=2)

params = {
    "small_weight": jax.ShapeDtypeStruct((8, 8), jnp.float32),        # 256 B → replicated
    "bias": jax.ShapeDtypeStruct((64,), jnp.float32),                  # 1-D → replicated
    "large_weight": jax.ShapeDtypeStruct((2048, 1024), jnp.float32),  # 8 MB → sharded
}

shardings = fsdp_sharding(params, mesh)
# shardings["small_weight"].spec == ()              (replicated)
# shardings["bias"].spec == ()                      (replicated)
# shardings["large_weight"].spec == ("fsdp", None)  (sharded along dim 0)
```

## Configuring `fsdp_devices`

The `fsdp_devices` parameter in `RunnerConfig` controls the FSDP axis size:

| `fsdp_devices` | Behavior |
|----------------|----------|
| `1` (default) | Pure data-parallelism. All parameters replicated. |
| `2` | 2-way FSDP. Large parameters split across 2 devices. Remaining devices do data-parallelism. |
| `N` | N-way FSDP. Must evenly divide total device count. |

```python
# 8 GPUs: 4-way data-parallel, 2-way FSDP
runner_config = RunnerConfig(
    num_devices=8,
    num_envs=4,
    fsdp_devices=2,
)
```

## Device utilities

`device_utils` provides helpers for managing data across devices:

```python
from vibe_rl.runner.device_utils import (
    get_num_devices,
    replicate,
    unreplicate,
    split_key_across_devices,
    shard_pytree,
    replicate_on_mesh,
)
```

### `get_num_devices`

Auto-detect or validate the number of available devices:

```python
n = get_num_devices()       # returns jax.local_device_count()
n = get_num_devices(4)      # returns 4 (raises ValueError if < 4 available)
```

### `replicate` / `unreplicate`

Add or remove a leading device dimension on pytrees:

```python
# Replicate: (3, 4) → (n_devices, 3, 4)
replicated = replicate(state, n_devices=4)

# Unreplicate: take first replica (n_devices, 3, 4) → (3, 4)
single = unreplicate(replicated)
```

These work on any JAX pytree — dicts, NamedTuples, Equinox modules, etc.

### `shard_pytree` / `replicate_on_mesh`

Place data onto a mesh using `jax.device_put`:

```python
from vibe_rl.sharding import make_mesh

mesh = make_mesh()

# Shard leading axis across data axes
sharded_data = shard_pytree(batch, mesh)

# Full replication (e.g. for model params)
replicated_params = replicate_on_mesh(params, mesh)
```

### `split_key_across_devices`

Split a PRNG key into per-device keys:

```python
device_keys = split_key_across_devices(rng, n_devices=4)
# shape: (4, 2)
```

## `train_ppo_multigpu` details

The full training function signature:

```python
def train_ppo_multigpu(
    env: Environment,
    env_params: EnvParams,
    *,
    ppo_config: PPOConfig,
    runner_config: RunnerConfig,
    obs_shape: tuple[int, ...] | None = None,
    n_actions: int | None = None,
) -> tuple[PPOTrainState, PPOMetricsHistory]:
```

**Arguments:**

| Parameter | Description |
|-----------|-------------|
| `env` | Pure-JAX environment (must auto-reset on done) |
| `env_params` | Environment parameters |
| `ppo_config` | PPO algorithm hyperparameters |
| `runner_config` | Outer-loop settings — `total_timesteps`, `num_devices`, `num_envs`, `fsdp_devices`, `seed` |
| `obs_shape` | Observation shape. Inferred from `env` if `None` |
| `n_actions` | Number of discrete actions. Inferred from `env` if `None` |

**Returns:**

- `PPOTrainState` — final agent state, env state, obs, and RNG.
- `PPOMetricsHistory` — per-update scalars with shape `(n_updates,)`: `total_loss`, `actor_loss`, `critic_loss`, `entropy`, `approx_kl`.

**How `n_updates` is computed:**

```
n_updates = total_timesteps // (n_devices * num_envs * n_steps)
```

A `ValueError` is raised if `total_timesteps` is less than one update's worth of steps.

### Gradient synchronization

With GSPMD, there is **one logical copy** of model parameters. The compiler sees the declared shardings on `jax.jit` and automatically inserts all-reduce operations where needed. No manual `jax.lax.pmean` calls are required.

```python
# Inside train_ppo_multigpu — the key jit call:
jitted_train = jax.jit(
    _train_loop,
    in_shardings=(train_state_shardings,),
    out_shardings=(train_state_shardings, metrics_shardings),
    donate_argnums=(0,),
)
```

### Checkpointing multi-GPU state

With GSPMD, agent state params are a single logical copy — you can save them directly:

```python
from vibe_rl.checkpoint import save_checkpoint, load_checkpoint

# Save directly (no unreplicate needed with GSPMD)
save_checkpoint("checkpoints/", train_state.agent_state)

# Load into a single-device template
template = PPO.init(jax.random.PRNGKey(0), obs_shape=(4,), n_actions=2, config=ppo_config)
loaded = load_checkpoint("checkpoints/", template)
```

For the older `replicate`/`unreplicate` workflow (e.g. legacy `pmap` code):

```python
# Save: strip device dimension
save_checkpoint("checkpoints/", train_state, unreplicate=True)

# Load: restore and replicate
loaded = load_checkpoint("checkpoints/", template, replicate_to=4)
```

## Testing without real GPUs

Use `XLA_FLAGS` to simulate multiple devices on a single CPU:

```bash
XLA_FLAGS="--xla_force_host_platform_device_count=4" python my_script.py
```

::: warning
The environment variable must be set **before** JAX is imported. If JAX is already initialized, the flag has no effect.
:::

In test files, set it at the top of the module before any JAX imports:

```python
import os
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_force_host_platform_device_count=4",
)

import jax  # now sees 4 fake CPU devices
```

Then use `num_devices` in `RunnerConfig` to control how many simulated devices to use:

```python
runner_config = RunnerConfig(
    total_timesteps=64,
    num_devices=2,    # use 2 of the 4 fake devices
    num_envs=2,
)

train_state, history = train_ppo_multigpu(
    env, env_params,
    ppo_config=ppo_config,
    runner_config=runner_config,
)
```

## Full example

A complete multi-GPU training script:

```python
from vibe_rl.algorithms.ppo import PPOConfig
from vibe_rl.env import make
from vibe_rl.env.wrappers import AutoResetWrapper
from vibe_rl.runner import RunnerConfig, train_ppo_multigpu
from vibe_rl.runner.device_utils import get_num_devices, unreplicate
from vibe_rl.checkpoint import save_checkpoint

# Environment
env, env_params = make("CartPole-v1")
env = AutoResetWrapper(env)

# Config
ppo_config = PPOConfig(
    n_steps=128,
    n_minibatches=4,
    n_epochs=4,
    hidden_sizes=(64, 64),
)
runner_config = RunnerConfig(
    total_timesteps=100_000,
    num_envs=4,
    fsdp_devices=1,      # pure data-parallel
)

print(f"Training on {get_num_devices()} devices")

# Train
train_state, metrics = train_ppo_multigpu(
    env, env_params,
    ppo_config=ppo_config,
    runner_config=runner_config,
)

# Save checkpoint (GSPMD: params are single logical copy)
save_checkpoint("checkpoints/multigpu_run", train_state.agent_state)

print(f"Final loss: {float(metrics.total_loss[-1]):.4f}")
```

## API summary

### `vibe_rl.sharding`

| Function | Description |
|----------|-------------|
| `make_mesh(num_fsdp_devices, num_devices)` | Create a 2-D `(batch, fsdp)` device mesh |
| `set_mesh(num_fsdp_devices, num_devices)` | Context manager — creates and yields a mesh |
| `data_sharding(mesh)` | `NamedSharding` for batch data (leading axis split over both mesh axes) |
| `replicate_sharding(mesh)` | `NamedSharding` for full replication |
| `fsdp_sharding(pytree, mesh, min_size_mbytes)` | Per-parameter FSDP sharding based on size/shape |
| `activation_sharding_constraint(x, mesh)` | Apply `with_sharding_constraint` inside jit-compiled functions |

### `vibe_rl.runner.device_utils`

| Function | Description |
|----------|-------------|
| `get_num_devices(requested)` | Auto-detect or validate device count |
| `replicate(pytree, n_devices)` | Add leading `(n_devices,)` axis |
| `unreplicate(pytree)` | Take first replica `pytree[0]` |
| `split_key_across_devices(rng, n_devices)` | Split PRNG key for per-device randomness |
| `shard_pytree(pytree, mesh)` | Place data on mesh with leading-axis sharding |
| `replicate_on_mesh(pytree, mesh)` | Place data on mesh, fully replicated |

### `RunnerConfig` multi-GPU fields

| Field | Default | Description |
|-------|---------|-------------|
| `num_devices` | `None` | Device count (`None` = auto-detect all) |
| `num_envs` | `1` | Parallel environments per device |
| `fsdp_devices` | `1` | FSDP axis size (`1` = pure data-parallel) |
