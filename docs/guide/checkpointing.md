# Checkpointing

vibe-rl uses [Orbax](https://orbax.readthedocs.io/) with Equinox serialization for checkpointing. The system supports one-shot save/load, periodic checkpointing with retention policies, best-model tracking, and seamless resume from interruptions.

Requires the optional dependency:

```bash
pip install vibe-rl[checkpoint]
```

## Quick start

The simplest way to enable checkpointing is through `RunnerConfig`:

```python
from vibe_rl.runner import RunnerConfig, train_dqn

runner_config = RunnerConfig(
    total_timesteps=100_000,
    checkpoint_interval=5_000,   # save every 5k steps
    max_checkpoints=5,           # keep 5 most recent
    resume=True,                 # resume if checkpoints exist
)

result = train_dqn(env, env_params, dqn_config=config, runner_config=runner_config)
```

Off-policy runners (DQN, SAC) integrate checkpointing automatically. The rest of this page covers the underlying API for custom workflows.

## One-shot save / load

For saving and loading a single checkpoint outside a training loop, use the top-level functions:

```python
from vibe_rl.checkpoint import save_checkpoint, load_checkpoint

# Save a train state (creates directory structure automatically)
save_checkpoint("checkpoints/", train_state, step=1000)

# Save with metadata
save_checkpoint(
    "checkpoints/", train_state,
    step=1000,
    metadata={"env": "CartPole-v1", "algorithm": "dqn"},
)

# Load (needs a template pytree for structure)
restored = load_checkpoint("checkpoints/", like=train_state, step=1000)
```

### `save_checkpoint()`

```python
save_checkpoint(
    directory,              # root directory for the checkpoint
    pytree,                 # JAX pytree to save
    *,
    step=None,              # if given, creates step_{step}/ subdirectory
    metadata=None,          # optional JSON-serializable dict
    unreplicate=False,      # strip leading device dimension before saving
)
```

When `unreplicate=True`, the first replica is extracted (`pytree[0]`) before saving. This makes the checkpoint device-count agnostic — useful when saving from multi-GPU training.

### `load_checkpoint()`

```python
load_checkpoint(
    directory,              # root checkpoint directory
    like,                   # template pytree (structure, shapes, dtypes)
    *,
    step=None,              # if given, load from step_{step}/ subdirectory
    replicate_to=None,      # broadcast each leaf to (N, *shape)
)
```

The `like` parameter is a pytree with the same structure as the saved state. Typically a freshly initialized copy of the state — shapes and dtypes must match.

### `load_metadata()`

```python
from vibe_rl.checkpoint import load_metadata

meta = load_metadata("checkpoints/", step=1000)
# {"env": "CartPole-v1", "algorithm": "dqn"}
# Returns None if no metadata file exists
```

## Low-level Equinox serialization

For lightweight checkpointing without Orbax (no async, no retention policies):

```python
from vibe_rl.checkpoint import save_eqx, load_eqx

# Save any pytree (Equinox models, NamedTuples, optax states, etc.)
save_eqx("model.eqx", pytree)

# Load with a template
restored = load_eqx("model.eqx", like=pytree)
```

This is useful for quick experiments or when you don't need Orbax's managed features. Parent directories are created automatically.

## CheckpointManager

`CheckpointManager` handles periodic checkpointing during training loops — retention policies, async writes, and best-model tracking.

```python
from vibe_rl.checkpoint import CheckpointManager

with CheckpointManager(
    "checkpoints/",
    max_to_keep=5,
    keep_period=500,            # permanently keep every 500 steps
    save_interval_steps=100,    # only save on multiples of 100
    async_timeout_secs=7200,    # 2-hour timeout for async writes
    best_fn=lambda m: m["loss"],
    best_mode="min",
) as mgr:
    for step in range(num_steps):
        state = train_step(state)
        mgr.save(step, state, metrics={"loss": float(loss)})
    mgr.wait()  # block until all async saves complete
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `directory` | `str \| Path` | required | Root directory for all checkpoints |
| `max_to_keep` | `int` | `5` | Number of recent checkpoints to retain |
| `keep_period` | `int \| None` | `None` | Permanently keep every N-th step (never pruned) |
| `save_interval_steps` | `int` | `1` | Only save when step is a multiple of this value |
| `async_timeout_secs` | `int \| None` | `7200` | Async write timeout; `None` disables async |
| `best_fn` | `callable \| None` | `None` | `metrics -> scalar` for best-model ranking |
| `best_mode` | `str` | `"min"` | `"min"` or `"max"` for ranking direction |

### Methods

| Method | Description |
|--------|-------------|
| `save(step, pytree, *, metrics=None, metadata=None)` | Save if step is on the interval. Returns `True` if written. |
| `restore(step, like)` | Restore checkpoint into a template pytree. `step=None` restores latest. |
| `latest_step()` | Latest available step number, or `None`. |
| `all_steps()` | All checkpoint step numbers (sorted ascending). |
| `wait()` | Block until any async saves complete. |
| `close()` | Wait + close the Orbax manager. |

### Retention policies

The manager prunes old checkpoints automatically:

- **`max_to_keep=5`** — only the 5 most recent checkpoints survive.
- **`keep_period=500`** — steps 500, 1000, 1500, ... are **never** pruned, regardless of `max_to_keep`. Use this for long-term experiment archival.
- **`best_fn` + `best_mode`** — when metrics are passed to `save()`, the manager tracks which checkpoint scored best.

Example with all three:

```python
with CheckpointManager(
    ckpt_dir,
    max_to_keep=3,              # keep 3 most recent
    keep_period=10_000,         # permanently keep every 10k steps
    best_fn=lambda m: m["mean_return"],
    best_mode="max",            # higher return = better
) as mgr:
    for step in range(num_steps):
        state, metrics = train_step(state)
        mgr.save(step, state, metrics={"mean_return": float(metrics.mean_return)})
```

## Initializing checkpoint directories

`initialize_checkpoint_dir()` implements resume/overwrite semantics for the start of a training run:

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
    state = mgr.restore(step, state)
    start_step = step + 1
else:
    start_step = 0
```

### Behavior matrix

| Existing checkpoints? | `resume=True` | `overwrite=True` | Neither | Result |
|----------------------|---------------|------------------|---------|--------|
| No | - | - | - | Fresh start |
| Yes | Yes | - | - | Resume from latest |
| Yes | - | Yes | - | Wipe directory, start fresh |
| Yes | - | - | Yes | `FileExistsError` |

The function accepts all the same parameters as `CheckpointManager` (`keep_period`, `max_to_keep`, `save_interval_steps`, `async_timeout_secs`, `best_fn`, `best_mode`), forwarding them when creating the manager.

If the directory exists but contains no valid checkpoints (e.g. only stale files), `resuming` returns `False` even with `resume=True`.

## Resume training (`--resume` flag)

The DQN and SAC runners handle resume automatically through `RunnerConfig`:

```python
runner_config = RunnerConfig(
    checkpoint_dir="checkpoints/dqn_run",
    checkpoint_interval=5_000,
    max_checkpoints=5,
    keep_period=10_000,
    resume=True,
)
result = train_dqn(env, env_params, dqn_config=config, runner_config=runner_config)
```

From the CLI:

```bash
python scripts/train.py cartpole_dqn --runner.resume true
```

When `resume=True` and existing checkpoints are found:

1. The manager is initialized in resume mode.
2. The latest checkpoint step is identified.
3. Agent state (params, optimizer state, step counter, RNG) is restored.
4. Training continues from `latest_step + 1`.

If no checkpoints exist, training starts from scratch.

## RunnerConfig checkpoint fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `checkpoint_dir` | `str \| None` | `None` | Checkpoint location; `None` = no checkpointing |
| `checkpoint_interval` | `int` | `5_000` | Save every N timesteps |
| `max_checkpoints` | `int` | `5` | Recent checkpoints to retain |
| `keep_period` | `int \| None` | `None` | Permanently keep every N-th step |
| `resume` | `bool` | `False` | Resume from existing checkpoint |
| `overwrite` | `bool` | `False` | Wipe existing checkpoints before starting |

## Run directory checkpoint layout

When using `RunDir`, checkpoints are stored under the `checkpoints/` subdirectory:

```
runs/
└── dqn_cartpole_20260215_143022/
    ├── config.json
    ├── checkpoints/
    │   ├── step_5000/
    │   │   └── state.eqx
    │   ├── step_10000/
    │   │   ├── state.eqx
    │   │   └── metadata.json
    │   ├── step_15000/
    │   │   └── state.eqx
    │   └── best -> step_10000
    ├── logs/
    └── ...
```

### RunDir checkpoint helpers

```python
from vibe_rl.run_dir import RunDir

run = RunDir("dqn_cartpole")

# Path accessors
run.checkpoints                    # .../checkpoints/
run.checkpoint_dir(step=10000)     # .../checkpoints/step_10000/ (created)

# List & discover
run.list_checkpoints()             # [(5000, Path), (10000, Path), (15000, Path)]
run.latest_checkpoint              # Path to step_15000/

# Best-model management
run.mark_best(step=10000)          # creates checkpoints/best -> step_10000 symlink
run.best_checkpoint                # resolves symlink, or None

# Cleanup
run.cleanup_checkpoints(keep=5)    # prune old checkpoints, preserving best symlink target
```

`mark_best()` creates an atomic symlink — it writes to a temp name first, then renames, so it's safe even if the process is interrupted.

`cleanup_checkpoints()` never deletes the `best` symlink target, even if it falls outside the `keep` window.

## Best-model management

There are two complementary ways to track the best checkpoint:

**1. Via `CheckpointManager`** — automatic ranking using `best_fn`:

```python
mgr = CheckpointManager(
    ckpt_dir,
    best_fn=lambda m: m["eval_return"],
    best_mode="max",
)
# Orbax internally tracks which checkpoint has the best metric.
```

**2. Via `RunDir.mark_best()`** — explicit symlink for easy access:

```python
# After evaluation
if eval_return > best_eval_return:
    best_eval_return = eval_return
    run.mark_best(step=current_step)

# Later, load the best model
best_path = run.best_checkpoint  # resolves the symlink
if best_path is not None:
    best_state = load_checkpoint(best_path, like=state)
```

## Cross-device loading

Checkpoints saved from a single device can be loaded into a multi-device setup, and vice versa.

### Single GPU → Multi-GPU

Use `replicate_to` to broadcast each array to `(N, *shape)`:

```python
# Checkpoint was saved from 1 GPU
restored = load_checkpoint(
    "checkpoints/", like=train_state, step=1000,
    replicate_to=4,  # broadcast to 4 devices
)
```

### Multi-GPU → Single GPU

Use `unreplicate=True` when saving to strip the device dimension:

```python
# Save from multi-GPU training (takes first replica)
save_checkpoint("checkpoints/", train_state, step=1000, unreplicate=True)

# Load normally on a single GPU
restored = load_checkpoint("checkpoints/", like=single_device_state, step=1000)
```

This makes checkpoints portable across different hardware configurations.

## Code examples

### Full training loop with checkpointing

```python
from vibe_rl.checkpoint import CheckpointManager, initialize_checkpoint_dir
from vibe_rl.run_dir import RunDir
from vibe_rl.metrics import MetricsLogger

run = RunDir("dqn_cartpole")
run.save_config(config)

mgr, resuming = initialize_checkpoint_dir(
    run.checkpoints,
    keep_period=10_000,
    max_to_keep=5,
    save_interval_steps=5_000,
    resume=True,
)

if resuming:
    start_step = mgr.latest_step()
    state = mgr.restore(start_step, state)
    start_step += 1
else:
    start_step = 0

with mgr, MetricsLogger(run.log_path()) as logger:
    best_return = float("-inf")

    for step in range(start_step, total_steps):
        state, metrics = train_step(state)

        # Periodic checkpoint
        mgr.save(step, state, metrics={"loss": float(metrics.loss)})

        # Evaluation + best-model tracking
        if step % eval_interval == 0:
            eval_return = evaluate(state)
            logger.write({"step": step, "eval_return": float(eval_return)})

            if eval_return > best_return:
                best_return = eval_return
                run.mark_best(step=step)

    mgr.wait()
```

### Algorithm state checkpoint roundtrip

All algorithm states (DQN, PPO, SAC) are JAX pytrees and can be checkpointed directly:

```python
from vibe_rl.algorithms.dqn import DQN, DQNConfig
from vibe_rl.checkpoint import save_eqx, load_eqx

# Initialize and train
config = DQNConfig(hidden_sizes=(128, 128))
state = DQN.init(jax.random.PRNGKey(0), obs_shape=(4,), n_actions=2, config=config)

# Save
save_eqx("dqn_state.eqx", state)

# Load into a fresh template
template = DQN.init(jax.random.PRNGKey(99), obs_shape=(4,), n_actions=2, config=config)
restored = load_eqx("dqn_state.eqx", like=template)

# Verify: same actions from same observations
obs = jnp.ones(4)
a1, _ = DQN.act(state, obs, config=config, explore=False)
a2, _ = DQN.act(restored, obs, config=config, explore=False)
assert jnp.array_equal(a1, a2)
```

### Loading a checkpoint for inference only

```python
from vibe_rl.algorithms.sac import SAC, SACConfig
from vibe_rl.checkpoint import load_eqx
from vibe_rl.run_dir import RunDir

# Find the best checkpoint from a training run
run = RunDir(base_dir="runs", run_id="sac_pendulum_20260215_143022")
best_path = run.best_checkpoint

# Create a skeleton state (no memory allocated for weights)
config = SACConfig(hidden_sizes=(256, 256))
skeleton = SAC.init(jax.random.PRNGKey(0), obs_shape=(3,), n_actions=1, config=config)

# Load weights into skeleton
state = load_eqx(best_path / "state.eqx", like=skeleton)

# Use for inference
obs = env.reset()
action, _ = SAC.act(state, obs, config=config, explore=False)
```
