# Orbax & JAX Checkpointing: Comprehensive Research

> Research date: 2026-02-15
> Sources: Orbax official docs, Flax docs, Equinox docs, HuggingFace safetensors docs, GitHub issues/discussions

---

## Table of Contents

1. [Orbax Core API](#1-orbax-core-api)
2. [Integration with Flax NNX](#2-integration-with-flax-nnx)
3. [Integration with Equinox](#3-integration-with-equinox)
4. [Checkpointing Strategies](#4-checkpointing-strategies)
5. [What Gets Checkpointed in RL](#5-what-gets-checkpointed-in-rl)
6. [Alternatives](#6-alternatives)
7. [Performance: Async & Large Model Support](#7-performance-async--large-model-support)
8. [Recommendations](#8-recommendations)

---

## 1. Orbax Core API

**Package**: `orbax-checkpoint` (install via `pip install orbax-checkpoint`)
**Import**: `import orbax.checkpoint as ocp`

Orbax provides three hierarchical API layers:

```
CheckpointManager  (high-level: training loop management)
    -> Checkpointer  (mid-level: single checkpoint save/restore with atomicity)
        -> CheckpointHandler  (low-level: actual serialization logic)
```

### 1.1 StandardCheckpointer (Simplest API)

The simplest way to save/restore a single PyTree:

```python
import orbax.checkpoint as ocp
import jax.numpy as jnp

# Your training state as a PyTree (dict, namedtuple, dataclass, etc.)
state = {
    'params': {'w': jnp.ones((3, 4)), 'b': jnp.zeros(4)},
    'step': 0,
}

# --- SAVE ---
checkpointer = ocp.StandardCheckpointer()
checkpointer.save('/tmp/my_checkpoint', state)

# --- RESTORE ---
# You need an "abstract" version of the state (shapes/dtypes, no data).
# For simple cases, just pass the original or a template.
import jax
abstract_state = jax.tree.map(
    lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype) if hasattr(x, 'shape') else x,
    state
)
restored = checkpointer.restore('/tmp/my_checkpoint', abstract_state)
```

### 1.2 Checkpointer with CompositeCheckpointHandler

For checkpoints with multiple heterogeneous items (e.g., PyTree state + JSON metadata):

```python
checkpointer = ocp.Checkpointer(ocp.CompositeCheckpointHandler())

# --- SAVE ---
checkpointer.save(
    '/tmp/composite_checkpoint',
    args=ocp.args.Composite(
        state=ocp.args.StandardSave(train_state),
        metadata=ocp.args.JsonSave({'learning_rate': 0.001, 'epoch': 5}),
    ),
)

# --- RESTORE ---
result = checkpointer.restore(
    '/tmp/composite_checkpoint',
    args=ocp.args.Composite(
        state=ocp.args.StandardRestore(abstract_state),
        metadata=ocp.args.JsonRestore(),
    ),
)
print(result.state)     # restored PyTree
print(result.metadata)  # {'learning_rate': 0.001, 'epoch': 5}
```

### 1.3 CheckpointManager (Recommended for Training)

The high-level API for managing a sequence of checkpoints during training:

```python
import orbax.checkpoint as ocp
from pathlib import Path

checkpoint_dir = Path('/tmp/training_checkpoints')

options = ocp.CheckpointManagerOptions(
    save_interval_steps=100,       # save every 100 steps
    max_to_keep=5,                 # keep only the 5 most recent
    enable_async_checkpointing=True,  # async by default (True)
)

with ocp.CheckpointManager(checkpoint_dir, options=options) as mngr:
    for step in range(10000):
        state = train_step(state)

        # save() respects save_interval_steps automatically
        mngr.save(step, args=ocp.args.StandardSave(state))

    # CRITICAL: wait for async saves to finish before exiting
    mngr.wait_until_finished()

# --- RESTORE ---
with ocp.CheckpointManager(checkpoint_dir, options=options) as mngr:
    latest = mngr.latest_step()           # e.g., 9900
    all_steps = mngr.all_steps()          # e.g., [9500, 9600, 9700, 9800, 9900]
    restored = mngr.restore(latest)       # restore latest
```

### 1.4 CheckpointManager with Multiple Items

```python
options = ocp.CheckpointManagerOptions(
    save_interval_steps=100,
    max_to_keep=3,
)

with ocp.CheckpointManager(
    checkpoint_dir,
    options=options,
    item_names=('state', 'metadata'),
) as mngr:
    mngr.save(
        step,
        args=ocp.args.Composite(
            state=ocp.args.StandardSave(train_state),
            metadata=ocp.args.JsonSave({'loss': float(loss), 'lr': lr}),
        ),
    )
```

### 1.5 Orbax v1 API (New, 2025+)

Orbax is introducing a simplified v1 API (`orbax.checkpoint.v1`) that unifies `CheckpointManager` and `Checkpointer`:

```python
# v1 top-level functions (simpler)
from orbax.checkpoint.v1 import save_pytree, load_pytree

save_pytree(path, state)
restored = load_pytree(path, abstract_state)

# v1 async variants
from orbax.checkpoint.v1 import save_pytree_async, load_pytree_async

# v1 training Checkpointer (replaces CheckpointManager)
from orbax.checkpoint.v1.training import Checkpointer
from orbax.checkpoint.v1 import preservation_policy_lib, save_decision_policy_lib

ckptr = Checkpointer(
    directory=checkpoint_dir,
    save_decision_policy=save_decision_policy_lib.FixedIntervalPolicy(100),
    preservation_policy=preservation_policy_lib.LatestN(5),
)
```

**Migration note**: v0 API continues to work. v1 is optional but recommended for new projects.

---

## 2. Integration with Flax NNX

Flax NNX (the new API, replacing Linen) uses `nnx.split()` and `nnx.merge()` to convert models to/from checkpointable PyTrees.

### 2.1 Basic Save/Restore Pattern

```python
from flax import nnx
import orbax.checkpoint as ocp

# Define an NNX model
class MLP(nnx.Module):
    def __init__(self, din, dhidden, dout, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(din, dhidden, rngs=rngs)
        self.linear2 = nnx.Linear(dhidden, dout, rngs=rngs)

    def __call__(self, x):
        return self.linear2(nnx.relu(self.linear1(x)))

# Create model
model = MLP(4, 32, 2, rngs=nnx.Rngs(0))

# --- SAVE ---
# Step 1: Split model into GraphDef + State
graphdef, state = nnx.split(model)

# Step 2: Save state (which is a PyTree of nnx.Variable values)
checkpointer = ocp.StandardCheckpointer()
checkpointer.save('/tmp/flax_ckpt/state', state)

# --- RESTORE ---
# Step 1: Create abstract model (no memory allocation)
abstract_model = nnx.eval_shape(lambda: MLP(4, 32, 2, rngs=nnx.Rngs(0)))
graphdef, abstract_state = nnx.split(abstract_model)

# Step 2: Restore state using the abstract template
state_restored = checkpointer.restore('/tmp/flax_ckpt/state', abstract_state)

# Step 3: Merge back into a live model
model = nnx.merge(graphdef, state_restored)
```

### 2.2 Full Training State (Model + Optimizer)

```python
from flax import nnx
import optax
import orbax.checkpoint as ocp

# Create model and optimizer
model = MLP(4, 32, 2, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-3))

# --- SAVE entire training state ---
# nnx.state() extracts the full state including optimizer
_, state = nnx.split(optimizer)  # optimizer contains model
checkpointer = ocp.StandardCheckpointer()
checkpointer.save('/tmp/train_ckpt/state', state)

# --- RESTORE ---
# Recreate abstract optimizer (no memory)
abstract_model = nnx.eval_shape(lambda: MLP(4, 32, 2, rngs=nnx.Rngs(0)))
abstract_optimizer = nnx.eval_shape(
    lambda m: nnx.Optimizer(m, optax.adam(1e-3)), abstract_model
)
graphdef, abstract_state = nnx.split(abstract_optimizer)

state_restored = checkpointer.restore('/tmp/train_ckpt/state', abstract_state)
optimizer = nnx.merge(graphdef, state_restored)
# optimizer.model gives you back the model
```

### 2.3 With CheckpointManager

```python
from flax import nnx
import orbax.checkpoint as ocp

options = ocp.CheckpointManagerOptions(
    save_interval_steps=500,
    max_to_keep=3,
)

with ocp.CheckpointManager('/tmp/flax_training', options=options) as mngr:
    for step in range(num_steps):
        # ... training logic ...

        # Extract state for saving
        _, state = nnx.split(optimizer)
        mngr.save(step, args=ocp.args.StandardSave(state))

    mngr.wait_until_finished()

# Restore
with ocp.CheckpointManager('/tmp/flax_training', options=options) as mngr:
    abstract_model = nnx.eval_shape(lambda: MLP(4, 32, 2, rngs=nnx.Rngs(0)))
    abstract_opt = nnx.eval_shape(
        lambda m: nnx.Optimizer(m, optax.adam(1e-3)), abstract_model
    )
    graphdef, abstract_state = nnx.split(abstract_opt)

    state = mngr.restore(mngr.latest_step(), args=ocp.args.StandardRestore(abstract_state))
    optimizer = nnx.merge(graphdef, state)
```

### Key NNX Functions

| Function | Purpose |
|---|---|
| `nnx.split(module)` | Returns `(GraphDef, State)` -- State is a checkpointable PyTree |
| `nnx.merge(graphdef, state)` | Reconstructs module from GraphDef + State |
| `nnx.eval_shape(fn)` | Creates abstract module without allocating arrays |
| `nnx.state(module)` | Extracts State without GraphDef |
| `nnx.Optimizer` | Wraps model + optax optimizer, state includes both |

---

## 3. Integration with Equinox

Equinox has its own built-in serialization that is simpler than Orbax but less feature-rich.

### 3.1 Equinox Built-in Serialization

```python
import equinox as eqx
import jax
import jax.numpy as jnp

# Define an Equinox model
class MLP(eqx.Module):
    layers: list

    def __init__(self, key):
        key1, key2 = jax.random.split(key)
        self.layers = [
            eqx.nn.Linear(4, 32, key=key1),
            eqx.nn.Linear(32, 2, key=key2),
        ]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x)

model = MLP(jax.random.PRNGKey(0))

# --- SAVE ---
eqx.tree_serialise_leaves("model.eqx", model)

# --- RESTORE ---
# Need a "skeleton" model with correct structure (shapes/dtypes)
skeleton = MLP(jax.random.PRNGKey(0))  # or use eqx.filter_eval_shape
model_restored = eqx.tree_deserialise_leaves("model.eqx", skeleton)
```

### 3.2 Saving Model + Optimizer State

```python
import equinox as eqx
import optax

model = MLP(jax.random.PRNGKey(0))
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

# Save both as a tuple
eqx.tree_serialise_leaves("checkpoint.eqx", (model, opt_state))

# Restore
skeleton_model = MLP(jax.random.PRNGKey(0))
skeleton_opt_state = optimizer.init(eqx.filter(skeleton_model, eqx.is_array))
model_restored, opt_state_restored = eqx.tree_deserialise_leaves(
    "checkpoint.eqx", (skeleton_model, skeleton_opt_state)
)
```

### 3.3 Saving with Hyperparameters (JSON + Weights)

```python
import json
import equinox as eqx

def save_checkpoint(filename, hyperparams, model):
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)

def load_checkpoint(filename, model_factory):
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        model = model_factory(**hyperparams)
        model = eqx.tree_deserialise_leaves(f, model)
    return hyperparams, model

# Usage
save_checkpoint("ckpt.eqx", {"dim": 4, "hidden": 32}, model)
hyperparams, model = load_checkpoint("ckpt.eqx", lambda **kw: MLP(jax.random.PRNGKey(0)))
```

### 3.4 Memory-Efficient Restore with filter_eval_shape

```python
# Create abstract skeleton without allocating memory
skeleton = eqx.filter_eval_shape(MLP, jax.random.PRNGKey(0))
model = eqx.tree_deserialise_leaves("model.eqx", skeleton)
```

### 3.5 Custom Equinox Handler for Orbax

If you want Equinox models managed by Orbax's CheckpointManager (for periodic/best-model checkpointing):

```python
import equinox as eqx
import orbax.checkpoint as ocp
from etils import epath

class EquinoxSave(ocp.args.CheckpointArgs):
    item: eqx.Module

class EquinoxRestore(ocp.args.CheckpointArgs):
    item: eqx.Module  # skeleton

class EquinoxCheckpointHandler(ocp.CheckpointHandler):
    def save(self, directory: epath.Path, args: EquinoxSave):
        full_path = directory / "model.eqx"
        # IMPORTANT: convert path to str to avoid .npy extension bug
        eqx.tree_serialise_leaves(str(full_path), args.item)

    def restore(self, directory: epath.Path, args: EquinoxRestore) -> eqx.Module:
        full_path = directory / "model.eqx"
        return eqx.tree_deserialise_leaves(str(full_path), args.item)

    def metadata(self, directory):
        return None

# Usage with CheckpointManager
handler_registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
handler_registry.add('model', EquinoxSave, handler=EquinoxCheckpointHandler())
handler_registry.add('model', EquinoxRestore, handler=EquinoxCheckpointHandler())

options = ocp.CheckpointManagerOptions(max_to_keep=5, save_interval_steps=100)
with ocp.CheckpointManager(
    '/tmp/eqx_checkpoints',
    options=options,
    handler_registry=handler_registry,
    item_names=('model', 'metadata'),
) as mngr:
    mngr.save(step, args=ocp.args.Composite(
        model=EquinoxSave(model),
        metadata=ocp.args.JsonSave({'step': step, 'loss': float(loss)}),
    ))
```

**IMPORTANT**: When using `eqx.tree_serialise_leaves` with Orbax's `epath.Path`, always convert to `str()` first. The epath.Path type causes Equinox to append `.npy` and corrupt the output (see orbax issue #741).

### 3.6 Equinox Serialization API Reference

| Function | Signature |
|---|---|
| `eqx.tree_serialise_leaves(path_or_file, pytree, filter_spec=..., is_leaf=None)` | Save leaves to file |
| `eqx.tree_deserialise_leaves(path_or_file, like, filter_spec=..., is_leaf=None)` | Load leaves from file |
| `eqx.default_serialise_filter_spec(f, x)` | Default: saves jax arrays, numpy arrays, python scalars |
| `eqx.default_deserialise_filter_spec(f, x)` | Default: loads arrays, keeps other leaves as-is |

---

## 4. Checkpointing Strategies

### 4.1 Periodic Checkpointing

```python
# Using save_interval_steps
options = ocp.CheckpointManagerOptions(
    save_interval_steps=1000,  # every 1000 steps
    max_to_keep=5,             # keep last 5
)

# Using save_on_steps for specific steps
options = ocp.CheckpointManagerOptions(
    save_on_steps=frozenset({0, 1000, 5000, 10000, 50000}),
)

# Using should_save_fn for custom logic
def should_save(step, latest_step):
    # Save on powers of 10
    import math
    return step > 0 and math.log10(step) == int(math.log10(step))

options = ocp.CheckpointManagerOptions(should_save_fn=should_save)

# V1 API: FixedIntervalPolicy
from orbax.checkpoint.v1 import save_decision_policy_lib, preservation_policy_lib

options = ocp.CheckpointManagerOptions(
    save_decision_policy=save_decision_policy_lib.FixedIntervalPolicy(1000),
    preservation_policy=preservation_policy_lib.LatestN(5),
)
```

### 4.2 Best-Model Checkpointing

```python
# Keep top-k checkpoints by metric (e.g., lowest validation loss)
options = ocp.CheckpointManagerOptions(
    max_to_keep=3,              # keep top 3
    best_fn=lambda metrics: metrics['val_loss'],  # scalar to rank by
    best_mode='min',            # lower is better ('max' for accuracy)
    keep_checkpoints_without_metrics=True,  # keep ckpts with no metrics
)

with ocp.CheckpointManager('/tmp/best_ckpts', options=options) as mngr:
    for step in range(num_steps):
        state, loss = train_step(state)

        if step % eval_interval == 0:
            val_loss = evaluate(state)
            # Pass metrics to save() for ranking
            mngr.save(
                step,
                args=ocp.args.StandardSave(state),
                metrics={'val_loss': val_loss},
            )
        else:
            mngr.save(step, args=ocp.args.StandardSave(state))

    mngr.wait_until_finished()

# V1 API: BestN preservation policy
options = ocp.CheckpointManagerOptions(
    preservation_policy=preservation_policy_lib.BestN(
        n=3,
        metric_fn=lambda metrics: metrics['val_loss'],
        mode='min',
    ),
)
```

### 4.3 Keeping Both Best AND Latest

A common need: keep the N best checkpoints by metric PLUS the most recent checkpoint (for crash recovery). The Orbax team acknowledged this is a gap (issue #526). Workarounds:

```python
# Approach 1: Use keep_checkpoints_without_metrics
# - Only pass metrics when you're OK with that checkpoint being ranked
# - Recent checkpoints without metrics are always kept
options = ocp.CheckpointManagerOptions(
    max_to_keep=5,
    best_fn=lambda m: m['val_loss'],
    best_mode='min',
    keep_checkpoints_without_metrics=True,  # <-- key setting
)

# Approach 2: Two separate CheckpointManagers
best_mngr = ocp.CheckpointManager('/tmp/best', options=best_options)
latest_mngr = ocp.CheckpointManager('/tmp/latest', options=latest_options)

# Approach 3: Manual with should_keep_fn (deprecated in favor of preservation_policy)
# Approach 4 (v1 API): Combine policies
# preservation_policy=CompositePolicy(LatestN(2), BestN(3, ...))
```

### 4.4 Top-K Checkpoint Summary

| Strategy | Options |
|---|---|
| Keep latest N | `max_to_keep=N` (or `LatestN(N)` in v1) |
| Keep best N by metric | `max_to_keep=N, best_fn=..., best_mode='min'/'max'` |
| Keep every Kth step | `keep_period=K` |
| Keep by time interval | `keep_time_interval=timedelta(hours=1)` |
| Custom retention | `should_keep_fn=lambda step, latest: ...` |
| Save at specific steps | `save_on_steps=frozenset({...})` |

---

## 5. What Gets Checkpointed in RL

### 5.1 Complete RL Training State

For RL, the full resumable training state typically includes:

```python
import jax
import jax.numpy as jnp
from typing import NamedTuple

class RLTrainState(NamedTuple):
    """Complete RL training state as a PyTree."""
    # Core model
    params: dict              # policy/value network parameters
    target_params: dict       # target network parameters (DQN, SAC, etc.)

    # Optimizer
    opt_state: dict           # optax optimizer state (momentum, etc.)

    # Training progress
    step: int                 # global training step
    episodes: int             # total episodes completed
    env_steps: int            # total environment steps

    # RNG state
    rng: jax.Array            # JAX PRNG key for reproducibility

    # Optional: running statistics
    obs_mean: jax.Array       # observation normalization mean
    obs_var: jax.Array        # observation normalization variance
```

### 5.2 Replay Buffer Considerations

Replay buffers are the trickiest part of RL checkpointing:

**Option A: Do NOT checkpoint the buffer** (simpler, recommended for most cases)
- Buffers are large (millions of transitions x state dim)
- For on-policy algorithms (PPO, A2C), the buffer is discarded each iteration anyway
- For off-policy (SAC, DQN), the buffer can be re-filled relatively quickly

**Option B: Checkpoint the buffer** (for exact reproducibility or expensive environments)
```python
# If buffer is a JAX pytree (e.g., Flashbax), it's directly checkpointable
from flashbax import make_flat_buffer

buffer = make_flat_buffer(max_length=100000, min_length=1000, sample_batch_size=256)
buffer_state = buffer.init(example_timestep)

# Save everything as one composite checkpoint
with ocp.CheckpointManager(ckpt_dir, options=options) as mngr:
    mngr.save(
        step,
        args=ocp.args.Composite(
            train_state=ocp.args.StandardSave(train_state),
            buffer_state=ocp.args.StandardSave(buffer_state),
            metadata=ocp.args.JsonSave({
                'step': int(step),
                'buffer_size': int(buffer_state.current_index),
            }),
        ),
    )
```

**Option C: Checkpoint buffer separately** (for large buffers)
```python
# Use Flashbax Vault for persistent buffer storage
from flashbax.vault import Vault

vault = Vault(
    vault_name="replay_buffer",
    experience=buffer_state.experience,
    vault_uid="experiment_001",
)
# Vault writes slices to disk incrementally
vault.write(buffer_state)
# Read back
buffer_state = vault.read()
```

### 5.3 Complete RL Checkpointing Example

```python
import orbax.checkpoint as ocp
import optax
from flax import nnx

# -- Setup --
policy = PolicyNetwork(obs_dim=8, act_dim=4, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(policy, optax.adam(3e-4))

options = ocp.CheckpointManagerOptions(
    save_interval_steps=10000,
    max_to_keep=3,
    best_fn=lambda m: m['mean_return'],
    best_mode='max',
    keep_checkpoints_without_metrics=True,
    enable_async_checkpointing=True,
)

with ocp.CheckpointManager(
    '/tmp/rl_experiment',
    options=options,
    item_names=('train_state', 'metadata'),
) as mngr:

    # -- Check for existing checkpoint --
    if mngr.latest_step() is not None:
        # Restore
        abstract_model = nnx.eval_shape(lambda: PolicyNetwork(8, 4, rngs=nnx.Rngs(0)))
        abstract_opt = nnx.eval_shape(
            lambda m: nnx.Optimizer(m, optax.adam(3e-4)), abstract_model
        )
        graphdef, abstract_state = nnx.split(abstract_opt)

        result = mngr.restore(
            mngr.latest_step(),
            args=ocp.args.Composite(
                train_state=ocp.args.StandardRestore(abstract_state),
                metadata=ocp.args.JsonRestore(),
            ),
        )
        optimizer = nnx.merge(graphdef, result.train_state)
        start_step = result.metadata['step']
        print(f"Resumed from step {start_step}")
    else:
        start_step = 0

    # -- Training loop --
    for step in range(start_step, total_steps):
        # ... RL training logic ...

        _, state = nnx.split(optimizer)

        if step % eval_interval == 0:
            mean_return = evaluate_policy(optimizer.model)
            mngr.save(
                step,
                args=ocp.args.Composite(
                    train_state=ocp.args.StandardSave(state),
                    metadata=ocp.args.JsonSave({
                        'step': step,
                        'mean_return': float(mean_return),
                    }),
                ),
                metrics={'mean_return': float(mean_return)},
            )
        else:
            _, state = nnx.split(optimizer)
            mngr.save(
                step,
                args=ocp.args.Composite(
                    train_state=ocp.args.StandardSave(state),
                    metadata=ocp.args.JsonSave({'step': step}),
                ),
            )

    mngr.wait_until_finished()
```

### 5.4 Equinox RL Checkpointing Example

```python
import equinox as eqx
import optax
import jax

model = PolicyMLP(key=jax.random.PRNGKey(0))
optimizer = optax.adam(3e-4)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

# Checkpoint = (model, opt_state, step, rng_key)
checkpoint = (model, opt_state, step, rng_key)

# Save
eqx.tree_serialise_leaves(f"checkpoints/step_{step}.eqx", checkpoint)

# Restore
skeleton_model = eqx.filter_eval_shape(PolicyMLP, key=jax.random.PRNGKey(0))
skeleton_opt_state = optimizer.init(eqx.filter(skeleton_model, eqx.is_array))
skeleton = (skeleton_model, skeleton_opt_state, 0, jax.random.PRNGKey(0))

model, opt_state, step, rng_key = eqx.tree_deserialise_leaves(
    f"checkpoints/step_{step}.eqx", skeleton
)
```

---

## 6. Alternatives

### 6.1 Equinox Built-in vs Orbax

| Feature | Equinox (`eqx.tree_serialise_leaves`) | Orbax (`ocp.CheckpointManager`) |
|---|---|---|
| **Simplicity** | Very simple, 1 function call | More setup required |
| **Format** | Binary file with numpy arrays | TensorStore-backed directories |
| **Async save** | No | Yes (default) |
| **Multi-host/device** | No (single process) | Yes (distributed arrays) |
| **Periodic save** | Manual | Built-in (`save_interval_steps`) |
| **Best-model tracking** | Manual | Built-in (`best_fn`) |
| **Top-K retention** | Manual | Built-in (`max_to_keep`) |
| **JSON metadata** | Manual (see hyperparameter pattern) | Built-in (`JsonSave`) |
| **Sharded arrays** | No | Yes (automatic repartitioning) |
| **Atomic writes** | No | Yes |
| **When to use** | Small models, simple training | Production, distributed, RL |

**Recommendation**: Use Equinox serialization for quick experiments and small models. Use Orbax for anything production-grade, distributed, or requiring async/periodic/best-model features. The custom EquinoxCheckpointHandler (section 3.5) gives you the best of both worlds.

### 6.2 safetensors for JAX

HuggingFace's `safetensors` library has native JAX/Flax support:

```python
# pip install safetensors

# --- Direct safetensors.flax API ---
from safetensors.flax import save_file, load_file
from jax import numpy as jnp

# Save: requires flat Dict[str, jax.Array]
tensors = {
    "embedding": jnp.zeros((512, 1024)),
    "attention": jnp.zeros((256, 256)),
}
save_file(tensors, "model.safetensors")

# Load
loaded = load_file("model.safetensors")  # Dict[str, jax.Array]

# --- In-memory ---
from safetensors.flax import save, load

byte_data = save(tensors)                # -> bytes
loaded = load(byte_data)                 # -> Dict[str, jax.Array]
```

**Limitation**: safetensors only supports flat `Dict[str, Array]`. Nested PyTrees need flattening:

```python
# Flax NNX -> safetensors
from flax import nnx
from safetensors.flax import save_file, load_file

model = MLP(4, 32, 2, rngs=nnx.Rngs(0))

# Flatten to dict
params = nnx.to_flat_state(nnx.state(model))
flat_dict = {".".join(map(str, k)): v for k, v in params}
save_file(flat_dict, "model.safetensors")

# Restore
loaded = load_file("model.safetensors")
flat_state = [
    (tuple(int(x) if x.isdigit() else x for x in k.split(".")), v)
    for k, v in loaded.items()
]
restored_state = nnx.from_flat_state(flat_state)
nnx.update(model, restored_state)
```

**safejax** (third-party): Wraps safetensors for JAX/Flax/Haiku/Objax:
```python
# pip install safejax
from safejax import serialize, deserialize

encoded = serialize(params)  # bytes
restored = deserialize(encoded, params)  # back to FrozenDict
```

**pytree2safetensors**: Another option for arbitrary PyTrees:
```python
# pip install pytree2safetensors
from pytree2safetensors import save_pytree, load_pytree, load_into_pytree

save_pytree(state, "state.safetensors")
loaded = load_pytree("state.safetensors")
# Or load into existing structure:
loaded = load_into_pytree("state.safetensors", skeleton)
```

| Feature | safetensors | safejax | pytree2safetensors |
|---|---|---|---|
| Nested PyTrees | No (flat dict only) | Yes (Flax FrozenDict) | Yes |
| Security | Excellent (no code exec) | Excellent | Excellent |
| Speed | Very fast (zero-copy) | Fast | Fast |
| JAX native | Yes (safetensors.flax) | Yes | Yes |
| Maturity | Production (HuggingFace) | Early stage | Early stage |

### 6.3 Simple Pickle / msgpack

**pickle** (not recommended):
```python
import pickle
import jax.numpy as jnp

# Save (INSECURE - arbitrary code execution on load)
with open("state.pkl", "wb") as f:
    pickle.dump(jax.tree.map(lambda x: np.asarray(x), state), f)

# Load
with open("state.pkl", "rb") as f:
    state = pickle.load(f)
```

**Flax msgpack serialization** (Linen-era, mostly deprecated):
```python
from flax.serialization import msgpack_serialize, msgpack_restore

# Only works with simple pytrees (dicts of numpy arrays)
# Does NOT work with modern jax.Array type
encoded = msgpack_serialize(params)  # bytes
restored = msgpack_restore(encoded)  # dict

# Higher-level: to_bytes / from_bytes
from flax.serialization import to_bytes, from_bytes
data = to_bytes(train_state)
restored = from_bytes(train_state, data)
```

**WARNING**: `flax.serialization.msgpack_serialize` is broken with modern `jax.Array` types (see flax issue #2696, jax issue #13540). Use Orbax or Equinox serialization instead.

**numpy save** (minimal):
```python
import jax.numpy as jnp
import numpy as np

# Save individual arrays
jnp.save("weights.npy", params['w'])

# Save multiple arrays
np.savez("checkpoint.npz", **jax.tree.map(np.asarray, flat_params))

# Load
loaded = np.load("checkpoint.npz")
```

### 6.4 Comparison Matrix

| Method | Security | Speed | Distributed | PyTree Support | Async | Maintained |
|---|---|---|---|---|---|---|
| **Orbax** | Good | Fast | Yes | Full | Yes | Google (active) |
| **Equinox** | Good | Fast | No | Full | No | Patrick Kidger (active) |
| **safetensors** | Excellent | Very fast | No | Flat only | No | HuggingFace (active) |
| **pickle** | UNSAFE | Medium | No | Full | No | stdlib |
| **msgpack (flax)** | Good | Fast | No | Limited | No | Deprecated |
| **numpy save** | Good | Fast | No | Manual | No | numpy |

---

## 7. Performance: Async & Large Model Support

### 7.1 Async Checkpointing Performance

Orbax's `AsyncCheckpointer` copies arrays from device to host synchronously (blocking), then writes to disk asynchronously in a background thread. Documented performance gains:

| Model Size | Save Time Reduction |
|---|---|
| 300M params | ~40% |
| 8B params | ~85% |
| 340B params | ~97% |

```python
# AsyncCheckpointer (standalone)
ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
ckptr.save(path, args=ocp.args.StandardSave(state))
# ... continue training while save happens in background ...
ckptr.wait_until_finished()  # MUST call before exit

# CheckpointManager (async by default)
options = ocp.CheckpointManagerOptions(
    enable_async_checkpointing=True,  # default is True
)
mngr = ocp.CheckpointManager(ckpt_dir, options=options)
mngr.save(step, args=ocp.args.StandardSave(state))
# Consecutive saves auto-wait for previous save to finish
mngr.wait_until_finished()  # call at end of training
```

### 7.2 Critical Async Rules

1. **Always call `wait_until_finished()`** before program exit
2. **Consecutive saves auto-block**: calling `save()` again waits for the previous save
3. **The checkpointer object must stay alive** during the entire async operation
4. **Device-to-host copy is always blocking** (to prevent corruption of train state)

### 7.3 Large Model / Multi-Host Support

Orbax handles distributed/sharded arrays natively:

```python
import jax
from jax.sharding import PartitionSpec, NamedSharding

# Arrays sharded across devices
mesh = jax.sharding.Mesh(jax.devices(), ('data',))
sharding = NamedSharding(mesh, PartitionSpec('data'))
sharded_state = jax.device_put(state, sharding)

# Same save API -- Orbax handles sharding automatically
mngr.save(step, args=ocp.args.StandardSave(sharded_state))

# Restore with potentially different sharding
# Orbax re-shards automatically based on the abstract target
restored = mngr.restore(step, args=ocp.args.StandardRestore(abstract_state))
```

### 7.4 Background Deletion

For large checkpoints, deletion can also be slow:

```python
options = ocp.CheckpointManagerOptions(
    enable_background_delete=True,  # delete old checkpoints in background
)
```

### 7.5 Disable Async (for debugging)

```python
options = ocp.CheckpointManagerOptions(
    enable_async_checkpointing=False,  # synchronous saves
)
```

---

## 8. Recommendations

### For a JAX RL Project Using Both Flax NNX and Equinox

**Primary recommendation: Use Orbax CheckpointManager** with framework-specific state extraction.

1. **For Flax NNX models**: Use `nnx.split()` / `nnx.merge()` + `ocp.args.StandardSave`
2. **For Equinox models**: Either:
   - Use `eqx.tree_serialise_leaves` for simplicity (if no async/distributed needed)
   - Write a custom `EquinoxCheckpointHandler` for Orbax integration (section 3.5)
3. **For mixed projects**: Use Orbax's `Composite` to save different components:
   ```python
   mngr.save(step, args=ocp.args.Composite(
       flax_state=ocp.args.StandardSave(flax_state),
       eqx_model=EquinoxSave(eqx_model),
       metadata=ocp.args.JsonSave({...}),
   ))
   ```

4. **RL-specific**:
   - Always checkpoint: params, optimizer state, step counter, RNG key
   - For PPO/on-policy: skip replay buffer
   - For SAC/off-policy: checkpoint buffer only if environment is expensive
   - Use `best_fn` with `mean_return` for best-model tracking
   - Keep `max_to_keep=5` for crash recovery + top models

5. **Performance**:
   - Keep async checkpointing enabled (default)
   - Use `save_interval_steps` to avoid excessive I/O
   - For very large models, ensure `wait_until_finished()` is called

---

## Sources

- Orbax documentation: https://orbax.readthedocs.io/
- Orbax GitHub: https://github.com/google/orbax
- Orbax v1 RFC: https://github.com/google/orbax/issues/1624
- Flax NNX checkpointing: https://flax.readthedocs.io/en/latest/guides/checkpointing.html
- Equinox serialization: https://docs.kidger.site/equinox/examples/serialisation/
- Equinox serialization API: https://docs.kidger.site/equinox/api/serialisation/
- safetensors Flax API: https://huggingface.co/docs/safetensors/en/api/flax
- safejax: https://github.com/alvarobartt/safejax
- pytree2safetensors: https://pypi.org/project/pytree2safetensors/
- Orbax best+latest discussion: https://github.com/google/orbax/issues/526
- Equinox+Orbax handler issue: https://github.com/google/orbax/issues/741
- Flashbax replay buffers: https://github.com/instadeepai/flashbax
- JSAC (JAX RL with checkpointing): https://github.com/fahimfss/JSAC
- Flax serialization API: https://flax-linen.readthedocs.io/en/latest/api_reference/flax.serialization.html
