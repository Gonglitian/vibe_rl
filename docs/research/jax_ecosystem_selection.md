# JAX Ecosystem Selection Report

> P1 Research: Neural Network Library, Optimizer, Testing, Serialization

---

## Executive Summary

| Category | Recommendation | Rationale |
|----------|---------------|-----------|
| NN Library | **Equinox** | More Pythonic, faster for small models (RL-typical), stable API, pure PyTree design |
| Optimizer | **Optax** (standard) | De facto standard, no alternatives needed |
| Testing/Types | **Chex** + `NamedTuple` | Chex for assertions/testing, NamedTuple for immutable containers |
| Data Containers | **NamedTuple** (immutable) / **chex.dataclass** (mutable) / **flax.struct.PyTreeNode** (mixed) | Use the right tool per use case |
| Serialization | **Equinox built-in** + **Orbax** for advanced | `eqx.tree_serialise_leaves` for simplicity, Orbax CheckpointManager for periodic/best-model |

---

## 1. Neural Network Library: Equinox

### Decision: Equinox over Flax NNX

**Key reasons:**

1. **Performance for RL workloads**: Equinox shows ~3x faster forward+grad for small models (benchmarked in Flax #4045). RL typically uses small MLPs with high-frequency forward passes -- this overhead matters.

2. **Pure functional design**: Everything is a PyTree. No special wrappers needed for `jax.jit`, `jax.vmap`, `jax.grad`. This composes naturally with vectorized environments (PureJaxRL-style).

3. **API stability**: Equinox's design has been consistent since ~v0.11 (2023). Flax NNX is still evolving with frequent breaking changes (promoted from experimental in Sep 2024).

4. **Minimal abstraction**: Equinox adds almost nothing on top of JAX -- `eqx.Module` is just a frozen dataclass registered as a PyTree. This means less magic, easier debugging, and better composability.

5. **Ecosystem quality**: Patrick Kidger's ecosystem (Diffrax, Lineax, Optimistix) demonstrates high code quality and consistent design philosophy.

**Trade-offs accepted:**

- Smaller community than Flax (~2.8k vs ~7.1k stars)
- Single primary maintainer (risk factor, though codebase is clean)
- Most existing JAX RL code uses Flax Linen (migration examples are scarce)
- No Google backing (but also no Google-style API churn)

### Code Pattern: Equinox MLP

```python
import equinox as eqx
import jax

class MLP(eqx.Module):
    layers: list

    def __init__(self, dims: list[int], *, key):
        keys = jax.random.split(key, len(dims) - 1)
        self.layers = [eqx.nn.Linear(d_in, d_out, key=k)
                       for d_in, d_out, k in zip(dims[:-1], dims[1:], keys)]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x)
```

### Training Loop Pattern

```python
import optax

@eqx.filter_jit
def train_step(model, opt_state, batch):
    def loss_fn(model):
        # ... compute loss
        return loss, aux

    (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss, aux
```

---

## 2. Optimizer: Optax

### Decision: Optax (confirmed as standard choice)

Optax is the only serious option in the JAX ecosystem. No decision needed -- it's universally adopted.

### Standard Recipes for RL

**PPO (default recipe):**
```python
optimizer = optax.chain(
    optax.clip_by_global_norm(max_norm=0.5),
    optax.adam(learning_rate=2.5e-4, eps=1e-5),
)
# With linear LR decay:
schedule = optax.linear_schedule(init_value=2.5e-4, end_value=0.0,
                                  transition_steps=total_updates)
optimizer = optax.chain(
    optax.clip_by_global_norm(0.5),
    optax.adam(learning_rate=schedule, eps=1e-5),
)
```

**SAC (separate actor/critic):**
```python
actor_tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-4))
critic_tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-4))
alpha_tx = optax.adam(3e-4)  # temperature parameter
```

**Algorithm reference table:**

| Algorithm | Optimizer | LR | Clip | Schedule |
|-----------|-----------|-----|------|----------|
| PPO | Adam (eps=1e-5) | 2.5e-4 | global_norm=0.5 | Linear decay |
| SAC | Adam x3 | 3e-4 | global_norm=1.0 | Constant |
| TD3 | Adam x2 | 3e-4 | None | Constant |
| DQN | Adam (eps=1.5e-4) | 6.25e-5 | global_norm=10 | Constant |

### Key Patterns

- **Clip before optimizer** in `optax.chain()`
- **Schedules as `learning_rate` arg**: All schedules are callables that plug directly into optimizer constructors
- **`optax.inject_hyperparams`**: For runtime hyperparameter access/logging
- **`optax.MultiSteps`**: For gradient accumulation (wraps any optimizer)
- **`optax.multi_transform`**: For per-parameter-group optimizers

---

## 3. Testing & Type Checking: Chex

### Decision: Use Chex for assertions and testing infrastructure

### Data Container Strategy (3-tier)

| Use Case | Container | Why |
|----------|-----------|-----|
| Immutable RL data (Transition, Trajectory, Experience) | `NamedTuple` | Zero boilerplate, auto-PyTree, immutable, positional construction |
| Mutable state (TrainState, all fields are arrays) | `chex.dataclass` | Auto-PyTree, mutable, Mapping interface |
| Mixed array + metadata (configs with callables/strings) | `flax.struct.PyTreeNode` | `pytree_node=False` for static fields |

```python
# Transitions: NamedTuple (immutable, all arrays)
from typing import NamedTuple
class Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    next_obs: jnp.ndarray
    done: jnp.ndarray

# Train state: chex.dataclass (mutable, all arrays)
@chex.dataclass
class TrainState:
    params: chex.ArrayTree
    opt_state: chex.ArrayTree
    step: chex.Array

# Config with callables: flax.struct.PyTreeNode
class AgentConfig(struct.PyTreeNode):
    hidden_dim: int = struct.field(pytree_node=False)
    gamma: float = struct.field(pytree_node=False)
```

### Essential Chex Assertions for RL

```python
# Shape validation with named dimensions
dims = chex.Dimensions(B=batch_size, T=seq_len, O=obs_dim, A=act_dim)
chex.assert_shape(observations, dims['BTO'])
chex.assert_shape(actions, dims['BTA'])

# NaN/Inf detection during training
chex.assert_tree_all_finite(grads)
chex.assert_tree_all_finite(loss)

# Gradient correctness
chex.assert_numerical_grads(loss_fn, (params,), order=1)

# Prevent accidental retracing (performance)
@chex.assert_max_traces(n=1)
def critic_forward(params, obs): ...

# Test under jit/no-jit/vmap automatically
class PolicyTest(chex.TestCase):
    @chex.all_variants
    def test_policy_output(self):
        result = self.variant(policy_fn)(params, obs)
        chex.assert_tree_all_finite(result)
```

### Debugging Utilities

- `chex.fake_jit()` -- disable JIT for step-through debugging
- `chex.fake_pmap()` -- replace pmap with vmap for single-device debugging
- `chex.disable_asserts()` -- turn off all assertions in production
- `@chex.chexify` -- enable value assertions inside JIT

---

## 4. Serialization: Equinox Built-in + Orbax

### Decision: Two-tier strategy

**Tier 1 -- Simple (default):** `eqx.tree_serialise_leaves` / `eqx.tree_deserialise_leaves`
- Single function call, works with any PyTree
- Sufficient for single-machine RL training
- No async, no periodic, no best-model tracking

**Tier 2 -- Advanced (when needed):** Orbax `CheckpointManager`
- Periodic checkpointing with `save_interval_steps`
- Best-model tracking with `best_fn` + `best_mode`
- Top-K retention with `max_to_keep`
- Async checkpointing (dramatic speedups for large models)
- Requires custom `EquinoxCheckpointHandler` wrapper

### Simple Checkpointing (Equinox built-in)

```python
import equinox as eqx

# Save
eqx.tree_serialise_leaves("checkpoint.eqx", (model, opt_state, step))

# Load (need template with correct structure)
template = (model_template, opt_state_template, step_template)
model, opt_state, step = eqx.tree_deserialise_leaves("checkpoint.eqx", template)
```

### What to Checkpoint in RL

Must checkpoint:
- Model parameters (actor, critic)
- Target network parameters (if applicable)
- Optimizer state(s)
- Training step counter
- RNG key state
- Running normalization statistics (obs/reward normalizer)

Optional:
- Replay buffer (skip for on-policy like PPO; consider for off-policy like SAC)
- Environment state (for deterministic resume of vectorized envs)

### Orbax Integration (when graduating to production)

```python
import orbax.checkpoint as ocp

# Custom handler for Equinox models
class EqxCheckpointHandler(ocp.CheckpointHandler):
    def save(self, directory, item, *args, **kwargs):
        path = str(directory / "model.eqx")
        eqx.tree_serialise_leaves(path, item)

    def restore(self, directory, item=None, *args, **kwargs):
        path = str(directory / "model.eqx")  # str() is critical!
        return eqx.tree_deserialise_leaves(path, item)

mgr = ocp.CheckpointManager(
    directory="/checkpoints/run_001",
    options=ocp.CheckpointManagerOptions(
        max_to_keep=5,
        save_interval_steps=1000,
        best_fn=lambda m: m['mean_return'],
        best_mode='max',
    ),
)
```

---

## Dependency Summary

```toml
[project]
dependencies = [
    "jax[cuda12]",      # or jax[cpu] for dev
    "equinox>=0.13",
    "optax>=0.2",
    "chex>=0.1.89",
    "orbax-checkpoint",  # optional, for advanced checkpointing
]
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Equinox maintainer burnout | Low-Medium | High | Codebase is clean, could be forked; Flax NNX as fallback |
| Equinox API breaking change | Low | Medium | Pin version; API has been stable 2+ years |
| Flax ecosystem tools assume Flax | Medium | Low | Optax/Chex/Orbax all work with any PyTree, not Flax-specific |
| Community examples mostly Flax | High | Low | Core patterns translate directly; we document our own patterns |
