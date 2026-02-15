# JAX RL Framework Research Report

## Executive Summary

This report analyzes five major JAX-based RL frameworks to inform the architectural decisions for `vibe_rl`. The key finding: **the JAX RL ecosystem has converged on a functional-closure pattern** with `jax.lax.scan`-based training loops, NamedTuple/dataclass state containers, and Flax `nn.Module` networks. We recommend adopting this consensus with targeted improvements from each framework.

---

## 1. Architecture Pattern Comparison

| Framework | Style | Algorithm Organization | Config |
|-----------|-------|----------------------|--------|
| **PureJaxRL** | Pure functional closures | `make_train(config) -> train(rng)` factory | Python dict |
| **Stoix** | Hybrid (Flax classes + closures) | `get_learner_fn` + `learner_setup` + `run_experiment` | Hydra YAML |
| **CleanRL-JAX** | Hybrid single-file | Separate `nn.Module`s + closure-captured functions | `@dataclass Args` + tyro |
| **RLax** | Pure stateless functions | Composable primitives (loss fns, returns, etc.) | No config (library) |
| **Mctx** | Pure functional with frozen dataclasses | `search()` with pluggable callables | functools.partial |

### Key Observations

**PureJaxRL** is the purest: `make_train(config)` returns a `train(rng)` function where config is closed over. The *entire* training loop (init, rollout, GAE, update, for all updates) compiles into one XLA program via `jax.jit(make_train(config))`. Multi-seed training is trivial: `jax.vmap(train)(rngs)`.

**Stoix** adds structure on top of the same core pattern. Its `get_learner_fn(env, apply_fns, update_fns, config)` factory is essentially PureJaxRL's pattern with explicit function-passing instead of closure capture. The three-function convention (`get_learner_fn`, `learner_setup`, `run_experiment`) is consistent across 25+ algorithms.

**CleanRL-JAX** provides both styles side-by-side: Python-loop version (traced by JIT) and `jax.lax.scan` version. This demonstrates that both approaches produce identical results, but the scan version is more amenable to full XLA fusion.

**RLax** operates at a lower level -- it provides composable *primitives* (PPO loss, GAE, TD returns, distributional RL) as pure functions. It does not manage training state or loops. Users compose these into their own training pipelines.

**Mctx** demonstrates how to handle inherently dynamic structures (search trees) in pure JAX: pre-allocated fixed-size arrays, sentinel values, `jax.lax.while_loop` for data-dependent control flow.

---

## 2. TrainState Organization

| Framework | State Container | What It Holds |
|-----------|----------------|---------------|
| **PureJaxRL** | Flax `TrainState` + plain tuple carry | `(train_state, env_state, obs, rng)` |
| **Stoix** | Custom NamedTuple hierarchy | `OnPolicyLearnerState(params, opt_states, key, env_state, timestep)` |
| **CleanRL-JAX** | Flax `TrainState` (sometimes subclassed) | `AgentParams` dataclass bundles all network params |
| **RLax** | None (stateless library) | Users manage their own state |
| **Mctx** | Frozen `chex.dataclass` | `Tree` with all search state as fixed-size arrays |

### Detailed Patterns

**PureJaxRL** uses Flax's `TrainState` directly for simple cases (PPO) and subclasses it for extras (DQN adds `target_network_params`). The full mutable state is a **plain tuple** called `runner_state`:

```python
# PPO
runner_state = (train_state, env_state, last_obs, rng)
# DQN
runner_state = (train_state, buffer_state, env_state, last_obs, rng)
```

**Stoix** defines a **typed NamedTuple hierarchy**:

```python
class OnPolicyLearnerState(NamedTuple):
    params: ActorCriticParams       # NamedTuple of actor + critic params
    opt_states: ActorCriticOptStates  # NamedTuple of optimizer states
    key: chex.PRNGKey
    env_state: WrapperState
    timestep: TimeStep
```

Each algorithm defines its own `Params` and `OptStates` NamedTuples. This gives type safety and named field access while remaining pytree-compatible.

**CleanRL-JAX** bundles multi-network params into a `flax.struct.dataclass`:

```python
@flax.struct.dataclass
class AgentParams:
    network_params: FrozenDict
    actor_params: FrozenDict
    critic_params: FrozenDict
```

For off-policy algorithms (TD3, DDPG), it uses **separate TrainStates per network**, each with its own `target_params`.

### Trajectory Storage

All frameworks use NamedTuples or dataclasses:

```python
# PureJaxRL
class Transition(NamedTuple):
    done: jnp.ndarray; action: jnp.ndarray; value: jnp.ndarray
    reward: jnp.ndarray; log_prob: jnp.ndarray; obs: jnp.ndarray

# Stoix (adds truncation handling)
class PPOTransition(NamedTuple):
    done: Done; truncated: Truncated; action: Action; value: Value
    reward: Array; bootstrap_value: Value; log_prob: Array; obs: Array

# CleanRL-JAX
@flax.struct.dataclass
class Storage:
    obs: jnp.array; actions: jnp.array; logprobs: jnp.array
    dones: jnp.array; values: jnp.array; rewards: jnp.array
    advantages: jnp.array; returns: jnp.array
```

---

## 3. Training Loop Implementation

### The Consensus: Nested `jax.lax.scan`

All three training frameworks (PureJaxRL, Stoix, CleanRL-JAX scan variant) use the same nested structure:

```
scan(num_updates):                    # Outer: full training run
    scan(num_steps):                  # Rollout: environment stepping
        vmap(env.step)(...)           # Vectorized environments
    calculate_gae(reverse scan)       # Advantage estimation
    scan(num_epochs):                 # PPO update epochs
        shuffle + split minibatches
        scan(num_minibatches):        # Minibatch gradient updates
            value_and_grad -> apply_gradients
```

### Framework-Specific Variations

**PureJaxRL**: The outermost scan covers ALL updates. Python loop only at the very top (jit + run). GAE uses `reverse=True, unroll=16` for performance.

**Stoix**: Python loop at the evaluation boundary only. Each `learn()` call runs `num_updates_per_eval` update steps in a compiled scan. This allows periodic evaluation and checkpointing:

```python
for eval_step in range(num_evaluation):
    learner_output = learn(learner_state)  # Compiled scan inside
    evaluator_output = evaluator(params, eval_keys)
    # Log, checkpoint
```

**CleanRL-JAX**: Provides both Python-loop (JIT-traced) and scan versions. The Python-loop version is more readable; the scan version is more performant.

### Environment Vectorization

All frameworks use `jax.vmap(env.step)` for parallel environments:

```python
obsv, env_state, reward, done, info = jax.vmap(
    env.step, in_axes=(0, 0, 0, None)
)(rng_step, env_state, action, env_params)
```

### Distributed Training (Stoix only)

Stoix adds multi-device support via `pmap` + `vmap` with double gradient synchronization:

```python
# Data shape: (n_devices, update_batch_size, num_envs, ...)
grads = jax.lax.pmean(grads, axis_name="batch")    # Within device
grads = jax.lax.pmean(grads, axis_name="device")   # Across devices
learn = jax.pmap(learner_fn, axis_name="device")
```

---

## 4. Design Patterns Worth Borrowing

### From PureJaxRL

| Pattern | Description | Value |
|---------|-------------|-------|
| **Factory-closure** | `make_train(config) -> train(rng)` | Full JIT compilation of entire pipeline |
| **vmap multi-seed** | `jax.vmap(train)(rngs)` | Trivial parallel experiments |
| **Triple-nested scan** | Updates > rollout > epochs/minibatches | Zero Python overhead in training |
| **LogWrapper** | Functional episode stat tracking | No Python side-effects in JIT |
| **`unroll=16` in GAE** | Scan unrolling hint for XLA | Better optimization of sequential ops |

### From Stoix

| Pattern | Description | Value |
|---------|-------------|-------|
| **NamedTuple state hierarchy** | Typed, composable state containers | Type safety + pytree compatibility |
| **Separate apply_fns/update_fns** | Pass functions explicitly, not in state | Minimal compiled state |
| **Hydra network composition** | Swap architectures via config YAML | Algorithm-agnostic network code |
| **Double pmean** | Sync gradients across both vmap and pmap axes | Clean multi-device scaling |
| **Three-function convention** | `get_learner_fn/learner_setup/run_experiment` | Consistent algorithm interface |

### From CleanRL-JAX

| Pattern | Description | Value |
|---------|-------------|-------|
| **`flax.struct.dataclass`** | Storage containers with `.replace()` | Clean functional updates |
| **`optax.inject_hyperparams`** | LR scheduling inside optimizer chain | No manual schedule management |
| **`optax.incremental_update`** | Target network polyak averaging | One-line soft updates |
| **Explicit PRNG threading** | Every fn takes and returns key | Clear randomness flow |

### From RLax

| Pattern | Description | Value |
|---------|-------------|-------|
| **Composable primitives** | Standalone loss/return functions | Reusable across algorithms |
| **Batched loss functions** | Explicit batch dims (faster than vmap) | Avoid vmap overhead on losses |
| **`chex` assertions** | `chex.assert_rank`, `chex.assert_type` | Catch shape errors early |
| **TxPair transforms** | Paired apply/apply_inv for value rescaling | Clean non-linear Bellman |

### From Mctx

| Pattern | Description | Value |
|---------|-------------|-------|
| **Frozen dataclass + `.replace()`** | Immutable state with functional updates | Enforced purity |
| **Pre-allocated fixed-size arrays** | Bounded "dynamic" structures | JIT-compatible variable-size data |
| **Sentinel values** | `-1` instead of `None` in traced code | No optional types in JIT |
| **`batch_update = vmap(update)`** | Tiny utility for batched scatter writes | Pervasive and useful |
| **Higher-order function composition** | `functools.partial` instead of class hierarchies | Clean strategy pattern |

---

## 5. Recommendation: Proposed Style for `vibe_rl`

### Core Architecture: PureJaxRL's functional closure pattern + Stoix's typed state

**Rationale**: PureJaxRL proved that the factory-closure pattern enables maximum XLA optimization. Stoix showed how to scale it to 25+ algorithms without losing structure. We should combine both.

### Specific Recommendations

#### 5.1 Training Pipeline: Factory-Closure with Typed State

```python
def make_train(config: TrainConfig) -> Callable[[PRNGKey], TrainOutput]:
    """Factory that returns a fully JIT-compilable train function."""
    env = make_env(config.env)

    def train(rng: PRNGKey) -> TrainOutput:
        # Init
        state = init_learner_state(rng, env, config)
        # Train (single compiled scan)
        state, metrics = jax.lax.scan(
            update_step, state, None, config.num_updates
        )
        return TrainOutput(state=state, metrics=metrics)

    return train

# Usage
train = jax.jit(make_train(config))
out = train(rng)
# Multi-seed
outs = jax.jit(jax.vmap(make_train(config)))(rngs)
```

#### 5.2 State: NamedTuples with Algorithm-Specific Composition

Borrow Stoix's typed hierarchy but keep it simple:

```python
class LearnerState(NamedTuple):
    params: Params              # Algorithm-specific NamedTuple
    opt_state: OptState         # Algorithm-specific NamedTuple
    env_state: EnvState
    obs: chex.Array
    rng: chex.PRNGKey

class ActorCriticParams(NamedTuple):
    actor: FrozenDict
    critic: FrozenDict

class Transition(NamedTuple):
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    log_prob: chex.Array
    value: chex.Array
```

**Why NamedTuples over Flax TrainState**: NamedTuples are zero-overhead pytrees, composable, and don't couple us to Flax's optimizer management. We can use Optax directly.

**Why NamedTuples over plain tuples**: Named fields prevent destructuring bugs across 25+ algorithms.

#### 5.3 Training Loop: `jax.lax.scan` Everywhere

Follow PureJaxRL's triple-nested scan. Break out of JIT only for evaluation/logging (Stoix pattern):

```python
for eval_step in range(num_evaluations):
    state, metrics = train_n_steps(state)    # Compiled scan
    eval_metrics = evaluate(state.params)    # Compiled eval
    log(metrics, eval_metrics)               # Python logging
```

#### 5.4 Loss Functions: RLax-Style Composable Primitives

Write losses as standalone pure functions with explicit batch dimensions:

```python
def ppo_clip_loss(
    log_prob_new: Array,    # [B]
    log_prob_old: Array,    # [B]
    advantages: Array,      # [B]
    epsilon: float,
) -> Array:                 # scalar
    ...

def gae(
    rewards: Array,         # [T, B]
    values: Array,          # [T, B]
    dones: Array,           # [T, B]
    gamma: float,
    lam: float,
) -> Array:                 # [T, B]
    ...
```

#### 5.5 Networks: Flax `nn.Module` (the Only Classes)

Following the universal consensus, neural networks are Flax modules:

```python
class Actor(nn.Module):
    action_dim: int
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256, kernel_init=orthogonal(jnp.sqrt(2)))(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(x)
        return distrax.Categorical(logits=x)
```

Network *definitions* are the only class-based code. Everything else is functional.

#### 5.6 Config: Python Dataclass (Not Hydra)

For a focused library, `@dataclass` with sensible defaults is simpler than Hydra:

```python
@dataclass
class PPOConfig:
    # Env
    env_name: str = "CartPole-v1"
    num_envs: int = 2048
    # Training
    total_timesteps: int = 1_000_000
    num_steps: int = 128           # rollout length
    num_epochs: int = 4
    num_minibatches: int = 4
    # PPO
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 2.5e-4
```

Can upgrade to Hydra later if needed for large-scale experimentation.

#### 5.7 Assertions: Use chex

Follow RLax's practice of validating shapes/types at trace time:

```python
def ppo_clip_loss(log_prob_new, log_prob_old, advantages, epsilon):
    chex.assert_rank([log_prob_new, log_prob_old, advantages], 1)
    chex.assert_equal_shape([log_prob_new, log_prob_old, advantages])
    ...
```

---

## Summary: Style Priority

1. **PureJaxRL** -- Core training pattern (factory-closure, nested scan, vmap seeds)
2. **Stoix** -- State organization (typed NamedTuples, consistent algorithm interface)
3. **RLax** -- Loss primitives (composable, batched, well-typed pure functions)
4. **Mctx** -- Advanced patterns (frozen dataclasses, while_loop, sentinel values)
5. **CleanRL-JAX** -- Readability reference (single-file clarity, explicit PRNG threading)

The north star: **every training pipeline should be expressible as `jax.jit(jax.vmap(make_train(config)))(rngs)` -- a single compiled program that trains N agents from N seeds with zero Python overhead.**
