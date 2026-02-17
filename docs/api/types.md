# Types Reference

Core type definitions and data structures for **vibe_rl**.

All state containers are `NamedTuple`s for zero-overhead JAX pytree compatibility — immutable, auto-registered as pytrees, and composable with `jax.jit`, `jax.vmap`, and `jax.lax.scan`.

---

## Type Aliases

`vibe_rl.types`

```python
from vibe_rl.types import Action, Value, Reward, Done, LogProb, Params, OptState
```

| Alias | Resolves To | Description |
|---|---|---|
| `Action` | `chex.Array` | Action array. |
| `Value` | `chex.Array` | Value estimate. |
| `Reward` | `chex.Array` | Scalar reward. |
| `Done` | `chex.Array` | Episode termination flag. |
| `LogProb` | `chex.Array` | Log-probability of an action. |
| `Params` | `Any` | Network parameter pytree (Equinox model or nested dict). |
| `OptState` | `Any` | Optax optimizer state pytree. |

---

## Transition

`vibe_rl.dataprotocol.transition.Transition`

A single `(s, a, r, s', done)` experience tuple. All fields are JAX arrays. For batched usage, each field carries a leading batch dimension — the type is the same, just higher-rank.

```python
from vibe_rl.types import Transition
# or
from vibe_rl.dataprotocol import Transition
```

```python
class Transition(NamedTuple):
    obs: Array         # (*obs_shape,)  or (B, *obs_shape)
    action: Array      # ()  or (A,)    or (B,) or (B, A)
    reward: Array      # ()             or (B,)
    next_obs: Array    # (*obs_shape,)  or (B, *obs_shape)
    done: Array        # ()             or (B,)
```

| Field | Single Shape | Batched Shape | Description |
|---|---|---|---|
| `obs` | `(*obs_shape,)` | `(B, *obs_shape)` | Observation. |
| `action` | `()` or `(A,)` | `(B,)` or `(B, A)` | Action taken. |
| `reward` | `()` | `(B,)` | Scalar reward. |
| `next_obs` | `(*obs_shape,)` | `(B, *obs_shape)` | Next observation. |
| `done` | `()` | `(B,)` | Episode termination flag. |

---

## PPOTransition

`vibe_rl.dataprotocol.transition.PPOTransition`

Extended transition for on-policy algorithms (PPO). Includes value estimates and action log-probabilities needed for GAE computation and the PPO surrogate objective.

```python
from vibe_rl.dataprotocol import PPOTransition
```

```python
class PPOTransition(NamedTuple):
    obs: Array
    action: Array
    reward: Array
    next_obs: Array
    done: Array
    log_prob: Array    # Log-probability of the action under the policy
    value: Array       # Value estimate V(s)
```

---

## Batch / PPOBatch

```python
from vibe_rl.dataprotocol import Batch, PPOBatch
```

Convenience aliases — a "Batch" is simply a `Transition` (or `PPOTransition`) whose fields have a leading batch dimension. Using the same type avoids unnecessary conversion and keeps code `jit`/`vmap`-friendly.

```python
Batch = Transition         # Transition with leading (B, ...) dims
PPOBatch = PPOTransition   # PPOTransition with leading (B, ...) dims
```

---

## make_dummy_transition

```python
def make_dummy_transition(obs_shape: tuple[int, ...]) -> Transition
```

Create a zero-filled `Transition` (useful as a pytree template for `jax.eval_shape` or serialization).

---

## TrainState Variants

`vibe_rl.dataprotocol.train_state`

Immutable training state containers. Use `state._replace(step=state.step + 1)` for functional updates.

```python
from vibe_rl.dataprotocol import (
    TrainState, DQNTrainState, ActorCriticTrainState, SACTrainState,
    create_train_state, create_dqn_train_state,
)
```

### TrainState

Core training state shared by all algorithms.

```python
class TrainState(NamedTuple):
    params: Params          # Network parameters (Equinox model PyTree)
    opt_state: OptState     # Optax optimizer state
    step: Array             # Scalar int step counter
```

### DQNTrainState

Extends `TrainState` with a target network and epsilon schedule.

```python
class DQNTrainState(NamedTuple):
    params: Params
    target_params: Params   # Target network for stable TD targets
    opt_state: OptState
    step: Array
    epsilon: Array          # Current epsilon for exploration
```

### ActorCriticTrainState

Training state for actor-critic algorithms (PPO, A2C). Single optimizer over joint (actor, critic) params.

```python
class ActorCriticTrainState(NamedTuple):
    params: Params          # Typically an ActorCriticParams NamedTuple
    opt_state: OptState
    step: Array
```

### SACTrainState

Training state for SAC with separate actor/critic optimizers.

```python
class SACTrainState(NamedTuple):
    actor_params: Params
    critic_params: Params
    target_critic_params: Params     # Polyak-averaged target
    actor_opt_state: OptState
    critic_opt_state: OptState
    alpha_params: Params             # Temperature (log_alpha) parameter
    alpha_opt_state: OptState
    step: Array
```

---

## Factory Functions

### create_train_state

```python
def create_train_state(
    params: Params,
    tx: optax.GradientTransformation,
) -> TrainState
```

Initialize a basic `TrainState` from params and an optax optimizer. Step counter starts at 0.

| Parameter | Type | Description |
|---|---|---|
| `params` | `Params` | Network parameters (Equinox model). |
| `tx` | `GradientTransformation` | Optax optimizer. |

**Returns:** `TrainState`

### create_dqn_train_state

```python
def create_dqn_train_state(
    params: Params,
    tx: optax.GradientTransformation,
    *,
    epsilon_start: float = 1.0,
) -> DQNTrainState
```

Initialize a `DQNTrainState` with a copy of params as the target network.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `params` | `Params` | — | Q-network parameters. |
| `tx` | `GradientTransformation` | — | Optax optimizer. |
| `epsilon_start` | `float` | `1.0` | Initial epsilon value. |

**Returns:** `DQNTrainState`

---

## ReplayBuffer

`vibe_rl.dataprotocol.replay_buffer.ReplayBuffer`

Fixed-size circular buffer with uniform random sampling. Storage uses pre-allocated numpy arrays for O(1) insertion. Sampling returns a `Transition` of JAX arrays, ready for `jax.jit`.

```python
from vibe_rl.dataprotocol import ReplayBuffer
```

> **Note:** The buffer is **not** JIT-compatible — it lives outside the compiled training step (in Python/numpy).

### Constructor

```python
ReplayBuffer(
    capacity: int,
    obs_shape: tuple[int, ...],
    action_shape: tuple[int, ...] = (),
    action_dtype: np.dtype = np.int32,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `capacity` | `int` | — | Maximum number of transitions to store. |
| `obs_shape` | `tuple[int, ...]` | — | Shape of a single observation. |
| `action_shape` | `tuple[int, ...]` | `()` | Shape of a single action. `()` for scalar (discrete), `(A,)` for vector. |
| `action_dtype` | `np.dtype` | `np.int32` | Data type for actions. Use `np.float32` for continuous actions. |

### Methods

#### push

```python
def push(
    obs: np.ndarray,
    action: int | np.ndarray,
    reward: float,
    next_obs: np.ndarray,
    done: bool,
) -> None
```

Store a single transition.

#### push_transition

```python
def push_transition(t: Transition) -> None
```

Store a `Transition` (convenience wrapper that accepts JAX arrays, converting to numpy internally).

#### sample

```python
def sample(batch_size: int) -> Transition
```

Uniformly sample a batch and return as JAX arrays. Returns a `Transition` with each field having shape `(batch_size, ...)`.

#### \_\_len\_\_

```python
def __len__() -> int
```

Current number of stored transitions.

### Usage Example

```python
buffer = ReplayBuffer(capacity=100_000, obs_shape=(4,))

# In training loop:
buffer.push(obs, int(action), float(reward), next_obs, bool(done))

if len(buffer) >= warmup_steps:
    batch = buffer.sample(batch_size=64)  # Transition of jax arrays
    state, metrics = DQN.update(state, batch, config=config)
```
