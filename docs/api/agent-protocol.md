# Agent Protocol

Functional agent interface for JAX-based RL.

```python
from vibe_rl.agent import Agent
from vibe_rl.types import AgentState, Metrics
```

## Design Philosophy

- All methods are **pure functions**: state in, state out.
- No mutable `self` — agents are namespaces of static methods.
- Every method is `jax.jit`-compatible (or already jitted).
- Checkpoint save/load is handled externally (Equinox/Orbax), not part of the agent interface.

Two complementary interfaces are provided:

1. **`Agent` (Protocol)** — structural typing contract that any agent must satisfy. Use this for generic code that operates on *any* agent.
2. **Concrete agents** (e.g. `DQN`, `PPO`, `SAC`) — namespace classes with `@staticmethod` implementations. These are *not* instantiated; they group related pure functions under a readable name.

---

## Agent Protocol

`vibe_rl.agent.base.Agent`

```python
@runtime_checkable
class Agent(Protocol):
    @staticmethod
    def init(
        rng: chex.PRNGKey,
        obs_shape: tuple[int, ...],
        n_actions: int,
        config: object,
    ) -> AgentState: ...

    @staticmethod
    def act(
        state: AgentState,
        obs: chex.Array,
        *,
        explore: bool = True,
    ) -> tuple[chex.Array, AgentState]: ...

    @staticmethod
    def update(
        state: AgentState,
        batch: Transition,
    ) -> tuple[AgentState, Metrics]: ...
```

Any class that implements `init`, `act`, and `update` as static/class methods with compatible signatures satisfies this protocol — no inheritance required.

### init

```python
@staticmethod
def init(
    rng: chex.PRNGKey,
    obs_shape: tuple[int, ...],
    n_actions: int,
    config: object,
) -> AgentState
```

Initialize agent state (params, optimizer, step counter).

| Parameter | Type | Description |
|---|---|---|
| `rng` | `PRNGKey` | PRNG key for parameter initialization. |
| `obs_shape` | `tuple[int, ...]` | Shape of a single observation. |
| `n_actions` | `int` | Size of the discrete action space (or action dimensionality for continuous). |
| `config` | `object` | Algorithm-specific config dataclass (`PPOConfig`, `DQNConfig`, `SACConfig`). |

**Returns:** Initial `AgentState` (or algorithm-specific subtype).

### act

```python
@staticmethod
def act(
    state: AgentState,
    obs: chex.Array,
    *,
    explore: bool = True,
) -> tuple[chex.Array, AgentState]
```

Select an action given an observation. Pure function — the returned state carries an updated PRNG key.

| Parameter | Type | Description |
|---|---|---|
| `state` | `AgentState` | Current agent state. |
| `obs` | `Array` | Single observation array, shape `(*obs_shape,)`. |
| `explore` | `bool` | If `True`, use exploration (e.g. epsilon-greedy, stochastic sampling). If `False`, act greedily / deterministically. |

**Returns:** `(action, new_state)` tuple.

### update

```python
@staticmethod
def update(
    state: AgentState,
    batch: Transition,
) -> tuple[AgentState, Metrics]
```

Perform one gradient update on a batch of transitions.

| Parameter | Type | Description |
|---|---|---|
| `state` | `AgentState` | Current agent state. |
| `batch` | `Transition` | Batched transitions with leading batch dimension. |

**Returns:** `(new_state, metrics)` tuple.

---

## AgentState

`vibe_rl.types.AgentState`

Minimal agent state shared across all algorithms.

```python
class AgentState(NamedTuple):
    params: Params          # Network parameter pytree (Equinox model or nested dict)
    opt_state: OptState     # Optax optimizer state
    step: chex.Array        # Scalar training step counter
    rng: chex.PRNGKey       # PRNG key for stochastic operations
```

### AgentState.initial

```python
@staticmethod
def initial(
    params: Params,
    opt_state: OptState,
    rng: chex.PRNGKey,
) -> AgentState
```

Create an `AgentState` at step 0.

---

## Metrics

`vibe_rl.types.Metrics`

Base training metrics returned from an update step. Algorithms define their own `NamedTuple` subtypes with additional fields.

```python
class Metrics(NamedTuple):
    loss: chex.Array
```

---

## Concrete Agent Implementations

### PPO

`vibe_rl.algorithms.ppo.PPO`

Namespace for PPO pure functions. Supports discrete (Categorical) action spaces.

```python
from vibe_rl.algorithms.ppo import PPO, PPOConfig
```

#### PPO.init

```python
@staticmethod
def init(
    rng: chex.PRNGKey,
    obs_shape: tuple[int, ...],
    n_actions: int,
    config: PPOConfig,
) -> PPOState
```

Create initial PPO state. When `config.shared_backbone=True`, uses `ActorCriticShared`; otherwise creates separate `ActorCategorical` + `Critic`.

#### PPO.act

```python
@staticmethod
def act(
    state: PPOState,
    obs: chex.Array,
    *,
    config: PPOConfig,
) -> tuple[chex.Array, chex.Array, chex.Array, PPOState]
```

Select action, returning `(action, log_prob, value, new_state)`.

| Parameter | Type | Description |
|---|---|---|
| `state` | `PPOState` | Current PPO state. |
| `obs` | `Array` | Single observation, shape `(*obs_shape,)`. |
| `config` | `PPOConfig` | PPO hyperparameters (static). |

**Returns:** `(action, log_prob, value, new_state)` tuple.

#### PPO.act_batch

```python
@staticmethod
def act_batch(
    state: PPOState,
    obs: chex.Array,
    *,
    config: PPOConfig,
) -> tuple[chex.Array, chex.Array, chex.Array, PPOState]
```

Vectorized action selection for a batch of observations.

| Parameter | Type | Description |
|---|---|---|
| `obs` | `Array` | Batch of observations, shape `(N, *obs_shape)`. |

**Returns:** `(actions, log_probs, values, new_state)` each with leading dim `N`.

#### PPO.evaluate_actions

```python
@staticmethod
def evaluate_actions(
    params: chex.ArrayTree,
    obs: chex.Array,
    actions: chex.Array,
    *,
    config: PPOConfig,
) -> tuple[chex.Array, chex.Array, chex.Array]
```

Re-evaluate log_prob, value, entropy for a batch of `(obs, action)`. Used inside the update step to compute losses with current params.

**Returns:** `(log_probs, values, entropy)` each shape `(B,)`.

#### PPO.get_value / PPO.get_value_batch

```python
@staticmethod
def get_value(state: PPOState, obs: chex.Array, *, config: PPOConfig) -> chex.Array
@staticmethod
def get_value_batch(state: PPOState, obs: chex.Array, *, config: PPOConfig) -> chex.Array
```

Compute value estimate(s) for a single observation or a batch of observations.

#### PPO.update

```python
@staticmethod
def update(
    state: PPOState,
    trajectories: PPOTransition,
    last_value: chex.Array,
    *,
    config: PPOConfig,
) -> tuple[PPOState, PPOMetrics]
```

PPO update: compute GAE, then run multiple epochs of mini-batch SGD.

| Parameter | Type | Description |
|---|---|---|
| `state` | `PPOState` | Current PPO state. |
| `trajectories` | `PPOTransition` | Collected rollout data, each field shape `(T, ...)` or `(T, N, ...)` for vectorized envs. |
| `last_value` | `Array` | Bootstrap value for the final observation. |
| `config` | `PPOConfig` | PPO hyperparameters (static). |

**Returns:** `(new_state, PPOMetrics)`.

#### PPO.collect_rollout / PPO.collect_rollout_batch

```python
@staticmethod
def collect_rollout(state, env_obs, env_state, env_step_fn, env_params, *, config)
    -> tuple[PPOState, PPOTransition, Array, ArrayTree, Array]

@staticmethod
def collect_rollout_batch(state, env_obs, env_states, env_step_fn, env_params, *, config)
    -> tuple[PPOState, PPOTransition, Array, ArrayTree, Array]
```

Collect a fixed-length rollout using `lax.scan`. `collect_rollout` is for a single env, `collect_rollout_batch` for N parallel environments (trajectories have shape `(T, N, ...)`).

**Returns:** `(new_agent_state, trajectories, final_obs, final_env_state, last_value)`

#### PPOConfig

```python
@dataclass(frozen=True)
class PPOConfig:
    hidden_sizes: tuple[int, ...] = (64, 64)
    lr: float = 2.5e-4
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    n_steps: int = 128
    n_minibatches: int = 4
    n_epochs: int = 4
    num_envs: int = 1
    shared_backbone: bool = False
```

| Field | Type | Default | Description |
|---|---|---|---|
| `hidden_sizes` | `tuple[int, ...]` | `(64, 64)` | MLP hidden layer widths. |
| `lr` | `float` | `2.5e-4` | Learning rate. |
| `max_grad_norm` | `float` | `0.5` | Global gradient clipping norm. |
| `gamma` | `float` | `0.99` | Discount factor. |
| `gae_lambda` | `float` | `0.95` | GAE lambda. |
| `clip_eps` | `float` | `0.2` | PPO clipping epsilon. |
| `vf_coef` | `float` | `0.5` | Value function loss coefficient. |
| `ent_coef` | `float` | `0.01` | Entropy bonus coefficient. |
| `n_steps` | `int` | `128` | Rollout length per update. |
| `n_minibatches` | `int` | `4` | Number of minibatches per epoch. |
| `n_epochs` | `int` | `4` | Optimization epochs per update. |
| `num_envs` | `int` | `1` | Number of vectorized environments. |
| `shared_backbone` | `bool` | `False` | Use shared actor-critic backbone. |

#### PPOState

```python
class PPOState(NamedTuple):
    params: Params          # ActorCriticParams or ActorCriticShared
    opt_state: OptState     # Optax optimizer state
    step: chex.Array        # Scalar step counter
    rng: chex.PRNGKey       # PRNG key
```

#### PPOMetrics

```python
class PPOMetrics(NamedTuple):
    total_loss: chex.Array
    actor_loss: chex.Array
    critic_loss: chex.Array
    entropy: chex.Array
    approx_kl: chex.Array
```

#### compute_gae

```python
def compute_gae(
    rewards: chex.Array,
    values: chex.Array,
    dones: chex.Array,
    last_value: chex.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[chex.Array, chex.Array]
```

Compute Generalized Advantage Estimation.

| Parameter | Type | Description |
|---|---|---|
| `rewards` | `Array` | Shape `(T,)` or `(T, N)` for vectorized envs. |
| `values` | `Array` | Shape `(T,)` or `(T, N)`. |
| `dones` | `Array` | Shape `(T,)` or `(T, N)`. |
| `last_value` | `Array` | Shape `()` or `(N,)` — bootstrap value. |
| `gamma` | `float` | Discount factor. |
| `gae_lambda` | `float` | GAE lambda. |

**Returns:** `(advantages, returns)` each with same shape as `rewards`.

---

### DQN

`vibe_rl.algorithms.dqn.DQN`

Namespace for DQN pure functions. Supports discrete action spaces with epsilon-greedy exploration.

```python
from vibe_rl.algorithms.dqn import DQN, DQNConfig
```

#### DQN.init

```python
@staticmethod
def init(
    rng: chex.PRNGKey,
    obs_shape: tuple[int, ...],
    n_actions: int,
    config: DQNConfig,
) -> DQNState
```

Create initial DQN state with online and target Q-networks.

#### DQN.act

```python
@staticmethod
def act(
    state: DQNState,
    obs: chex.Array,
    *,
    config: DQNConfig,
    explore: bool = True,
) -> tuple[chex.Array, DQNState]
```

Select action via epsilon-greedy policy. Epsilon decays linearly from `epsilon_start` to `epsilon_end` over `epsilon_decay_steps`.

**Returns:** `(action, new_state)` — action is a scalar int array.

#### DQN.update

```python
@staticmethod
def update(
    state: DQNState,
    batch: Transition,
    *,
    config: DQNConfig,
) -> tuple[DQNState, DQNMetrics]
```

One gradient step on a batch of transitions. Periodically copies online params to target network every `config.target_update_freq` steps.

#### DQNConfig

```python
@dataclass(frozen=True)
class DQNConfig:
    hidden_sizes: tuple[int, ...] = (128, 128)
    lr: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 64
    max_grad_norm: float = 10.0
    target_update_freq: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 50_000
```

| Field | Type | Default | Description |
|---|---|---|---|
| `hidden_sizes` | `tuple[int, ...]` | `(128, 128)` | Q-network hidden layer widths. |
| `lr` | `float` | `1e-3` | Learning rate. |
| `gamma` | `float` | `0.99` | Discount factor. |
| `batch_size` | `int` | `64` | Minibatch size for replay sampling. |
| `max_grad_norm` | `float` | `10.0` | Global gradient clipping norm. |
| `target_update_freq` | `int` | `1_000` | Hard-copy target network every N steps. |
| `epsilon_start` | `float` | `1.0` | Initial exploration epsilon. |
| `epsilon_end` | `float` | `0.01` | Final exploration epsilon. |
| `epsilon_decay_steps` | `int` | `50_000` | Steps over which epsilon decays linearly. |

#### DQNState

```python
class DQNState(NamedTuple):
    params: Params          # Online Q-network (Equinox model)
    target_params: Params   # Target Q-network
    opt_state: OptState     # Optax optimizer state
    step: chex.Array        # Scalar step counter
    rng: chex.PRNGKey       # PRNG key
```

#### DQNMetrics

```python
class DQNMetrics(NamedTuple):
    loss: chex.Array        # TD loss
    q_mean: chex.Array      # Mean Q-value of chosen actions
    epsilon: chex.Array     # Current epsilon
```

---

### SAC

`vibe_rl.algorithms.sac.SAC`

Namespace for SAC pure functions. Supports continuous action spaces with reparameterized Gaussian policy and automatic temperature tuning.

```python
from vibe_rl.algorithms.sac import SAC, SACConfig
```

#### SAC.init

```python
@staticmethod
def init(
    rng: chex.PRNGKey,
    obs_shape: tuple[int, ...],
    action_dim: int,
    config: SACConfig,
) -> SACState
```

Create initial SAC state with actor, twin Q-networks, target critic, and alpha.

#### SAC.act

```python
@staticmethod
def act(
    state: SACState,
    obs: chex.Array,
    *,
    config: SACConfig,
    explore: bool = True,
) -> tuple[chex.Array, SACState]
```

Select action from the policy. When `explore=True`, uses reparameterized Gaussian sampling with tanh squashing. When `explore=False`, uses the deterministic mean.

**Returns:** `(action, new_state)` — action shape `(action_dim,)`.

#### SAC.update

```python
@staticmethod
def update(
    state: SACState,
    batch: Transition,
    *,
    config: SACConfig,
) -> tuple[SACState, SACMetrics]
```

One gradient step performing three sequential updates:

1. **Critic update** — twin Q-networks via soft Bellman residual.
2. **Actor update** — maximum entropy objective.
3. **Alpha (temperature) update** — automatic tuning (when `autotune_alpha=True`).

Soft target network update (Polyak averaging) is applied after the critic update.

#### SACConfig

```python
@dataclass(frozen=True)
class SACConfig:
    hidden_sizes: tuple[int, ...] = (256, 256)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    batch_size: int = 256
    max_grad_norm: float = 10.0
    tau: float = 0.005
    init_alpha: float = 1.0
    autotune_alpha: bool = True
    target_entropy_scale: float = 1.0
    action_low: float = -1.0
    action_high: float = 1.0
    log_std_min: float = -20.0
    log_std_max: float = 2.0
```

| Field | Type | Default | Description |
|---|---|---|---|
| `hidden_sizes` | `tuple[int, ...]` | `(256, 256)` | Network hidden layer widths. |
| `actor_lr` | `float` | `3e-4` | Actor learning rate. |
| `critic_lr` | `float` | `3e-4` | Critic learning rate. |
| `alpha_lr` | `float` | `3e-4` | Alpha (temperature) learning rate. |
| `gamma` | `float` | `0.99` | Discount factor. |
| `batch_size` | `int` | `256` | Minibatch size for replay sampling. |
| `max_grad_norm` | `float` | `10.0` | Global gradient clipping norm. |
| `tau` | `float` | `0.005` | Polyak averaging coefficient for target network. |
| `init_alpha` | `float` | `1.0` | Initial entropy temperature. |
| `autotune_alpha` | `bool` | `True` | Automatically tune alpha. |
| `target_entropy_scale` | `float` | `1.0` | `target_entropy = -scale * action_dim`. |
| `action_low` | `float` | `-1.0` | Lower action bound (tanh squashing rescale). |
| `action_high` | `float` | `1.0` | Upper action bound (tanh squashing rescale). |
| `log_std_min` | `float` | `-20.0` | Minimum log standard deviation clamp. |
| `log_std_max` | `float` | `2.0` | Maximum log standard deviation clamp. |

#### SACState

```python
class SACState(NamedTuple):
    actor_params: Params            # GaussianActor (Equinox model)
    critic_params: Params           # TwinQNetwork (Equinox model)
    target_critic_params: Params    # Target TwinQNetwork
    actor_opt_state: OptState       # Actor optimizer state
    critic_opt_state: OptState      # Critic optimizer state
    log_alpha: chex.Array           # Log temperature parameter (scalar)
    alpha_opt_state: OptState       # Alpha optimizer state
    step: chex.Array                # Scalar step counter
    rng: chex.PRNGKey               # PRNG key
```

#### SACMetrics

```python
class SACMetrics(NamedTuple):
    actor_loss: chex.Array
    critic_loss: chex.Array
    alpha_loss: chex.Array
    alpha: chex.Array
    entropy: chex.Array
    q_mean: chex.Array
```

---

## Custom Agent Guide

To implement a custom agent that satisfies the `Agent` protocol:

```python
import chex
import jax
import jax.numpy as jnp

from vibe_rl.agent import Agent
from vibe_rl.types import AgentState, Metrics, Transition


class MyAgent:
    """Custom agent — namespace of static methods."""

    @staticmethod
    def init(
        rng: chex.PRNGKey,
        obs_shape: tuple[int, ...],
        n_actions: int,
        config: MyConfig,
    ) -> AgentState:
        # Initialize networks, optimizer, etc.
        ...
        return AgentState.initial(params, opt_state, rng)

    @staticmethod
    @jax.jit
    def act(
        state: AgentState,
        obs: chex.Array,
        *,
        explore: bool = True,
    ) -> tuple[chex.Array, AgentState]:
        rng, key = jax.random.split(state.rng)
        # Select action from your policy
        action = ...
        return action, state._replace(rng=rng)

    @staticmethod
    @jax.jit
    def update(
        state: AgentState,
        batch: Transition,
    ) -> tuple[AgentState, Metrics]:
        # Compute loss, apply gradients
        ...
        return new_state, Metrics(loss=loss)


# Verify protocol compliance:
assert isinstance(MyAgent, Agent)
```

Key rules:
1. All methods must be **static** or **class** methods.
2. All methods must be **pure functions** — state in, state out.
3. All methods must be **JIT-compatible** (no Python side effects in the hot path).
4. Use `state._replace(...)` for functional state updates on NamedTuples.
