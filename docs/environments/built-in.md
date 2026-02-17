# Built-in Environments

vibe_rl ships with four pure-JAX environments covering classic control, discrete navigation, and visual observation tasks. All are fully compatible with `jax.jit`, `jax.vmap`, and `jax.lax.scan`.

---

## CartPole-v1

Classic cart-pole balancing task (Barto, Sutton & Anderson, 1983). Physics match Gymnasium's `CartPole-v1` defaults.

**Goal**: Keep the pole upright by pushing the cart left or right.

| Property | Value |
|----------|-------|
| Observation | `[x, x_dot, theta, theta_dot]` — `Box(4,)` float32 |
| Actions | `Discrete(2)` — `0` push left, `1` push right |
| Reward | `+1.0` per timestep the pole stays upright |
| Termination | Pole angle > ±12° or cart position > ±2.4 |
| Truncation | After `max_steps` (default 500) |

**Parameters** (`CartPoleParams`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gravity` | 9.8 | Gravitational acceleration |
| `masscart` | 1.0 | Cart mass |
| `masspole` | 0.1 | Pole mass |
| `length` | 0.5 | Half-pole length |
| `force_mag` | 10.0 | Force magnitude per action |
| `tau` | 0.02 | Integration timestep (Euler) |
| `theta_threshold` | 0.2094 | Termination angle (12°, radians) |
| `x_threshold` | 2.4 | Termination cart position |
| `max_steps` | 500 | Maximum episode length |

**Example**:

```python
import jax
import jax.numpy as jnp
from vibe_rl.env import make

env, params = make("CartPole-v1")
key = jax.random.PRNGKey(0)

obs, state = env.reset(key, params)
print(obs.shape)  # (4,)

# Take a step: push right
obs, state, reward, done, info = env.step(key, state, jnp.int32(1), params)
print(reward)  # 1.0
```

---

## GridWorld-v0

Simple N×N grid navigation task. The agent starts at the top-left corner `(0, 0)` and must reach the bottom-right corner `(size-1, size-1)`.

**Goal**: Navigate to the goal cell in as few steps as possible.

| Property | Value |
|----------|-------|
| Observation | `[row/(size-1), col/(size-1)]` — `Box(2,)` float32, normalised to [0, 1] |
| Actions | `Discrete(4)` — `0` up, `1` right, `2` down, `3` left |
| Reward | `+1.0` on reaching the goal, `-0.01` per step otherwise |
| Termination | Agent reaches `(size-1, size-1)` |
| Truncation | After `max_steps` (default 100) |

**Parameters** (`GridWorldParams`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `size` | 5 | Grid dimensions (N×N) |
| `max_steps` | 100 | Maximum episode length |

**Example**:

```python
import jax
import jax.numpy as jnp
from vibe_rl.env import make

env, params = make("GridWorld-v0")
key = jax.random.PRNGKey(0)

obs, state = env.reset(key, params)
print(obs)  # [0., 0.]  — top-left corner

# Move right
obs, state, reward, done, info = env.step(key, state, jnp.int32(1), params)
print(obs)  # [0., 0.25]  — (0, 1) on a 5x5 grid
print(reward)  # -0.01
```

---

## Pendulum-v1

Classic inverted pendulum swingup task with continuous action space. Dynamics and reward function match Gymnasium's `Pendulum-v1`.

**Goal**: Swing the pendulum upright (theta=0) and keep it balanced.

| Property | Value |
|----------|-------|
| Observation | `[cos(theta), sin(theta), theta_dot]` — `Box(3,)` float32 |
| Actions | `Box(1,)` — continuous torque in `[-max_torque, max_torque]` |
| Reward | `-(theta^2 + 0.1 * theta_dot^2 + 0.001 * torque^2)` |
| Termination | None (no terminal state) |
| Truncation | After `max_steps` (default 200) |

**Parameters** (`PendulumParams`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_speed` | 8.0 | Angular velocity clamp |
| `max_torque` | 2.0 | Torque bounds |
| `dt` | 0.05 | Integration timestep (Euler) |
| `g` | 10.0 | Gravitational acceleration |
| `m` | 1.0 | Pendulum mass |
| `l` | 1.0 | Pendulum length |
| `max_steps` | 200 | Maximum episode length |

**Example**:

```python
import jax
import jax.numpy as jnp
from vibe_rl.env import make

env, params = make("Pendulum-v1")
key = jax.random.PRNGKey(42)

obs, state = env.reset(key, params)
print(obs.shape)  # (3,)

# Apply torque
action = jnp.array([0.5])
obs, state, reward, done, info = env.step(key, state, action, params)
print(reward)  # negative scalar (closer to 0 is better)
```

---

## PixelGridWorld-v0

Pixel-observation variant of GridWorld. Same navigation mechanics but produces RGB image observations instead of coordinate vectors.

**Goal**: Navigate to the goal cell using visual input.

**Rendering**:
- **Black** `(0, 0, 0)` — empty cell
- **White** `(255, 255, 255)` — the agent
- **Green** `(0, 255, 0)` — the goal

Each cell is `cell_px × cell_px` pixels, so the full image is `(size * cell_px, size * cell_px, 3)`.

| Property | Value |
|----------|-------|
| Observation | `Image(size*cell_px, size*cell_px, 3)` — uint8 RGB |
| Actions | `Discrete(4)` — `0` up, `1` right, `2` down, `3` left |
| Reward | `+1.0` on reaching the goal, `-0.01` per step otherwise |
| Termination | Agent reaches `(size-1, size-1)` |
| Truncation | After `max_steps` (default 100) |

**Parameters** (`PixelGridWorldParams`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `size` | 5 | Grid dimensions (N×N) |
| `max_steps` | 100 | Maximum episode length |
| `cell_px` | 8 | Pixels per grid cell |

With defaults, the observation shape is `(40, 40, 3)`.

**Example**:

```python
import jax
import jax.numpy as jnp
from vibe_rl.env import make

env, params = make("PixelGridWorld-v0")
key = jax.random.PRNGKey(0)

obs, state = env.reset(key, params)
print(obs.shape)   # (40, 40, 3)
print(obs.dtype)   # uint8

# This environment works well with image wrappers
from vibe_rl.env import GrayscaleWrapper, ImageNormWrapper

env = GrayscaleWrapper(env)         # (40, 40, 3) -> (40, 40, 1)
env = ImageNormWrapper(env)         # uint8 [0,255] -> float32 [-1,1]

obs, state = env.reset(key, params)
print(obs.shape)   # (40, 40, 1)
print(obs.dtype)   # float32
```

---

## Vectorisation with `jax.vmap`

All built-in environments support `jax.vmap` for parallel rollouts. Parameters can be shared (common) or vmapped (per-env):

```python
import jax
import jax.numpy as jnp
from vibe_rl.env import make

env, params = make("CartPole-v1")
n_envs = 64

# Split keys for each environment
keys = jax.random.split(jax.random.PRNGKey(0), n_envs)

# Vectorised reset — params shared across all envs
batch_reset = jax.vmap(env.reset, in_axes=(0, None))
obs, states = batch_reset(keys, params)
print(obs.shape)  # (64, 4)

# Vectorised step
batch_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
actions = jnp.ones(n_envs, dtype=jnp.int32)
obs, states, rewards, dones, infos = batch_step(keys, states, actions, params)
```

## Training Loop with `lax.scan`

Use `AutoResetWrapper` + `lax.scan` for compiled training loops (see [Wrappers](wrappers.md) for details):

```python
from vibe_rl.env import make, AutoResetWrapper

env, params = make("CartPole-v1")
env = AutoResetWrapper(env)

def env_step(carry, _):
    key, state = carry
    key, key_step = jax.random.split(key)
    action = env.action_space(params).sample(key_step)
    obs, state, reward, done, info = env.step(key_step, state, action, params)
    return (key, state), reward

key = jax.random.PRNGKey(0)
obs, state = env.reset(key, params)
(_, final_state), rewards = jax.lax.scan(env_step, (key, state), None, length=1000)
print(rewards.shape)  # (1000,)
```
