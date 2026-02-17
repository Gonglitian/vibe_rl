# Environment System Overview

vibe_rl provides a **pure-JAX** environment interface inspired by [Gymnax](https://github.com/RobertTLange/gymnax). Every function is pure — no hidden state mutation — and fully compatible with `jax.jit`, `jax.vmap`, and `jax.lax.scan`.

## Core Concepts

### Functional Interface

Unlike Gymnasium (which uses mutable Python objects), vibe_rl environments are **stateless callables**. State is passed in and returned explicitly:

```python
import jax
from vibe_rl.env import make

env, params = make("CartPole-v1")
key = jax.random.PRNGKey(0)

obs, state = env.reset(key, params)
obs, state, reward, done, info = env.step(key, state, action, params)
```

This functional design unlocks JAX transformations:

```python
# Vectorize across N parallel environments
batch_reset = jax.vmap(env.reset, in_axes=(0, None))
batch_step  = jax.vmap(env.step,  in_axes=(0, 0, 0, None))

# JIT-compile for speed
fast_step = jax.jit(env.step)
```

### The Three Core Classes

All three live in `vibe_rl.env.base`:

#### `EnvState`

Base class for environment states. All states are `eqx.Module` instances (immutable JAX PyTrees). Every state must contain a `time` field tracking the current timestep.

```python
class EnvState(eqx.Module):
    time: jax.Array  # current timestep within the episode
```

Concrete environments extend this with their own fields:

```python
class CartPoleState(EnvState):
    x: jax.Array
    x_dot: jax.Array
    theta: jax.Array
    theta_dot: jax.Array
```

#### `EnvParams`

Base class for environment parameters. Parameters are separated from state so they can be:

- **Shared** across vmapped environments (`in_axes=None`)
- **Vmapped** themselves for meta-learning / domain randomisation

```python
class CartPoleParams(EnvParams):
    gravity: float = eqx.field(static=True, default=9.8)
    masscart: float = eqx.field(static=True, default=1.0)
    max_steps: int = eqx.field(static=True, default=500)
```

#### `Environment`

Abstract base class defining the environment protocol. Subclasses must implement five methods:

| Method | Signature | Description |
|--------|-----------|-------------|
| `reset` | `(key, params) -> (obs, state)` | Reset and return initial observation + state |
| `step` | `(key, state, action, params) -> (obs, state, reward, done, info)` | Advance one timestep |
| `default_params` | `() -> EnvParams` | Return default parameter values |
| `observation_space` | `(params) -> Space` | Describe the observation space |
| `action_space` | `(params) -> Space` | Describe the action space |

The `step` method returns a 5-tuple:

- **obs** — the new observation (`jax.Array`)
- **state** — the new environment state (`EnvState`)
- **reward** — scalar reward (`jax.Array`, float32)
- **done** — episode termination flag (`jax.Array`, bool), merging `terminated` and `truncated`
- **info** — dictionary with extra data (e.g. `{"terminated": ..., "truncated": ...}`)

## Registry: `make()` and `register()`

The environment registry provides a string-based factory:

```python
from vibe_rl.env import make, register

# Create a built-in environment
env, params = make("CartPole-v1")

# Register and use a custom environment
register("MyEnv-v0", MyCustomEnv)
env, params = make("MyEnv-v0")
```

Available built-in names:

| Name | Class | Type |
|------|-------|------|
| `"CartPole-v1"` | `CartPole` | Classic control |
| `"GridWorld-v0"` | `GridWorld` | Discrete navigation |
| `"Pendulum-v1"` | `Pendulum` | Continuous control |
| `"PixelGridWorld-v0"` | `PixelGridWorld` | Visual observation |

## Spaces

Spaces describe the shape, dtype, and bounds of observations and actions. All spaces are immutable `eqx.Module` PyTrees with `sample(key)`, `contains(x)`, `shape`, and `dtype`.

### `Discrete(n)`

Integers `{0, 1, ..., n-1}`. Scalar shape `()`, dtype `int32`.

```python
from vibe_rl.env import Discrete

space = Discrete(n=4)
action = space.sample(key)        # -> int32 scalar in [0, 4)
space.contains(jnp.int32(2))      # -> True
```

### `Box(low, high, shape=None)`

Bounded continuous space. dtype `float32`.

```python
from vibe_rl.env import Box

# From explicit shape
space = Box(low=-1.0, high=1.0, shape=(3,))

# From array bounds
space = Box(low=jnp.array([0, -5]), high=jnp.array([1, 5]))
space.sample(key)  # -> float32 array of shape (2,)
```

Bounds are stored as static tuples internally, making `Box` hashable and usable as a JIT-static argument.

### `Image(height, width, channels)`

Image observation space with `uint8` pixels in `[0, 255]`. Shape is always 3-D `(H, W, C)`.

```python
from vibe_rl.env import Image

space = Image(height=84, width=84, channels=3)
space.sample(key)  # -> uint8 array (84, 84, 3)
```

### `MultiBinary(n)`

Binary vectors of length `n`. dtype `int32`, values are `0` or `1`.

```python
from vibe_rl.env import MultiBinary

space = MultiBinary(n=8)
space.sample(key)  # -> int32 array of shape (8,) with 0/1 values
```

## Differences from Gymnasium

| Aspect | Gymnasium | vibe_rl |
|--------|-----------|---------|
| State management | Hidden in `env` object | Explicit `state` passed in/out |
| Side effects | `env.step()` mutates internal state | Pure functions, no mutation |
| JIT compilation | Not supported | Full `jax.jit` support |
| Vectorisation | Needs `VectorEnv` wrapper | `jax.vmap` directly |
| Training loops | Python `for` loop | `jax.lax.scan` (compiled) |
| Random numbers | `env.np_random` | Explicit `key` splitting |
| Params | Set at init, then fixed | Separate `EnvParams` PyTree |

## Writing a Custom Environment

```python
import equinox as eqx
import jax
import jax.numpy as jnp

from vibe_rl.env.base import Environment, EnvParams, EnvState
from vibe_rl.env.spaces import Box, Discrete


class MyState(EnvState):
    position: jax.Array


class MyParams(EnvParams):
    goal: float = eqx.field(static=True, default=5.0)
    max_steps: int = eqx.field(static=True, default=100)


class MyEnv(Environment):
    def default_params(self) -> MyParams:
        return MyParams()

    def reset(self, key, params):
        pos = jax.random.uniform(key, shape=(), minval=-1.0, maxval=1.0)
        state = MyState(position=pos, time=jnp.int32(0))
        return jnp.array([pos]), state

    def step(self, key, state, action, params):
        new_pos = state.position + action.reshape(())
        time = state.time + 1
        new_state = MyState(position=new_pos, time=time)

        reward = -jnp.abs(new_pos - params.goal)
        done = time >= params.max_steps
        obs = jnp.array([new_pos])
        return obs, new_state, reward, done, {}

    def observation_space(self, params):
        return Box(low=-10.0, high=10.0, shape=(1,))

    def action_space(self, params):
        return Box(low=-1.0, high=1.0, shape=(1,))
```

Key rules for custom environments:

1. **State must be an `eqx.Module`** — this makes it a JAX PyTree automatically.
2. **Include a `time` field** — inherited from `EnvState`.
3. **All shapes must be static** — known at trace time for `jax.jit`.
4. **No Python side effects** — no `print`, no list mutation, no I/O inside `step`/`reset`.
5. **Use `eqx.field(static=True)`** for parameters that affect array shapes or control flow.
