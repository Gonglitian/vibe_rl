# Wrappers

Wrappers modify environment behaviour without changing the underlying environment. Every wrapper implements the `Environment` protocol, so wrappers compose freely and stay `jit`/`vmap` compatible.

```python
from vibe_rl.env import make, AutoResetWrapper, RewardScaleWrapper

env, params = make("CartPole-v1")
env = AutoResetWrapper(env)
env = RewardScaleWrapper(env, scale=0.1)

# Still works with jit/vmap — same API
obs, state = env.reset(key, params)
```

---

## Training Wrappers

### AutoResetWrapper

Automatically resets the environment when `done` is True. **Required for `jax.lax.scan` training loops** — without it, the scan would continue stepping a terminated environment.

On a done transition:
- The returned `obs` and `state` correspond to the **fresh episode**
- The terminal observation is available via `info["terminal_obs"]`
- Selection uses `jax.tree.map` + `jnp.where` for branchless conditional reset

**State**: `AutoResetState(inner, time)`

```python
from vibe_rl.env import make, AutoResetWrapper

env, params = make("CartPole-v1")
env = AutoResetWrapper(env)

key = jax.random.PRNGKey(0)
obs, state = env.reset(key, params)

# Training loop with lax.scan
def step_fn(carry, _):
    key, state = carry
    key, k = jax.random.split(key)
    action = env.action_space(params).sample(k)
    obs, state, reward, done, info = env.step(k, state, action, params)
    # When done=True: obs is from the new episode,
    # info["terminal_obs"] has the final observation
    return (key, state), (reward, done)

(_, final_state), (rewards, dones) = jax.lax.scan(
    step_fn, (key, state), None, length=10_000
)
```

### RewardScaleWrapper

Multiplies rewards by a constant factor. Useful for normalising reward magnitudes across environments.

No additional state — passes through the inner environment's state unchanged.

```python
from vibe_rl.env import make, RewardScaleWrapper

env, params = make("CartPole-v1")
env = RewardScaleWrapper(env, scale=0.1)

# Rewards are now 0.1 instead of 1.0 per step
obs, state, reward, done, info = env.step(key, state, action, params)
```

### ObsNormWrapper

Normalises observations using Welford's online algorithm for running mean/variance.

The running statistics are part of the state PyTree, so they're automatically handled by `jax.vmap` and `jax.lax.scan` — no separate tracking needed.

**State**: `NormState(inner, mean, var, count, time)`

**Normalisation formula**: `(obs - mean) / sqrt(var + epsilon)`

```python
from vibe_rl.env import make, ObsNormWrapper

env, params = make("CartPole-v1")
env = ObsNormWrapper(env, epsilon=1e-8)

# Observations are normalised to approximately zero mean, unit variance
obs, state = env.reset(key, params)
# state contains running statistics: state.mean, state.var, state.count
```

---

## Image Wrappers

These wrappers transform image observations and are designed for environments with `Image` observation spaces (like `PixelGridWorld-v0`).

### GrayscaleWrapper

Converts RGB image observations `(H, W, 3)` to single-channel grayscale `(H, W, 1)`.

Uses standard luminance weights: `0.2989 R + 0.5870 G + 0.1140 B`.

```python
from vibe_rl.env import make, GrayscaleWrapper

env, params = make("PixelGridWorld-v0")
env = GrayscaleWrapper(env)

obs, state = env.reset(key, params)
print(obs.shape)  # (40, 40, 1) — was (40, 40, 3)
```

### ImageNormWrapper

Normalises `uint8 [0, 255]` image observations to `float32 [-1, 1]`.

**Formula**: `x_norm = x / 127.5 - 1.0`

```python
from vibe_rl.env import make, ImageNormWrapper

env, params = make("PixelGridWorld-v0")
env = ImageNormWrapper(env)

obs, state = env.reset(key, params)
print(obs.dtype)          # float32
print(obs.min(), obs.max())  # approximately -1.0 to 1.0
```

### ImageResizeWrapper

Resizes image observations to a target `(height, width)` with aspect-ratio preserving padding.

- Scales the image to fit within the target dimensions using bilinear interpolation
- Centers the result with zero-padding if aspect ratios differ
- Source dimensions are captured at init time as static fields for JIT compatibility

```python
from vibe_rl.env import make, ImageResizeWrapper

env, params = make("PixelGridWorld-v0")
# Default obs is (40, 40, 3), resize to (84, 84)
env = ImageResizeWrapper(env, height=84, width=84)

obs, state = env.reset(key, params)
print(obs.shape)  # (84, 84, 3)
```

### FrameStackWrapper

Stacks the last `n_frames` observations along a new leading axis. The output has shape `(n_frames, *obs_shape)`.

On reset, the initial observation is repeated `n_frames` times.

**State**: `FrameStackState(inner, frames, time)` — `frames` is the `(n_frames, ...)` buffer.

```python
from vibe_rl.env import make, FrameStackWrapper

env, params = make("PixelGridWorld-v0")
env = FrameStackWrapper(env, n_frames=4)

obs, state = env.reset(key, params)
print(obs.shape)  # (4, 40, 40, 3)
```

---

## Gymnasium Adapter

### GymnasiumWrapper

Wraps a standard Gymnasium environment for use with vibe_rl's API. Converts numpy arrays to JAX arrays and translates the Gymnasium interface.

> **Warning**: This wrapper is **NOT** `jit`/`vmap` compatible because Gymnasium environments are stateful Python objects. Use it only for quick experiments or environments with no JAX-native equivalent.

```python
import gymnasium as gym
from vibe_rl.env import GymnasiumWrapper

gym_env = gym.make("MountainCar-v0")
env = GymnasiumWrapper(gym_env)
params = env.default_params()

key = jax.random.PRNGKey(0)
obs, state = env.reset(key, params)
obs, state, reward, done, info = env.step(key, state, jnp.int32(1), params)
```

The wrapper auto-detects observation and action space types:
- Continuous spaces (`gymnasium.spaces.Box`) map to `vibe_rl.env.Box`
- Discrete spaces (`gymnasium.spaces.Discrete`) map to `vibe_rl.env.Discrete`

---

## Composition Patterns

Wrappers compose via nesting — the outermost wrapper is applied last. Order matters.

### Standard Training Pipeline

```python
from vibe_rl.env import make, AutoResetWrapper, ObsNormWrapper, RewardScaleWrapper

env, params = make("CartPole-v1")
env = AutoResetWrapper(env)       # 1. Auto-reset on done
env = ObsNormWrapper(env)         # 2. Normalise observations
env = RewardScaleWrapper(env, scale=0.1)  # 3. Scale rewards
```

### Vision Pipeline

```python
from vibe_rl.env import (
    make,
    AutoResetWrapper,
    GrayscaleWrapper,
    ImageResizeWrapper,
    ImageNormWrapper,
    FrameStackWrapper,
)

env, params = make("PixelGridWorld-v0")
env = AutoResetWrapper(env)              # 1. Auto-reset
env = GrayscaleWrapper(env)              # 2. RGB -> grayscale (H,W,3) -> (H,W,1)
env = ImageResizeWrapper(env, 84, 84)    # 3. Resize to 84x84
env = ImageNormWrapper(env)              # 4. uint8 -> float32 [-1,1]
env = FrameStackWrapper(env, n_frames=4) # 5. Stack last 4 frames

obs, state = env.reset(key, params)
print(obs.shape)  # (4, 84, 84, 1)
print(obs.dtype)  # float32
```

### Combined with `vmap`

Wrappers work seamlessly with vectorisation:

```python
env, params = make("CartPole-v1")
env = AutoResetWrapper(env)
env = ObsNormWrapper(env)

n_envs = 32
keys = jax.random.split(jax.random.PRNGKey(0), n_envs)

batch_reset = jax.vmap(env.reset, in_axes=(0, None))
batch_step  = jax.vmap(env.step,  in_axes=(0, 0, 0, None))

obs, states = batch_reset(keys, params)
# Each env in the batch has its own running normalisation statistics
```

---

## Summary Table

| Wrapper | Purpose | Extra State | JIT/vmap |
|---------|---------|-------------|----------|
| `AutoResetWrapper` | Auto-reset on done | `AutoResetState` | Yes |
| `RewardScaleWrapper` | Multiply rewards | None | Yes |
| `ObsNormWrapper` | Running mean/var normalisation | `NormState` | Yes |
| `GrayscaleWrapper` | RGB to grayscale | None | Yes |
| `ImageNormWrapper` | uint8 to float32 [-1,1] | None | Yes |
| `ImageResizeWrapper` | Resize with padding | None | Yes |
| `FrameStackWrapper` | Stack N frames | `FrameStackState` | Yes |
| `GymnasiumWrapper` | Gymnasium adapter | None | **No** |
