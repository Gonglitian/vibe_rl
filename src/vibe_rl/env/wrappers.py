"""Functional environment wrappers for pure-JAX environments.

All wrappers follow the same ``Environment`` protocol so they compose
and remain jit/vmap compatible.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from vibe_rl.env.base import Environment, EnvParams, EnvState
from vibe_rl.env.spaces import Box, Discrete, Image


# ---------------------------------------------------------------------------
# Auto-reset wrapper
# ---------------------------------------------------------------------------

class AutoResetState(EnvState):
    """Wraps an inner state and tracks the underlying environment state."""

    inner: EnvState


class AutoResetWrapper(Environment):
    """Automatically resets the environment when ``done`` is True.

    On a done transition the returned ``obs`` and ``state`` already
    correspond to the fresh episode (the terminal observation is
    available via ``info["terminal_obs"]``).  This is the standard
    pattern for ``lax.scan``-based training loops.
    """

    env: Environment

    def __init__(self, env: Environment) -> None:
        self.env = env

    def default_params(self) -> EnvParams:
        return self.env.default_params()

    def reset(self, key: jax.Array, params: EnvParams) -> tuple[jax.Array, AutoResetState]:
        obs, inner_state = self.env.reset(key, params)
        return obs, AutoResetState(inner=inner_state, time=inner_state.time)

    def step(
        self,
        key: jax.Array,
        state: AutoResetState,
        action: jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, AutoResetState, jax.Array, jax.Array, dict[str, Any]]:
        key_step, key_reset = jax.random.split(key)

        obs, inner_state, reward, done, info = self.env.step(
            key_step, state.inner, action, params,
        )

        # Prepare a fresh state for when done==True
        reset_obs, reset_state = self.env.reset(key_reset, params)

        # Store terminal observation before overwriting
        info["terminal_obs"] = obs

        # Select between continuing and resetting
        new_inner = jax.tree.map(
            lambda r, c: jnp.where(done, r, c), reset_state, inner_state,
        )
        new_obs = jnp.where(done, reset_obs, obs)

        return new_obs, AutoResetState(inner=new_inner, time=new_inner.time), reward, done, info

    def observation_space(self, params: EnvParams) -> Box | Discrete:
        return self.env.observation_space(params)

    def action_space(self, params: EnvParams) -> Box | Discrete:
        return self.env.action_space(params)


# ---------------------------------------------------------------------------
# Reward scaling
# ---------------------------------------------------------------------------

class RewardScaleWrapper(Environment):
    """Multiplies rewards by a constant factor."""

    env: Environment
    scale: float

    def __init__(self, env: Environment, scale: float) -> None:
        self.env = env
        self.scale = scale

    def default_params(self) -> EnvParams:
        return self.env.default_params()

    def reset(self, key: jax.Array, params: EnvParams) -> tuple[jax.Array, EnvState]:
        return self.env.reset(key, params)

    def step(
        self,
        key: jax.Array,
        state: EnvState,
        action: jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[str, Any]]:
        obs, state, reward, done, info = self.env.step(key, state, action, params)
        return obs, state, reward * self.scale, done, info

    def observation_space(self, params: EnvParams) -> Box | Discrete:
        return self.env.observation_space(params)

    def action_space(self, params: EnvParams) -> Box | Discrete:
        return self.env.action_space(params)


# ---------------------------------------------------------------------------
# Observation normalization (running statistics)
# ---------------------------------------------------------------------------

class NormState(EnvState):
    """Tracks running mean/var alongside the inner env state."""

    inner: EnvState
    mean: jax.Array
    var: jax.Array
    count: jax.Array


class ObsNormWrapper(Environment):
    """Normalises observations using Welford's online algorithm.

    The running statistics are part of the state PyTree and thus
    automatically handled by ``jax.vmap`` / ``lax.scan``.
    """

    env: Environment
    epsilon: float

    def __init__(self, env: Environment, epsilon: float = 1e-8) -> None:
        self.env = env
        self.epsilon = epsilon

    def default_params(self) -> EnvParams:
        return self.env.default_params()

    def reset(self, key: jax.Array, params: EnvParams) -> tuple[jax.Array, NormState]:
        obs, inner = self.env.reset(key, params)
        state = NormState(
            inner=inner,
            time=inner.time,
            mean=jnp.zeros_like(obs),
            var=jnp.ones_like(obs),
            count=jnp.float32(1e-4),
        )
        normed = (obs - state.mean) / jnp.sqrt(state.var + self.epsilon)
        return normed, state

    def step(
        self,
        key: jax.Array,
        state: NormState,
        action: jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, NormState, jax.Array, jax.Array, dict[str, Any]]:
        obs, inner, reward, done, info = self.env.step(key, state.inner, action, params)

        # Welford update
        count = state.count + 1.0
        delta = obs - state.mean
        mean = state.mean + delta / count
        delta2 = obs - mean
        var = state.var * (state.count / count) + delta * delta2 / count

        new_state = NormState(inner=inner, time=inner.time, mean=mean, var=var, count=count)
        normed = (obs - mean) / jnp.sqrt(var + self.epsilon)
        return normed, new_state, reward, done, info

    def observation_space(self, params: EnvParams) -> Box | Discrete:
        return self.env.observation_space(params)

    def action_space(self, params: EnvParams) -> Box | Discrete:
        return self.env.action_space(params)


# ---------------------------------------------------------------------------
# Image resize with aspect-ratio padding
# ---------------------------------------------------------------------------

def _resize_with_pad(
    image: jax.Array,
    height: int,
    width: int,
    src_h: int,
    src_w: int,
) -> jax.Array:
    """Resize an (H, W, C) image to (height, width, C) with aspect-ratio padding.

    The image is scaled to fit within the target dimensions while
    preserving the aspect ratio, then centered with zero-padding.

    ``src_h`` and ``src_w`` are the *static* source dimensions so that
    all shapes are known at trace time and compatible with ``jax.jit``.
    """
    scale = min(height / src_h, width / src_w)
    new_h = int(round(src_h * scale))
    new_w = int(round(src_w * scale))

    # Resize using bilinear interpolation via jax.image.resize
    resized = jax.image.resize(
        image.astype(jnp.float32),
        (new_h, new_w, image.shape[2]),
        method="bilinear",
    )

    # Pad to target size, centering the resized image
    pad_top = (height - new_h) // 2
    pad_left = (width - new_w) // 2
    padded = jnp.pad(
        resized,
        (
            (pad_top, height - new_h - pad_top),
            (pad_left, width - new_w - pad_left),
            (0, 0),
        ),
    )
    return padded.astype(image.dtype)


class ImageResizeWrapper(Environment):
    """Resizes image observations to ``(height, width)`` with aspect-ratio padding.

    Input observations must be ``(H, W, C)`` arrays (e.g. from an ``Image`` space).
    The resized output preserves the original dtype.
    """

    env: Environment
    height: int = eqx.field(static=True)
    width: int = eqx.field(static=True)
    _src_h: int = eqx.field(static=True)
    _src_w: int = eqx.field(static=True)

    def __init__(self, env: Environment, height: int, width: int) -> None:
        self.env = env
        self.height = height
        self.width = width
        inner_space = env.observation_space(env.default_params())
        self._src_h = inner_space.shape[0]
        self._src_w = inner_space.shape[1]

    def default_params(self) -> EnvParams:
        return self.env.default_params()

    def _resize(self, obs: jax.Array) -> jax.Array:
        return _resize_with_pad(obs, self.height, self.width, self._src_h, self._src_w)

    def reset(self, key: jax.Array, params: EnvParams) -> tuple[jax.Array, EnvState]:
        obs, state = self.env.reset(key, params)
        return self._resize(obs), state

    def step(
        self,
        key: jax.Array,
        state: EnvState,
        action: jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[str, Any]]:
        obs, state, reward, done, info = self.env.step(key, state, action, params)
        return self._resize(obs), state, reward, done, info

    def observation_space(self, params: EnvParams) -> Image:
        inner_space = self.env.observation_space(params)
        channels = inner_space.shape[2] if len(inner_space.shape) == 3 else 1
        return Image(height=self.height, width=self.width, channels=channels)

    def action_space(self, params: EnvParams) -> Box | Discrete:
        return self.env.action_space(params)


# ---------------------------------------------------------------------------
# Frame stacking
# ---------------------------------------------------------------------------

class FrameStackState(EnvState):
    """Wraps inner state and stores the frame buffer."""

    inner: EnvState
    frames: jax.Array  # (n_frames, H, W, C) or (n_frames, ...) buffer


class FrameStackWrapper(Environment):
    """Stacks the last ``n_frames`` observations along a new leading axis.

    The output observation has shape ``(n_frames, *obs_shape)``.
    On reset the initial observation is repeated ``n_frames`` times.
    """

    env: Environment
    n_frames: int = eqx.field(static=True)

    def __init__(self, env: Environment, n_frames: int) -> None:
        self.env = env
        self.n_frames = n_frames

    def default_params(self) -> EnvParams:
        return self.env.default_params()

    def reset(self, key: jax.Array, params: EnvParams) -> tuple[jax.Array, FrameStackState]:
        obs, inner = self.env.reset(key, params)
        # Stack the first frame n_frames times
        frames = jnp.repeat(obs[None], self.n_frames, axis=0)
        state = FrameStackState(inner=inner, time=inner.time, frames=frames)
        return frames, state

    def step(
        self,
        key: jax.Array,
        state: FrameStackState,
        action: jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, FrameStackState, jax.Array, jax.Array, dict[str, Any]]:
        obs, inner, reward, done, info = self.env.step(key, state.inner, action, params)
        # Shift frames: drop oldest, append newest
        frames = jnp.concatenate([state.frames[1:], obs[None]], axis=0)
        new_state = FrameStackState(inner=inner, time=inner.time, frames=frames)
        return frames, new_state, reward, done, info

    def observation_space(self, params: EnvParams) -> Box:
        inner_space = self.env.observation_space(params)
        shape = (self.n_frames, *inner_space.shape)
        low = float(getattr(inner_space, 'low', jnp.array(0.0)).min()) if hasattr(inner_space, 'low') else 0.0
        high = float(getattr(inner_space, 'high', jnp.array(255.0)).max()) if hasattr(inner_space, 'high') else 255.0
        return Box(low=low, high=high, shape=shape)

    def action_space(self, params: EnvParams) -> Box | Discrete:
        return self.env.action_space(params)


# ---------------------------------------------------------------------------
# Grayscale conversion
# ---------------------------------------------------------------------------

class GrayscaleWrapper(Environment):
    """Converts RGB image observations to single-channel grayscale.

    Input must be ``(H, W, 3)`` RGB. Output is ``(H, W, 1)`` with the
    standard luminance weights ``0.2989 R + 0.5870 G + 0.1140 B``.
    """

    env: Environment

    def __init__(self, env: Environment) -> None:
        self.env = env

    def default_params(self) -> EnvParams:
        return self.env.default_params()

    @staticmethod
    def _to_gray(obs: jax.Array) -> jax.Array:
        weights = jnp.array([0.2989, 0.5870, 0.1140], dtype=jnp.float32)
        gray = jnp.sum(obs.astype(jnp.float32) * weights, axis=-1, keepdims=True)
        return gray.astype(obs.dtype)

    def reset(self, key: jax.Array, params: EnvParams) -> tuple[jax.Array, EnvState]:
        obs, state = self.env.reset(key, params)
        return self._to_gray(obs), state

    def step(
        self,
        key: jax.Array,
        state: EnvState,
        action: jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[str, Any]]:
        obs, state, reward, done, info = self.env.step(key, state, action, params)
        return self._to_gray(obs), state, reward, done, info

    def observation_space(self, params: EnvParams) -> Image:
        inner_space = self.env.observation_space(params)
        return Image(height=inner_space.shape[0], width=inner_space.shape[1], channels=1)

    def action_space(self, params: EnvParams) -> Box | Discrete:
        return self.env.action_space(params)


# ---------------------------------------------------------------------------
# Image normalization: uint8 [0,255] → float32 [-1,1]
# ---------------------------------------------------------------------------

class ImageNormWrapper(Environment):
    """Normalises uint8 ``[0, 255]`` image observations to float32 ``[-1, 1]``.

    The mapping is ``x_norm = x / 127.5 - 1.0``.
    """

    env: Environment

    def __init__(self, env: Environment) -> None:
        self.env = env

    def default_params(self) -> EnvParams:
        return self.env.default_params()

    @staticmethod
    def _normalize(obs: jax.Array) -> jax.Array:
        return obs.astype(jnp.float32) / 127.5 - 1.0

    def reset(self, key: jax.Array, params: EnvParams) -> tuple[jax.Array, EnvState]:
        obs, state = self.env.reset(key, params)
        return self._normalize(obs), state

    def step(
        self,
        key: jax.Array,
        state: EnvState,
        action: jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[str, Any]]:
        obs, state, reward, done, info = self.env.step(key, state, action, params)
        return self._normalize(obs), state, reward, done, info

    def observation_space(self, params: EnvParams) -> Box:
        inner_space = self.env.observation_space(params)
        return Box(low=-1.0, high=1.0, shape=inner_space.shape)

    def action_space(self, params: EnvParams) -> Box | Discrete:
        return self.env.action_space(params)


# ---------------------------------------------------------------------------
# Gymnasium → JAX adapter (convenience, NOT performance-critical)
# ---------------------------------------------------------------------------

class GymnasiumWrapper(Environment):
    """Wraps a Gymnasium environment for use with vibe_rl's API.

    **Warning**: This wrapper is NOT jit/vmap compatible because
    Gymnasium environments are stateful Python objects. Use it only
    for quick experiments or environments that have no JAX-native
    equivalent. For performance-critical training, prefer pure-JAX
    environments.

    The Gymnasium env is stored on the wrapper instance (not in the
    state PyTree). ``reset`` and ``step`` delegate to the underlying
    Gymnasium env and convert numpy arrays to JAX arrays.
    """

    _gym_env: Any
    _obs_space: Box | Discrete
    _act_space: Box | Discrete

    def __init__(self, gym_env: Any) -> None:
        import gymnasium  # noqa: F401 — fail fast if not installed

        self._gym_env = gym_env

        obs_sp = gym_env.observation_space
        if hasattr(obs_sp, "low"):
            self._obs_space = Box(
                low=jnp.asarray(obs_sp.low, dtype=jnp.float32),
                high=jnp.asarray(obs_sp.high, dtype=jnp.float32),
            )
        else:
            self._obs_space = Discrete(n=int(obs_sp.n))

        act_sp = gym_env.action_space
        if hasattr(act_sp, "n"):
            self._act_space = Discrete(n=int(act_sp.n))
        else:
            self._act_space = Box(
                low=jnp.asarray(act_sp.low, dtype=jnp.float32),
                high=jnp.asarray(act_sp.high, dtype=jnp.float32),
            )

    def default_params(self) -> EnvParams:
        return EnvParams()

    def reset(self, key: jax.Array, params: EnvParams) -> tuple[jax.Array, EnvState]:
        seed = int(jax.random.randint(key, (), 0, 2**31))
        obs, _info = self._gym_env.reset(seed=seed)
        return jnp.asarray(obs, dtype=jnp.float32), EnvState(time=jnp.int32(0))

    def step(
        self,
        key: jax.Array,
        state: EnvState,
        action: jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[str, Any]]:
        act = int(action) if self._act_space.shape == () else action.tolist()
        obs, reward, terminated, truncated, info = self._gym_env.step(act)
        new_state = EnvState(time=state.time + 1)
        return (
            jnp.asarray(obs, dtype=jnp.float32),
            new_state,
            jnp.float32(reward),
            jnp.bool_(terminated or truncated),
            {"terminated": jnp.bool_(terminated), "truncated": jnp.bool_(truncated)},
        )

    def observation_space(self, params: EnvParams) -> Box | Discrete:
        return self._obs_space

    def action_space(self, params: EnvParams) -> Box | Discrete:
        return self._act_space
