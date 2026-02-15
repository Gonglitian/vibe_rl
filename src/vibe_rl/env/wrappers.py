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
from vibe_rl.env.spaces import Box, Discrete


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
