"""Functional environment interface for pure-JAX RL.

Follows the Gymnax-style API where all functions are pure
(no hidden state mutation) and compatible with jit/vmap/lax.scan.

Core pattern::

    env = CartPole()
    params = env.default_params()
    key = jax.random.PRNGKey(0)

    obs, state = env.reset(key, params)
    obs, state, reward, done, info = env.step(key, state, action, params)

    # Vectorize across N parallel environments:
    batch_reset = jax.vmap(env.reset, in_axes=(0, None))
    batch_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import equinox as eqx
import jax

from vibe_rl.env.spaces import Box, Discrete, Image


class EnvState(eqx.Module):
    """Base class for environment states.

    All environment states must be registered JAX PyTrees (eqx.Module
    achieves this automatically). States must be immutable — step()
    returns a *new* state rather than mutating in place.
    """

    time: jax.Array  # current timestep within the episode


class EnvParams(eqx.Module):
    """Base class for environment parameters.

    Parameters are separated from state so they can be:
    - Shared across vmapped environments (in_axes=None)
    - Vmapped themselves for meta-learning / domain randomisation
    """


class Environment(ABC):
    """Abstract base for pure-JAX environments.

    Subclasses implement:
    - ``reset(key, params) -> (obs, state)``
    - ``step(key, state, action, params) -> (obs, state, reward, done, info)``
    - ``default_params() -> EnvParams``
    - ``observation_space(params) -> Space``
    - ``action_space(params) -> Space``
    """

    @abstractmethod
    def reset(
        self,
        key: jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState]:
        """Reset the environment and return ``(obs, state)``."""
        ...

    @abstractmethod
    def step(
        self,
        key: jax.Array,
        state: EnvState,
        action: jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[str, Any]]:
        """Advance one timestep.

        Returns:
            ``(obs, state, reward, done, info)`` where *done* merges
            terminated and truncated into a single flag.
        """
        ...

    @abstractmethod
    def default_params(self) -> EnvParams:
        """Return the default environment parameters."""
        ...

    @abstractmethod
    def observation_space(self, params: EnvParams) -> Box | Discrete | Image:
        """Return the observation space (may depend on params)."""
        ...

    @abstractmethod
    def action_space(self, params: EnvParams) -> Box | Discrete:
        """Return the action space (may depend on params)."""
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__
