"""Agent abstractions: functional protocol for JAX agents."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import jax


@runtime_checkable
class Agent(Protocol):
    """Functional agent protocol.

    An ``Agent`` is a *namespace* (typically a plain class with only
    ``@staticmethod`` methods) that exposes three operations:

    * ``init``   – create initial agent state from config
    * ``act``    – select an action given state + observation
    * ``update`` – perform one gradient step and return new state + metrics

    All methods are pure functions; mutable state is threaded through
    explicitly via an ``AgentState``-like NamedTuple.
    """

    @staticmethod
    def init(
        rng: jax.Array,
        obs_shape: tuple[int, ...],
        n_actions: int,
        config: Any,
    ) -> Any: ...

    @staticmethod
    def act(
        state: Any,
        obs: jax.Array,
        *,
        explore: bool = True,
        **kwargs: Any,
    ) -> tuple[jax.Array, Any]: ...

    @staticmethod
    def update(
        state: Any,
        batch: Any,
        **kwargs: Any,
    ) -> tuple[Any, Any]: ...
