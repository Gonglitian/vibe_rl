"""Functional Agent interface for JAX-based RL.

Design philosophy:
  - All methods are pure functions: state in, state out.
  - No mutable self — agents are namespaces of static methods.
  - Every method is jit-compatible (or already jitted).
  - Checkpoint save/load is handled externally (Equinox/Orbax),
    not part of the agent interface.

Two complementary interfaces are provided:

1. ``Agent`` (Protocol) — structural typing contract that any agent must
   satisfy. Use this for generic code that operates on *any* agent.

2. Concrete agents (e.g. ``DQN``) — namespace classes with ``@staticmethod``
   or ``@classmethod`` implementations.  These are *not* instantiated;
   they group related pure functions under a readable name.

Example usage::

    from vibe_rl.algorithms.dqn import DQN, DQNConfig

    config = DQNConfig()
    state = DQN.init(rng, obs_shape=(4,), n_actions=2, config=config)
    action, state = DQN.act(state, obs, explore=True)
    state, metrics = DQN.update(state, batch)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import chex

from vibe_rl.types import AgentState, Metrics, Transition


@runtime_checkable
class Agent(Protocol):
    """Structural typing protocol for a pure-functional RL agent.

    Any class that implements ``init``, ``act``, and ``update`` as
    static/class methods with compatible signatures satisfies this
    protocol — no inheritance required.

    All methods must be pure functions suitable for ``jax.jit``.
    State is threaded explicitly via ``AgentState`` (or a subtype).
    """

    @staticmethod
    def init(
        rng: chex.PRNGKey,
        obs_shape: tuple[int, ...],
        n_actions: int,
        config: object,
    ) -> AgentState:
        """Initialize agent state (params, optimizer, step counter).

        Args:
            rng: PRNG key for parameter initialization.
            obs_shape: Shape of a single observation.
            n_actions: Size of the discrete action space.
            config: Algorithm-specific config dataclass.

        Returns:
            Initial ``AgentState`` (or algorithm-specific subtype).
        """
        ...

    @staticmethod
    def act(
        state: AgentState,
        obs: chex.Array,
        *,
        explore: bool = True,
    ) -> tuple[chex.Array, AgentState]:
        """Select an action given an observation.

        This is a pure function: the returned state carries an updated
        PRNG key (and any other mutable bookkeeping like epsilon step).

        Args:
            state: Current agent state.
            obs: Single observation array.
            explore: If True, use exploration (e.g. epsilon-greedy).
                     If False, act greedily.

        Returns:
            (action, new_state) tuple.
        """
        ...

    @staticmethod
    def update(
        state: AgentState,
        batch: Transition,
    ) -> tuple[AgentState, Metrics]:
        """Perform one gradient update on a batch of transitions.

        Args:
            state: Current agent state.
            batch: Batched transitions with leading batch dimension.

        Returns:
            (new_state, metrics) tuple.
        """
        ...
