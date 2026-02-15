"""Pure-JAX GridWorld environment.

A simple NxN grid where the agent navigates from top-left ``(0,0)``
to bottom-right ``(size-1, size-1)``.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from vibe_rl.env.base import Environment, EnvParams, EnvState
from vibe_rl.env.spaces import Box, Discrete


class GridWorldState(EnvState):
    row: jax.Array
    col: jax.Array


class GridWorldParams(EnvParams):
    size: int = eqx.field(static=True, default=5)
    max_steps: int = eqx.field(static=True, default=100)


# Directional deltas: up, right, down, left
_DR = jnp.array([-1, 0, 1, 0], dtype=jnp.int32)
_DC = jnp.array([0, 1, 0, -1], dtype=jnp.int32)


class GridWorld(Environment):
    """Simple grid navigation task in pure JAX.

    Observation: ``[row / (size-1), col / (size-1)]`` normalised to [0, 1].
    Actions: ``0``=up, ``1``=right, ``2``=down, ``3``=left.
    Reward: ``+1`` on reaching the goal, ``-0.01`` per step otherwise.
    """

    def default_params(self) -> GridWorldParams:
        return GridWorldParams()

    def reset(
        self,
        key: jax.Array,
        params: GridWorldParams,
    ) -> tuple[jax.Array, GridWorldState]:
        state = GridWorldState(
            row=jnp.int32(0),
            col=jnp.int32(0),
            time=jnp.int32(0),
        )
        return self._get_obs(state, params), state

    def step(
        self,
        key: jax.Array,
        state: GridWorldState,
        action: jax.Array,
        params: GridWorldParams,
    ) -> tuple[jax.Array, GridWorldState, jax.Array, jax.Array, dict[str, Any]]:
        dr = _DR[action]
        dc = _DC[action]
        row = jnp.clip(state.row + dr, 0, params.size - 1)
        col = jnp.clip(state.col + dc, 0, params.size - 1)
        time = state.time + 1

        new_state = GridWorldState(row=row, col=col, time=time)

        at_goal = (row == params.size - 1) & (col == params.size - 1)
        terminated = at_goal
        truncated = time >= params.max_steps
        done = terminated | truncated
        reward = jnp.where(at_goal, jnp.float32(1.0), jnp.float32(-0.01))

        return self._get_obs(new_state, params), new_state, reward, done, {"terminated": terminated, "truncated": truncated}

    def observation_space(self, params: GridWorldParams) -> Box:
        return Box(low=0.0, high=1.0, shape=(2,))

    def action_space(self, params: GridWorldParams) -> Discrete:
        return Discrete(n=4)

    @staticmethod
    def _get_obs(state: GridWorldState, params: GridWorldParams) -> jax.Array:
        denom = jnp.maximum(params.size - 1, 1)
        return jnp.array(
            [state.row / denom, state.col / denom],
            dtype=jnp.float32,
        )
