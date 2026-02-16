"""Pixel-observation GridWorld environment.

Wraps the existing :class:`GridWorld` logic to produce image observations
instead of 2-D coordinate vectors.  The grid is rendered as an RGB image
where:

- **Black** ``(0, 0, 0)``: empty cell
- **White** ``(255, 255, 255)``: the agent
- **Green** ``(0, 255, 0)``: the goal

Each cell is ``cell_px × cell_px`` pixels, so the full image size is
``(size * cell_px, size * cell_px, 3)``.

This environment is fully compatible with ``jax.jit`` / ``jax.vmap``.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from vibe_rl.env.base import Environment, EnvParams, EnvState
from vibe_rl.env.grid_world import GridWorldState, _DC, _DR
from vibe_rl.env.spaces import Discrete, Image


class PixelGridWorldParams(EnvParams):
    size: int = eqx.field(static=True, default=5)
    max_steps: int = eqx.field(static=True, default=100)
    cell_px: int = eqx.field(static=True, default=8)


class PixelGridWorld(Environment):
    """GridWorld with pixel (image) observations.

    Observation: ``uint8`` RGB image of shape ``(size*cell_px, size*cell_px, 3)``.
    Actions: ``0``=up, ``1``=right, ``2``=down, ``3``=left.
    Reward: ``+1`` on reaching the goal, ``-0.01`` per step otherwise.
    """

    def default_params(self) -> PixelGridWorldParams:
        return PixelGridWorldParams()

    def reset(
        self,
        key: jax.Array,
        params: PixelGridWorldParams,
    ) -> tuple[jax.Array, GridWorldState]:
        state = GridWorldState(
            row=jnp.int32(0),
            col=jnp.int32(0),
            time=jnp.int32(0),
        )
        return self._render(state, params), state

    def step(
        self,
        key: jax.Array,
        state: GridWorldState,
        action: jax.Array,
        params: PixelGridWorldParams,
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

        return (
            self._render(new_state, params),
            new_state,
            reward,
            done,
            {"terminated": terminated, "truncated": truncated},
        )

    def observation_space(self, params: PixelGridWorldParams) -> Image:
        img_size = params.size * params.cell_px
        return Image(height=img_size, width=img_size, channels=3)

    def action_space(self, params: PixelGridWorldParams) -> Discrete:
        return Discrete(n=4)

    @staticmethod
    def _render(state: GridWorldState, params: PixelGridWorldParams) -> jax.Array:
        """Render the grid as an (H, W, 3) uint8 image."""
        size = params.size
        cell_px = params.cell_px
        img_h = size * cell_px
        img_w = size * cell_px

        # Build row/col index grids
        row_idx = jnp.arange(img_h) // cell_px  # (img_h,)
        col_idx = jnp.arange(img_w) // cell_px  # (img_w,)

        # Broadcast to (img_h, img_w)
        row_grid = row_idx[:, None] * jnp.ones(img_w, dtype=jnp.int32)[None, :]
        col_grid = jnp.ones(img_h, dtype=jnp.int32)[:, None] * col_idx[None, :]

        # Agent mask: white (255, 255, 255)
        agent_mask = (row_grid == state.row) & (col_grid == state.col)

        # Goal mask: green (0, 255, 0)
        goal_mask = (row_grid == size - 1) & (col_grid == size - 1)

        # Build RGB channels
        r = jnp.where(agent_mask, jnp.uint8(255), jnp.uint8(0))
        g = jnp.where(agent_mask | goal_mask, jnp.uint8(255), jnp.uint8(0))
        b = jnp.where(agent_mask, jnp.uint8(255), jnp.uint8(0))

        return jnp.stack([r, g, b], axis=-1)
