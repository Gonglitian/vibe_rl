"""Pure-function schedules for use inside ``jax.jit``.

All schedules follow the signature ``schedule(step) -> value`` and are
compatible with JIT compilation (no Python-side state).

Usage::

    from vibe_rl.schedule import linear_schedule

    schedule = linear_schedule(start=1.0, end=0.01, steps=10_000)
    eps = schedule(step)          # works inside @jax.jit
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp

Schedule = Callable[[int | jnp.ndarray], jnp.ndarray]


def linear_schedule(
    start: float,
    end: float,
    steps: int,
) -> Schedule:
    """Return a pure function that linearly interpolates from *start* to *end*.

    The returned callable maps ``step -> value`` and is safe to use
    inside ``jax.jit``.

    Parameters
    ----------
    start:
        Value at step 0.
    end:
        Value at step *steps* (and beyond).
    steps:
        Number of steps over which to interpolate.

    Returns
    -------
    A callable ``(step: int | Array) -> Array``.
    """
    _start = jnp.float32(start)
    _end = jnp.float32(end)
    _steps = jnp.float32(max(steps, 1))

    def _schedule(step: int | jnp.ndarray) -> jnp.ndarray:
        frac = jnp.clip(jnp.float32(step) / _steps, 0.0, 1.0)
        return _start + frac * (_end - _start)

    return _schedule
