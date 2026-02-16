"""JAX PRNG key management utilities.

Provides helpers for creating and splitting PRNG keys in the
``jax.random`` style. All randomness flows through explicit keys —
no global state.

Usage::

    from vibe_rl.seeding import make_rng, split_key, split_keys

    rng = make_rng(42)
    rng, agent_key, env_key = split_keys(rng, n=2)
"""

from __future__ import annotations

import jax


def make_rng(seed: int) -> jax.Array:
    """Create a JAX PRNG key from an integer seed.

    Equivalent to ``jax.random.PRNGKey(seed)`` but provides a
    consistent entry point for the library.
    """
    return jax.random.PRNGKey(seed)


def split_key(rng: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Split *rng* into two keys: ``(new_rng, subkey)``.

    Convenience wrapper around ``jax.random.split`` that unpacks the
    result into a 2-tuple for the common pattern::

        rng, subkey = split_key(rng)
    """
    return tuple(jax.random.split(rng))  # type: ignore[return-value]


def split_keys(rng: jax.Array, n: int) -> tuple[jax.Array, ...]:
    """Split *rng* into ``n + 1`` keys.

    Returns ``(new_rng, key_1, key_2, ..., key_n)``.  The first
    element is the continuation key; the remaining *n* keys are
    independent subkeys.

    Example::

        rng, agent_key, env_key = split_keys(rng, n=2)
    """
    keys = jax.random.split(rng, n + 1)
    return tuple(keys)  # type: ignore[return-value]


def fold_in(rng: jax.Array, data: int) -> jax.Array:
    """Deterministically derive a new key by folding *data* into *rng*.

    Useful for creating per-worker or per-episode keys without
    splitting::

        worker_key = fold_in(rng, worker_id)
    """
    return jax.random.fold_in(rng, data)
