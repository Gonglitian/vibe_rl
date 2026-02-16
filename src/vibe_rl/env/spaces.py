"""JAX-native action and observation spaces.

All spaces are immutable PyTree nodes compatible with jit/vmap.
They describe the shape, dtype and bounds of observations/actions
but do NOT carry mutable state.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


class Discrete(eqx.Module):
    """Space of integers {0, 1, ..., n-1}."""

    n: int = eqx.field(static=True)

    def sample(self, key: jax.Array) -> jax.Array:
        return jax.random.randint(key, shape=(), minval=0, maxval=self.n)

    def contains(self, x: jax.Array) -> jax.Array:
        return (x >= 0) & (x < self.n) & (x == jnp.floor(x))

    @property
    def shape(self) -> tuple[int, ...]:
        return ()

    @property
    def dtype(self) -> jnp.dtype:
        return jnp.int32


class Box(eqx.Module):
    """Bounded n-dimensional continuous space.

    Bounds are stored as static tuples so the space is hashable and
    can be used as a JIT-static argument.
    """

    _low: tuple[float, ...] = eqx.field(static=True)
    _high: tuple[float, ...] = eqx.field(static=True)
    _shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        low: float | jax.Array,
        high: float | jax.Array,
        shape: tuple[int, ...] | None = None,
    ) -> None:
        if shape is not None:
            low_arr = jnp.full(shape, low, dtype=jnp.float32)
            high_arr = jnp.full(shape, high, dtype=jnp.float32)
        else:
            low_arr = jnp.asarray(low, dtype=jnp.float32)
            high_arr = jnp.asarray(high, dtype=jnp.float32)
        self._low = tuple(float(x) for x in low_arr.flatten())
        self._high = tuple(float(x) for x in high_arr.flatten())
        self._shape = tuple(int(d) for d in low_arr.shape)

    @property
    def low(self) -> jax.Array:
        return jnp.array(self._low, dtype=jnp.float32).reshape(self._shape)

    @property
    def high(self) -> jax.Array:
        return jnp.array(self._high, dtype=jnp.float32).reshape(self._shape)

    def sample(self, key: jax.Array) -> jax.Array:
        return jax.random.uniform(key, shape=self._shape, minval=self.low, maxval=self.high)

    def contains(self, x: jax.Array) -> jax.Array:
        return jnp.all((x >= self.low) & (x <= self.high))

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> jnp.dtype:
        return jnp.float32


class Image(eqx.Module):
    """Image observation space: (height, width, channels) with uint8 pixels.

    Values are integers in [0, 255]. Shape is always 3-D ``(H, W, C)``.
    """

    height: int = eqx.field(static=True)
    width: int = eqx.field(static=True)
    channels: int = eqx.field(static=True)

    def sample(self, key: jax.Array) -> jax.Array:
        return jax.random.randint(
            key, shape=(self.height, self.width, self.channels), minval=0, maxval=256,
        ).astype(jnp.uint8)

    def contains(self, x: jax.Array) -> jax.Array:
        shape_ok = x.shape == (self.height, self.width, self.channels)
        return shape_ok & jnp.all((x >= 0) & (x <= 255))

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.height, self.width, self.channels)

    @property
    def dtype(self) -> jnp.dtype:
        return jnp.uint8


class MultiBinary(eqx.Module):
    """Space of binary vectors of length n."""

    n: int = eqx.field(static=True)

    def sample(self, key: jax.Array) -> jax.Array:
        return jax.random.bernoulli(key, shape=(self.n,)).astype(jnp.int32)

    def contains(self, x: jax.Array) -> jax.Array:
        return jnp.all((x == 0) | (x == 1))

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.n,)

    @property
    def dtype(self) -> jnp.dtype:
        return jnp.int32
