from __future__ import annotations

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp


class QNetwork(eqx.Module):
    """Simple MLP Q-network: obs -> Q-value for each action.

    Activation functions are stored as static fields so that all
    pytree leaves are JAX arrays.
    """

    layers: list[eqx.nn.Linear]
    activation: Callable = eqx.field(static=True)

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: tuple[int, ...] = (128, 128),
        *,
        key: jax.Array,
    ) -> None:
        sizes = [obs_dim, *hidden_sizes, n_actions]
        keys = jax.random.split(key, len(sizes) - 1)
        self.layers = [
            eqx.nn.Linear(sizes[i], sizes[i + 1], key=keys[i])
            for i in range(len(sizes) - 1)
        ]
        self.activation = jax.nn.relu

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)
