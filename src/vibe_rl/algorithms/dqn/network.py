"""Q-Network implemented with Equinox."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


class QNetwork(eqx.Module):
    """Simple MLP Q-network: obs -> Q(s, a) for each discrete action."""

    layers: list

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: tuple[int, ...] = (128, 128),
        *,
        key: jax.Array,
    ) -> None:
        dims = [obs_dim, *hidden_sizes, n_actions]
        keys = jax.random.split(key, len(dims) - 1)
        self.layers = [
            eqx.nn.Linear(d_in, d_out, key=k)
            for d_in, d_out, k in zip(dims[:-1], dims[1:], keys)
        ]

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x)
