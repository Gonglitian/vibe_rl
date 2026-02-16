"""SAC networks implemented with Equinox.

Actor: obs -> (mean, log_std) for reparameterized Gaussian sampling.
TwinQNetwork: two independent Q-networks (obs, action) -> scalar Q-value.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


class GaussianActor(eqx.Module):
    """Gaussian policy: obs -> (mean, log_std).

    Outputs parameterise a diagonal Gaussian. The caller is responsible
    for reparameterized sampling and tanh squashing.
    """

    layers: list
    mean_head: eqx.nn.Linear
    log_std_head: eqx.nn.Linear

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...] = (256, 256),
        *,
        key: jax.Array,
    ) -> None:
        dims = [obs_dim, *hidden_sizes]
        n_layers = len(dims) - 1
        keys = jax.random.split(key, n_layers + 2)

        self.layers = [
            eqx.nn.Linear(d_in, d_out, key=k)
            for d_in, d_out, k in zip(dims[:-1], dims[1:], keys[:n_layers], strict=True)
        ]
        self.mean_head = eqx.nn.Linear(dims[-1], action_dim, key=keys[n_layers])
        self.log_std_head = eqx.nn.Linear(dims[-1], action_dim, key=keys[n_layers + 1])

    def __call__(self, obs: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Forward pass.

        Args:
            obs: Observation array, shape ``(*obs_shape,)``.

        Returns:
            (mean, log_std) each of shape ``(action_dim,)``.
        """
        x = obs
        for layer in self.layers:
            x = jax.nn.relu(layer(x))
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        return mean, log_std


class QNetwork(eqx.Module):
    """Single Q-network: (obs, action) -> scalar Q-value."""

    layers: list

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...] = (256, 256),
        *,
        key: jax.Array,
    ) -> None:
        input_dim = obs_dim + action_dim
        dims = [input_dim, *hidden_sizes, 1]
        keys = jax.random.split(key, len(dims) - 1)
        self.layers = [
            eqx.nn.Linear(d_in, d_out, key=k)
            for d_in, d_out, k in zip(dims[:-1], dims[1:], keys, strict=True)
        ]

    def __call__(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        """Forward pass.

        Args:
            obs: shape ``(*obs_shape,)``.
            action: shape ``(action_dim,)``.

        Returns:
            Scalar Q-value (shape ``()``).
        """
        x = jnp.concatenate([obs, action], axis=-1)
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x).squeeze(-1)


class TwinQNetwork(eqx.Module):
    """Twin Q-networks for clipped double-Q learning."""

    q1: QNetwork
    q2: QNetwork

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...] = (256, 256),
        *,
        key: jax.Array,
    ) -> None:
        k1, k2 = jax.random.split(key)
        self.q1 = QNetwork(obs_dim, action_dim, hidden_sizes, key=k1)
        self.q2 = QNetwork(obs_dim, action_dim, hidden_sizes, key=k2)

    def __call__(
        self, obs: jax.Array, action: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Return (q1_value, q2_value), each a scalar."""
        return self.q1(obs, action), self.q2(obs, action)
