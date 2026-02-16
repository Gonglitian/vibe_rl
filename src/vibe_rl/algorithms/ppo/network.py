"""Actor-Critic networks implemented with Equinox.

Three variants:
- ``ActorCategorical``: for discrete action spaces (obs -> logits)
- ``Critic``: obs -> scalar value
- ``ActorCriticShared``: shared backbone + two heads
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


class ActorCategorical(eqx.Module):
    """MLP actor for discrete action spaces: obs -> action logits."""

    layers: list

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: tuple[int, ...] = (64, 64),
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
        """Return action logits (un-normalised log-probabilities)."""
        for layer in self.layers[:-1]:
            x = jax.nn.tanh(layer(x))
        return self.layers[-1](x)


class Critic(eqx.Module):
    """MLP critic: obs -> scalar state-value V(s)."""

    layers: list

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: tuple[int, ...] = (64, 64),
        *,
        key: jax.Array,
    ) -> None:
        dims = [obs_dim, *hidden_sizes, 1]
        keys = jax.random.split(key, len(dims) - 1)
        self.layers = [
            eqx.nn.Linear(d_in, d_out, key=k)
            for d_in, d_out, k in zip(dims[:-1], dims[1:], keys)
        ]

    def __call__(self, x: jax.Array) -> jax.Array:
        """Return scalar value estimate."""
        for layer in self.layers[:-1]:
            x = jax.nn.tanh(layer(x))
        return self.layers[-1](x).squeeze(-1)


class ActorCriticShared(eqx.Module):
    """Shared-backbone actor-critic: obs -> (logits, value).

    A single MLP backbone feeds into two separate linear heads.
    """

    backbone: list
    actor_head: eqx.nn.Linear
    critic_head: eqx.nn.Linear

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: tuple[int, ...] = (64, 64),
        *,
        key: jax.Array,
    ) -> None:
        k_backbone, k_actor, k_critic = jax.random.split(key, 3)

        dims = [obs_dim, *hidden_sizes]
        keys = jax.random.split(k_backbone, len(dims) - 1)
        self.backbone = [
            eqx.nn.Linear(d_in, d_out, key=k)
            for d_in, d_out, k in zip(dims[:-1], dims[1:], keys)
        ]
        self.actor_head = eqx.nn.Linear(hidden_sizes[-1], n_actions, key=k_actor)
        self.critic_head = eqx.nn.Linear(hidden_sizes[-1], 1, key=k_critic)

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return ``(action_logits, value)``."""
        for layer in self.backbone:
            x = jax.nn.tanh(layer(x))
        logits = self.actor_head(x)
        value = self.critic_head(x).squeeze(-1)
        return logits, value
