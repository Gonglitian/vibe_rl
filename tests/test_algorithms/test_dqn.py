"""Tests for JAX DQN network and config."""

import jax
import jax.numpy as jnp

from vibe_rl.algorithms.dqn.config import DQNConfig
from vibe_rl.algorithms.dqn.network import QNetwork


class TestQNetwork:
    def test_output_shape(self):
        key = jax.random.key(0)
        net = QNetwork(obs_dim=4, n_actions=2, hidden_sizes=(32, 32), key=key)
        x = jax.random.normal(key, (4,))
        out = net(x)
        assert out.shape == (2,)

    def test_batched_via_vmap(self):
        key = jax.random.key(0)
        net = QNetwork(obs_dim=4, n_actions=2, hidden_sizes=(32, 32), key=key)
        batch = jax.random.normal(key, (8, 4))
        out = jax.vmap(net)(batch)
        assert out.shape == (8, 2)

    def test_single_input(self):
        key = jax.random.key(1)
        net = QNetwork(obs_dim=2, n_actions=3, key=key)
        x = jax.random.normal(key, (2,))
        out = net(x)
        assert out.shape == (3,)


class TestDQNConfig:
    def test_defaults(self):
        config = DQNConfig()
        assert config.hidden_sizes == (128, 128)
        assert config.lr == 1e-3
        assert config.gamma == 0.99

    def test_frozen(self):
        config = DQNConfig()
        try:
            config.lr = 0.01  # type: ignore[misc]
            assert False, "Should have raised"
        except AttributeError:
            pass
