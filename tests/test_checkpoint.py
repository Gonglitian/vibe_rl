"""Tests for checkpoint save/load roundtrip using equinox serialization.

Verifies that all algorithm states can be serialized and deserialized
correctly, preserving parameter values and optimizer state.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

from vibe_rl.algorithms.dqn import DQN, DQNConfig, DQNState
from vibe_rl.algorithms.ppo import PPO, PPOConfig, PPOState
from vibe_rl.algorithms.sac import SAC, SACConfig, SACState
from vibe_rl.dataprotocol.transition import Transition


def _save_state(path: Path, state):
    """Save a PyTree state using equinox serialization."""
    eqx.tree_serialise_leaves(str(path), state)


def _load_state(path: Path, template):
    """Load a PyTree state using equinox deserialization."""
    return eqx.tree_deserialise_leaves(str(path), template)


class TestDQNCheckpoint:
    def test_save_load_roundtrip(self):
        config = DQNConfig(hidden_sizes=(32, 32))
        state = DQN.init(jax.random.PRNGKey(0), (4,), 2, config)

        # Do a few updates to change params
        k = jax.random.PRNGKey(1)
        batch = Transition(
            obs=jax.random.normal(k, (16, 4)),
            action=jax.random.randint(k, (16,), 0, 2),
            reward=jnp.ones(16),
            next_obs=jax.random.normal(k, (16, 4)),
            done=jnp.zeros(16),
        )
        for _ in range(3):
            state, _ = DQN.update(state, batch, config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dqn_state.eqx"
            _save_state(path, state)

            # Create a fresh template with same structure
            template = DQN.init(jax.random.PRNGKey(99), (4,), 2, config)
            loaded = _load_state(path, template)

        # Verify step matches
        assert int(loaded.step) == int(state.step)

        # Verify params match
        for a, b in zip(
            jax.tree.leaves(state.params),
            jax.tree.leaves(loaded.params),
            strict=False,
        ):
            assert jnp.allclose(a, b), "Params mismatch after checkpoint roundtrip"

    def test_loaded_state_produces_same_actions(self):
        config = DQNConfig(hidden_sizes=(32, 32))
        state = DQN.init(jax.random.PRNGKey(0), (4,), 2, config)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dqn_state.eqx"
            _save_state(path, state)
            template = DQN.init(jax.random.PRNGKey(99), (4,), 2, config)
            loaded = _load_state(path, template)

        obs = jnp.ones(4)
        a1, _ = DQN.act(state, obs, config=config, explore=False)
        a2, _ = DQN.act(loaded, obs, config=config, explore=False)
        assert jnp.array_equal(a1, a2)


class TestPPOCheckpoint:
    def test_save_load_roundtrip(self):
        config = PPOConfig(hidden_sizes=(32, 32), n_steps=16, n_minibatches=2, n_epochs=2)
        state = PPO.init(jax.random.PRNGKey(0), (4,), 2, config)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ppo_state.eqx"
            _save_state(path, state)
            template = PPO.init(jax.random.PRNGKey(99), (4,), 2, config)
            loaded = _load_state(path, template)

        assert int(loaded.step) == int(state.step)
        for a, b in zip(
            jax.tree.leaves(state.params),
            jax.tree.leaves(loaded.params),
            strict=False,
        ):
            assert jnp.allclose(a, b)

    def test_loaded_state_produces_same_actions(self):
        config = PPOConfig(hidden_sizes=(32, 32))
        state = PPO.init(jax.random.PRNGKey(0), (4,), 2, config)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ppo_state.eqx"
            _save_state(path, state)
            template = PPO.init(jax.random.PRNGKey(99), (4,), 2, config)
            loaded = _load_state(path, template)

        obs = jnp.ones(4)
        a1, lp1, v1, _ = PPO.act(state, obs, config=config)
        a2, lp2, v2, _ = PPO.act(loaded, obs, config=config)
        # Values should match (deterministic forward pass); actions may differ due to RNG
        assert jnp.allclose(v1, v2)


class TestSACCheckpoint:
    def test_save_load_roundtrip(self):
        config = SACConfig(hidden_sizes=(32, 32))
        state = SAC.init(jax.random.PRNGKey(0), (3,), 1, config)

        # Update to change params
        k = jax.random.PRNGKey(1)
        batch = Transition(
            obs=jax.random.normal(k, (16, 3)),
            action=jax.random.normal(k, (16, 1)),
            reward=jnp.ones(16),
            next_obs=jax.random.normal(k, (16, 3)),
            done=jnp.zeros(16),
        )
        for _ in range(3):
            state, _ = SAC.update(state, batch, config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sac_state.eqx"
            _save_state(path, state)
            template = SAC.init(jax.random.PRNGKey(99), (3,), 1, config)
            loaded = _load_state(path, template)

        assert int(loaded.step) == int(state.step)
        assert jnp.allclose(loaded.log_alpha, state.log_alpha)

        for a, b in zip(
            jax.tree.leaves(state.actor_params),
            jax.tree.leaves(loaded.actor_params),
            strict=False,
        ):
            assert jnp.allclose(a, b)

        for a, b in zip(
            jax.tree.leaves(state.critic_params),
            jax.tree.leaves(loaded.critic_params),
            strict=False,
        ):
            assert jnp.allclose(a, b)

    def test_loaded_state_deterministic_actions(self):
        config = SACConfig(hidden_sizes=(32, 32))
        state = SAC.init(jax.random.PRNGKey(0), (3,), 1, config)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sac_state.eqx"
            _save_state(path, state)
            template = SAC.init(jax.random.PRNGKey(99), (3,), 1, config)
            loaded = _load_state(path, template)

        obs = jnp.ones(3)
        a1, _ = SAC.act(state, obs, config=config, explore=False)
        a2, _ = SAC.act(loaded, obs, config=config, explore=False)
        assert jnp.allclose(a1, a2)
