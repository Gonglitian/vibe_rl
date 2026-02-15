"""Tests for the pure-functional DQN agent."""

import jax
import jax.numpy as jnp
import pytest

from vibe_rl.agent.base import Agent
from vibe_rl.algorithms.dqn import DQN, DQNConfig, DQNState
from vibe_rl.algorithms.dqn.agent import DQNMetrics
from vibe_rl.types import Transition

OBS_SHAPE = (4,)
N_ACTIONS = 2
RNG = jax.random.key(42)


@pytest.fixture
def config():
    return DQNConfig(
        hidden_sizes=(32, 32),
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=100,
        target_update_freq=10,
    )


@pytest.fixture
def state(config):
    return DQN.init(RNG, OBS_SHAPE, N_ACTIONS, config)


@pytest.fixture
def batch():
    """A small random batch of transitions."""
    rng = jax.random.key(0)
    k1, k2 = jax.random.split(rng)
    B = 16
    return Transition(
        obs=jax.random.normal(k1, (B, 4)),
        action=jax.random.randint(k2, (B,), 0, N_ACTIONS),
        reward=jnp.ones(B),
        next_obs=jax.random.normal(k1, (B, 4)),
        done=jnp.zeros(B),
    )


class TestDQNInit:
    def test_returns_dqn_state(self, state):
        assert isinstance(state, DQNState)

    def test_step_starts_at_zero(self, state):
        assert state.step == 0

    def test_params_are_equinox_module(self, state):
        import equinox as eqx
        assert isinstance(state.params, eqx.Module)
        assert isinstance(state.target_params, eqx.Module)

    def test_deterministic_init(self, config):
        """Same key produces same params."""
        s1 = DQN.init(RNG, OBS_SHAPE, N_ACTIONS, config)
        s2 = DQN.init(RNG, OBS_SHAPE, N_ACTIONS, config)
        leaves1 = jax.tree.leaves(s1.params)
        leaves2 = jax.tree.leaves(s2.params)
        for a, b in zip(leaves1, leaves2):
            assert jnp.array_equal(a, b)

    def test_different_keys_different_params(self, config):
        """Different keys produce different params."""
        s1 = DQN.init(jax.random.key(0), OBS_SHAPE, N_ACTIONS, config)
        s2 = DQN.init(jax.random.key(1), OBS_SHAPE, N_ACTIONS, config)
        leaves1 = jax.tree.leaves(s1.params)
        leaves2 = jax.tree.leaves(s2.params)
        any_different = any(not jnp.array_equal(a, b) for a, b in zip(leaves1, leaves2))
        assert any_different


class TestDQNAct:
    def test_returns_action_and_state(self, state, config):
        obs = jnp.zeros(4)
        action, new_state = DQN.act(state, obs, config=config, explore=False)
        assert action.shape == ()
        assert isinstance(new_state, DQNState)

    def test_greedy_action_is_deterministic(self, state, config):
        """Greedy mode should return the same action for same state."""
        obs = jnp.ones(4)
        a1, _ = DQN.act(state, obs, config=config, explore=False)
        a2, _ = DQN.act(state, obs, config=config, explore=False)
        assert jnp.array_equal(a1, a2)

    def test_explore_uses_different_rng(self, state, config):
        """Each act call advances the RNG key."""
        obs = jnp.zeros(4)
        _, s1 = DQN.act(state, obs, config=config, explore=True)
        _, s2 = DQN.act(s1, obs, config=config, explore=True)
        # RNG keys should differ
        assert not jnp.array_equal(s1.rng, s2.rng)

    def test_action_in_valid_range(self, state, config):
        """Actions should be in [0, n_actions)."""
        obs = jnp.zeros(4)
        for _ in range(20):
            action, state = DQN.act(state, obs, config=config, explore=True)
            assert 0 <= int(action) < N_ACTIONS

    def test_act_is_jittable(self, state, config):
        """act works inside jax.jit."""
        obs = jnp.zeros(4)
        # Already jitted via decorator, but verify it runs
        action, new_state = DQN.act(state, obs, config=config, explore=True)
        assert action.shape == ()


class TestDQNUpdate:
    def test_returns_state_and_metrics(self, state, batch, config):
        new_state, metrics = DQN.update(state, batch, config=config)
        assert isinstance(new_state, DQNState)
        assert isinstance(metrics, DQNMetrics)

    def test_step_increments(self, state, batch, config):
        new_state, _ = DQN.update(state, batch, config=config)
        assert new_state.step == 1

    def test_loss_is_finite(self, state, batch, config):
        _, metrics = DQN.update(state, batch, config=config)
        assert jnp.isfinite(metrics.loss)

    def test_params_change_after_update(self, state, batch, config):
        new_state, _ = DQN.update(state, batch, config=config)
        old_leaves = jax.tree.leaves(state.params)
        new_leaves = jax.tree.leaves(new_state.params)
        any_changed = any(not jnp.array_equal(a, b) for a, b in zip(old_leaves, new_leaves))
        assert any_changed

    def test_target_unchanged_before_freq(self, state, batch, config):
        """Target params should not change until target_update_freq steps."""
        new_state, _ = DQN.update(state, batch, config=config)
        old_target = jax.tree.leaves(state.target_params)
        new_target = jax.tree.leaves(new_state.target_params)
        for a, b in zip(old_target, new_target):
            assert jnp.array_equal(a, b)

    def test_target_updates_at_freq(self, state, batch, config):
        """Target params should update at target_update_freq."""
        # Run target_update_freq updates
        s = state
        for _ in range(config.target_update_freq):
            s, _ = DQN.update(s, batch, config=config)
        # After exactly target_update_freq steps, target should match online
        online_leaves = jax.tree.leaves(s.params)
        target_leaves = jax.tree.leaves(s.target_params)
        for a, b in zip(online_leaves, target_leaves):
            assert jnp.array_equal(a, b)

    def test_update_is_jittable(self, state, batch, config):
        """update works inside jax.jit (already jitted via decorator)."""
        new_state, metrics = DQN.update(state, batch, config=config)
        assert jnp.isfinite(metrics.loss)

    def test_multiple_updates_reduce_loss(self, state, batch, config):
        """Loss should generally decrease over repeated updates on same batch."""
        s = state
        losses = []
        for _ in range(50):
            s, metrics = DQN.update(s, batch, config=config)
            losses.append(float(metrics.loss))
        # Loss at end should be less than loss at start
        assert losses[-1] < losses[0]


class TestDQNProtocolCompliance:
    def test_satisfies_agent_protocol(self):
        """DQN class satisfies the Agent runtime protocol check."""
        assert isinstance(DQN, Agent)
