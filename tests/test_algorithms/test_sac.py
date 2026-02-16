"""Comprehensive tests for the SAC (Soft Actor-Critic) algorithm."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from vibe_rl.algorithms.sac.agent import SAC, SACMetrics
from vibe_rl.algorithms.sac.config import SACConfig
from vibe_rl.algorithms.sac.network import GaussianActor, QNetwork, TwinQNetwork
from vibe_rl.algorithms.sac.types import SACState
from vibe_rl.types import Transition

RNG = jax.random.PRNGKey(42)
OBS_SHAPE = (4,)
ACTION_DIM = 2
BATCH_SIZE = 16


@pytest.fixture
def config():
    return SACConfig(
        hidden_sizes=(32, 32),
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        batch_size=BATCH_SIZE,
    )


@pytest.fixture
def config_no_autotune():
    return SACConfig(
        hidden_sizes=(32, 32),
        autotune_alpha=False,
        batch_size=BATCH_SIZE,
    )


@pytest.fixture
def state(config):
    return SAC.init(RNG, OBS_SHAPE, ACTION_DIM, config)


@pytest.fixture
def state_no_autotune(config_no_autotune):
    return SAC.init(RNG, OBS_SHAPE, ACTION_DIM, config_no_autotune)


@pytest.fixture
def batch():
    key = jax.random.PRNGKey(123)
    k1, k2, k3 = jax.random.split(key, 3)
    return Transition(
        obs=jax.random.normal(k1, (BATCH_SIZE, *OBS_SHAPE)),
        action=jax.random.normal(k2, (BATCH_SIZE, ACTION_DIM)),
        reward=jax.random.normal(k3, (BATCH_SIZE,)),
        next_obs=jax.random.normal(k1, (BATCH_SIZE, *OBS_SHAPE)),
        done=jnp.zeros(BATCH_SIZE),
    )


# ── Network Tests ────────────────────────────────────────────────────


class TestGaussianActor:
    def test_output_shapes(self):
        key = jax.random.PRNGKey(0)
        actor = GaussianActor(4, 2, (32, 32), key=key)
        obs = jnp.zeros(4)
        mean, log_std = actor(obs)
        assert mean.shape == (2,)
        assert log_std.shape == (2,)

    def test_batched_via_vmap(self):
        key = jax.random.PRNGKey(0)
        actor = GaussianActor(4, 2, (32, 32), key=key)
        obs_batch = jnp.zeros((8, 4))
        means, log_stds = jax.vmap(actor)(obs_batch)
        assert means.shape == (8, 2)
        assert log_stds.shape == (8, 2)

    def test_jittable(self):
        key = jax.random.PRNGKey(0)
        actor = GaussianActor(4, 2, (32, 32), key=key)
        obs = jnp.zeros(4)
        jit_forward = eqx.filter_jit(actor)
        mean, log_std = jit_forward(obs)
        assert mean.shape == (2,)


class TestQNetwork:
    def test_output_shape(self):
        key = jax.random.PRNGKey(0)
        q_net = QNetwork(4, 2, (32, 32), key=key)
        obs = jnp.zeros(4)
        action = jnp.zeros(2)
        q = q_net(obs, action)
        assert q.shape == ()

    def test_batched_via_vmap(self):
        key = jax.random.PRNGKey(0)
        q_net = QNetwork(4, 2, (32, 32), key=key)
        obs_batch = jnp.zeros((8, 4))
        action_batch = jnp.zeros((8, 2))
        q_values = jax.vmap(q_net)(obs_batch, action_batch)
        assert q_values.shape == (8,)


class TestTwinQNetwork:
    def test_output_shapes(self):
        key = jax.random.PRNGKey(0)
        twin_q = TwinQNetwork(4, 2, (32, 32), key=key)
        obs = jnp.zeros(4)
        action = jnp.zeros(2)
        q1, q2 = twin_q(obs, action)
        assert q1.shape == ()
        assert q2.shape == ()

    def test_twin_networks_differ(self):
        key = jax.random.PRNGKey(0)
        twin_q = TwinQNetwork(4, 2, (32, 32), key=key)
        obs = jnp.ones(4)
        action = jnp.ones(2)
        q1, q2 = twin_q(obs, action)
        # Different random keys -> different params -> different outputs
        assert not jnp.allclose(q1, q2)


# ── Init Tests ───────────────────────────────────────────────────────


class TestSACInit:
    def test_returns_sac_state(self, state):
        assert isinstance(state, SACState)

    def test_step_starts_at_zero(self, state):
        assert int(state.step) == 0

    def test_deterministic_init(self, config):
        s1 = SAC.init(RNG, OBS_SHAPE, ACTION_DIM, config)
        s2 = SAC.init(RNG, OBS_SHAPE, ACTION_DIM, config)
        leaves1 = jax.tree.leaves(s1.actor_params)
        leaves2 = jax.tree.leaves(s2.actor_params)
        for a, b in zip(leaves1, leaves2, strict=False):
            assert jnp.array_equal(a, b)

    def test_different_keys_differ(self, config):
        s1 = SAC.init(jax.random.PRNGKey(0), OBS_SHAPE, ACTION_DIM, config)
        s2 = SAC.init(jax.random.PRNGKey(1), OBS_SHAPE, ACTION_DIM, config)
        leaves1 = jax.tree.leaves(s1.actor_params)
        leaves2 = jax.tree.leaves(s2.actor_params)
        any_different = any(
            not jnp.array_equal(a, b)
            for a, b in zip(leaves1, leaves2, strict=False)
        )
        assert any_different

    def test_log_alpha_initial_value(self, state, config):
        expected = jnp.log(jnp.array(config.init_alpha))
        assert jnp.allclose(state.log_alpha, expected)

    def test_critic_and_target_match_at_init(self, state):
        critic_leaves = jax.tree.leaves(state.critic_params)
        target_leaves = jax.tree.leaves(state.target_critic_params)
        for a, b in zip(critic_leaves, target_leaves, strict=False):
            assert jnp.array_equal(a, b)


# ── Act Tests ────────────────────────────────────────────────────────


class TestSACAct:
    def test_returns_action_and_state(self, state, config):
        obs = jnp.zeros(4)
        action, new_state = SAC.act(state, obs, config=config, explore=True)
        assert action.shape == (ACTION_DIM,)
        assert isinstance(new_state, SACState)

    def test_deterministic_action(self, state, config):
        obs = jnp.ones(4)
        a1, _ = SAC.act(state, obs, config=config, explore=False)
        a2, _ = SAC.act(state, obs, config=config, explore=False)
        assert jnp.allclose(a1, a2)

    def test_stochastic_actions_differ(self, state, config):
        obs = jnp.ones(4)
        a1, s1 = SAC.act(state, obs, config=config, explore=True)
        a2, _s2 = SAC.act(s1, obs, config=config, explore=True)
        # Different RNG keys should produce different actions
        assert not jnp.allclose(a1, a2)

    def test_action_in_bounds(self, state, config):
        obs = jnp.ones(4) * 5.0
        action, _ = SAC.act(state, obs, config=config, explore=True)
        assert jnp.all(action >= config.action_low)
        assert jnp.all(action <= config.action_high)

    def test_deterministic_action_in_bounds(self, state, config):
        obs = jnp.ones(4) * 10.0
        action, _ = SAC.act(state, obs, config=config, explore=False)
        assert jnp.all(action >= config.action_low)
        assert jnp.all(action <= config.action_high)

    def test_act_is_jittable(self, state, config):
        obs = jnp.zeros(4)
        # This call triggers JIT compilation via the decorator
        action, new_state = SAC.act(state, obs, config=config, explore=True)
        assert action.shape == (ACTION_DIM,)

    def test_rng_advances(self, state, config):
        obs = jnp.zeros(4)
        _, new_state = SAC.act(state, obs, config=config, explore=True)
        assert not jnp.array_equal(state.rng, new_state.rng)


# ── Update Tests ─────────────────────────────────────────────────────


class TestSACUpdate:
    def test_returns_state_and_metrics(self, state, batch, config):
        new_state, metrics = SAC.update(state, batch, config=config)
        assert isinstance(new_state, SACState)
        assert isinstance(metrics, SACMetrics)

    def test_step_increments(self, state, batch, config):
        new_state, _ = SAC.update(state, batch, config=config)
        assert int(new_state.step) == 1

    def test_metrics_are_finite(self, state, batch, config):
        _, metrics = SAC.update(state, batch, config=config)
        assert jnp.isfinite(metrics.actor_loss)
        assert jnp.isfinite(metrics.critic_loss)
        assert jnp.isfinite(metrics.alpha_loss)
        assert jnp.isfinite(metrics.alpha)
        assert jnp.isfinite(metrics.entropy)
        assert jnp.isfinite(metrics.q_mean)

    def test_alpha_positive(self, state, batch, config):
        _, metrics = SAC.update(state, batch, config=config)
        assert float(metrics.alpha) > 0

    def test_target_network_soft_updates(self, state, batch, config):
        """Target params should move toward online params after update."""
        new_state, _ = SAC.update(state, batch, config=config)
        # Target params should differ from initial (they got Polyak-updated)
        old_target_leaves = jax.tree.leaves(state.target_critic_params)
        new_target_leaves = jax.tree.leaves(new_state.target_critic_params)
        any_changed = any(
            not jnp.array_equal(a, b)
            for a, b in zip(old_target_leaves, new_target_leaves, strict=False)
        )
        assert any_changed

    def test_params_change_after_update(self, state, batch, config):
        """Both actor and critic params should change after one update."""
        new_state, _ = SAC.update(state, batch, config=config)

        actor_changed = any(
            not jnp.array_equal(a, b)
            for a, b in zip(
                jax.tree.leaves(state.actor_params),
                jax.tree.leaves(new_state.actor_params), strict=False,
            )
        )
        critic_changed = any(
            not jnp.array_equal(a, b)
            for a, b in zip(
                jax.tree.leaves(state.critic_params),
                jax.tree.leaves(new_state.critic_params), strict=False,
            )
        )
        assert actor_changed
        assert critic_changed

    def test_update_is_jittable(self, state, batch, config):
        """The update function should be jittable (via decorator)."""
        new_state, metrics = SAC.update(state, batch, config=config)
        assert jnp.isfinite(metrics.critic_loss)

    def test_multiple_updates_reduce_critic_loss(self, state, batch, config):
        """Critic loss should generally decrease over repeated updates."""
        s = state
        losses = []
        for _ in range(30):
            s, metrics = SAC.update(s, batch, config=config)
            losses.append(float(metrics.critic_loss))
        assert losses[-1] < losses[0]

    def test_no_autotune_alpha_stays_constant(
        self, state_no_autotune, batch, config_no_autotune
    ):
        """When autotune_alpha=False, log_alpha should not change."""
        initial_log_alpha = state_no_autotune.log_alpha
        s = state_no_autotune
        for _ in range(5):
            s, _ = SAC.update(s, batch, config=config_no_autotune)
        assert jnp.allclose(s.log_alpha, initial_log_alpha)

    def test_autotune_alpha_changes(self, state, batch, config):
        """When autotune_alpha=True, log_alpha should change after updates."""
        s = state
        for _ in range(5):
            s, _ = SAC.update(s, batch, config=config)
        assert not jnp.allclose(s.log_alpha, state.log_alpha)


# ── PyTree Compatibility Tests ───────────────────────────────────────


class TestSACPyTree:
    def test_sac_state_is_pytree(self, state):
        leaves = jax.tree.leaves(state)
        assert len(leaves) > 0

    def test_transition_pytree_roundtrip(self, batch):
        leaves, treedef = jax.tree.flatten(batch)
        reconstructed = treedef.unflatten(leaves)
        assert isinstance(reconstructed, Transition)
        for a, b in zip(
            jax.tree.leaves(batch), jax.tree.leaves(reconstructed), strict=False
        ):
            assert jnp.array_equal(a, b)
