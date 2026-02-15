"""Tests for Transition and PPOTransition containers."""

import jax
import jax.numpy as jnp
import pytest

from vibe_rl.dataprotocol.transition import (
    PPOTransition,
    Transition,
    make_dummy_transition,
)


def _make_transition() -> Transition:
    return Transition(
        obs=jnp.array([1.0, 2.0, 3.0]),
        action=jnp.array(1, dtype=jnp.int32),
        reward=jnp.array(0.5),
        next_obs=jnp.array([4.0, 5.0, 6.0]),
        done=jnp.array(False),
    )


def _make_batched_transition(batch_size: int = 4) -> Transition:
    return Transition(
        obs=jnp.ones((batch_size, 3)),
        action=jnp.zeros(batch_size, dtype=jnp.int32),
        reward=jnp.ones(batch_size),
        next_obs=jnp.ones((batch_size, 3)),
        done=jnp.zeros(batch_size, dtype=jnp.bool_),
    )


class TestTransition:
    def test_fields(self):
        t = _make_transition()
        assert t.obs.shape == (3,)
        assert t.action.shape == ()
        assert t.reward.shape == ()
        assert t.done.shape == ()

    def test_immutable(self):
        t = _make_transition()
        with pytest.raises(AttributeError):
            t.obs = jnp.zeros(3)  # type: ignore[misc]

    def test_is_pytree(self):
        t = _make_transition()
        leaves = jax.tree.leaves(t)
        assert len(leaves) == 5
        assert all(isinstance(l, jax.Array) for l in leaves)

    def test_tree_map(self):
        t = _make_transition()
        doubled = jax.tree.map(lambda x: x * 2, t)
        assert isinstance(doubled, Transition)
        assert jnp.allclose(doubled.reward, jnp.array(1.0))

    def test_jit_compatible(self):
        @jax.jit
        def add_reward(t: Transition, bonus: float) -> Transition:
            return t._replace(reward=t.reward + bonus)

        t = _make_transition()
        result = add_reward(t, 1.0)
        assert jnp.allclose(result.reward, jnp.array(1.5))

    def test_vmap_over_batch(self):
        """vmap a per-element function over a batched Transition."""

        def process_single(t: Transition) -> jax.Array:
            return t.reward * 2.0

        batch = _make_batched_transition(8)
        result = jax.vmap(process_single)(batch)
        assert result.shape == (8,)
        assert jnp.allclose(result, jnp.full(8, 2.0))

    def test_lax_scan(self):
        """Transition can be the carry in jax.lax.scan."""

        def step(carry: Transition, _x: None) -> tuple[Transition, jax.Array]:
            new_carry = carry._replace(reward=carry.reward + 1.0)
            return new_carry, carry.reward

        t = _make_transition()
        final, rewards = jax.lax.scan(step, t, None, length=5)
        assert jnp.allclose(final.reward, jnp.array(5.5))
        assert rewards.shape == (5,)


class TestPPOTransition:
    def test_fields(self):
        t = PPOTransition(
            obs=jnp.zeros(4),
            action=jnp.array(0, dtype=jnp.int32),
            reward=jnp.array(1.0),
            next_obs=jnp.zeros(4),
            done=jnp.array(False),
            log_prob=jnp.array(-0.5),
            value=jnp.array(0.9),
        )
        assert t.log_prob.shape == ()
        assert t.value.shape == ()

    def test_jit_compatible(self):
        @jax.jit
        def identity(t: PPOTransition) -> PPOTransition:
            return t

        t = PPOTransition(
            obs=jnp.zeros(4),
            action=jnp.array(0, dtype=jnp.int32),
            reward=jnp.array(1.0),
            next_obs=jnp.zeros(4),
            done=jnp.array(False),
            log_prob=jnp.array(-0.5),
            value=jnp.array(0.9),
        )
        result = identity(t)
        assert jnp.allclose(result.log_prob, t.log_prob)


class TestMakeDummyTransition:
    def test_shapes(self):
        t = make_dummy_transition((4,))
        assert t.obs.shape == (4,)
        assert t.action.dtype == jnp.int32
        assert t.done.dtype == jnp.bool_

    def test_multidim_obs(self):
        t = make_dummy_transition((84, 84, 4))
        assert t.obs.shape == (84, 84, 4)
