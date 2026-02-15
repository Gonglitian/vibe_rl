"""Tests for the Agent protocol and base types."""

import jax
import jax.numpy as jnp
import pytest

from vibe_rl.agent.base import Agent
from vibe_rl.types import AgentState, Metrics, Transition


class TestTransition:
    def test_pytree_roundtrip(self):
        """Transition is a valid JAX pytree."""
        t = Transition(
            obs=jnp.ones(4),
            action=jnp.array(1),
            reward=jnp.array(1.0),
            next_obs=jnp.zeros(4),
            done=jnp.array(0.0),
        )
        leaves, treedef = jax.tree.flatten(t)
        reconstructed = treedef.unflatten(leaves)
        assert isinstance(reconstructed, Transition)
        for a, b in zip(jax.tree.leaves(t), jax.tree.leaves(reconstructed)):
            assert jnp.array_equal(a, b)

    def test_batched_transition(self):
        """Transitions can be batched along a leading dimension."""
        batch = Transition(
            obs=jnp.ones((32, 4)),
            action=jnp.ones(32, dtype=jnp.int32),
            reward=jnp.ones(32),
            next_obs=jnp.zeros((32, 4)),
            done=jnp.zeros(32),
        )
        assert batch.obs.shape == (32, 4)
        assert batch.action.shape == (32,)

    def test_vmap_over_transitions(self):
        """Transitions work with jax.vmap."""
        def process(t: Transition) -> jax.Array:
            return t.obs.sum() + t.reward

        batch = Transition(
            obs=jnp.ones((8, 4)),
            action=jnp.zeros(8, dtype=jnp.int32),
            reward=jnp.arange(8, dtype=jnp.float32),
            next_obs=jnp.zeros((8, 4)),
            done=jnp.zeros(8),
        )
        results = jax.vmap(process)(batch)
        assert results.shape == (8,)
        # obs.sum() = 4.0 for each, reward = 0..7
        expected = 4.0 + jnp.arange(8, dtype=jnp.float32)
        assert jnp.allclose(results, expected)


class TestAgentState:
    def test_initial_factory(self):
        """AgentState.initial creates state at step 0."""
        rng = jax.random.key(0)
        state = AgentState.initial(
            params={"w": jnp.ones(3)},
            opt_state=None,
            rng=rng,
        )
        assert state.step == 0
        assert jnp.array_equal(state.params["w"], jnp.ones(3))

    def test_pytree_compatible(self):
        """AgentState is a valid JAX pytree."""
        rng = jax.random.key(42)
        state = AgentState.initial(
            params={"w": jnp.array([1.0, 2.0])},
            opt_state=(jnp.zeros(2),),
            rng=rng,
        )
        leaves = jax.tree.leaves(state)
        assert len(leaves) > 0
        # Can be used inside jit
        @jax.jit
        def get_step(s):
            return s.step
        assert get_step(state) == 0

    def test_replace(self):
        """NamedTuple._replace works for functional state updates."""
        rng = jax.random.key(0)
        state = AgentState.initial(params=None, opt_state=None, rng=rng)
        new_state = state._replace(step=jnp.array(10, dtype=jnp.int32))
        assert new_state.step == 10
        assert state.step == 0  # original unchanged


class TestAgentProtocol:
    def test_protocol_is_runtime_checkable(self):
        """Agent protocol can be checked at runtime."""
        # A class that satisfies the protocol
        class FakeAgent:
            @staticmethod
            def init(rng, obs_shape, n_actions, config):
                return AgentState.initial(params=None, opt_state=None, rng=rng)

            @staticmethod
            def act(state, obs, *, explore=True):
                return jnp.array(0), state

            @staticmethod
            def update(state, batch):
                return state, Metrics(loss=jnp.array(0.0))

        assert isinstance(FakeAgent, Agent)

    def test_non_agent_fails_check(self):
        """Classes missing methods do not satisfy the protocol."""
        class NotAnAgent:
            pass

        assert not isinstance(NotAnAgent, Agent)
