"""Tests for TrainState containers and factory functions."""

import jax
import jax.numpy as jnp
import optax

from vibe_rl.dataprotocol.train_state import (
    DQNTrainState,
    TrainState,
    create_dqn_train_state,
    create_train_state,
)


def _dummy_params() -> dict[str, jax.Array]:
    """Minimal param dict standing in for an Equinox model."""
    return {"w": jnp.ones((4, 2)), "b": jnp.zeros(2)}


class TestTrainState:
    def test_create(self):
        params = _dummy_params()
        tx = optax.adam(1e-3)
        state = create_train_state(params, tx)
        assert isinstance(state, TrainState)
        assert state.step.shape == ()
        assert int(state.step) == 0

    def test_immutable(self):
        params = _dummy_params()
        tx = optax.adam(1e-3)
        state = create_train_state(params, tx)
        # NamedTuples don't allow attribute assignment
        try:
            state.step = jnp.array(1)  # type: ignore[misc]
            assert False, "Should have raised"
        except AttributeError:
            pass

    def test_functional_update(self):
        params = _dummy_params()
        tx = optax.adam(1e-3)
        state = create_train_state(params, tx)
        new_state = state._replace(step=state.step + 1)
        assert int(new_state.step) == 1
        assert int(state.step) == 0  # original unchanged

    def test_is_pytree(self):
        params = _dummy_params()
        tx = optax.adam(1e-3)
        state = create_train_state(params, tx)
        leaves = jax.tree.leaves(state)
        assert len(leaves) > 0
        assert all(isinstance(l, jax.Array) for l in leaves)

    def test_jit_compatible(self):
        """TrainState can pass through jit boundaries."""
        params = _dummy_params()
        tx = optax.adam(1e-3)
        state = create_train_state(params, tx)

        @jax.jit
        def increment(s: TrainState) -> TrainState:
            return s._replace(step=s.step + 1)

        result = increment(state)
        assert int(result.step) == 1

    def test_optax_update_in_jit(self):
        """Full gradient-update cycle inside jit."""
        params = _dummy_params()
        tx = optax.adam(1e-3)
        state = create_train_state(params, tx)

        @jax.jit
        def update(s: TrainState) -> TrainState:
            # Fake gradients
            grads = jax.tree.map(jnp.ones_like, s.params)
            updates, new_opt_state = tx.update(grads, s.opt_state, s.params)
            new_params = optax.apply_updates(s.params, updates)
            return s._replace(
                params=new_params,
                opt_state=new_opt_state,
                step=s.step + 1,
            )

        new_state = update(state)
        assert int(new_state.step) == 1
        # Params should have changed
        assert not jnp.allclose(new_state.params["w"], state.params["w"])


class TestDQNTrainState:
    def test_create(self):
        params = _dummy_params()
        tx = optax.adam(1e-3)
        state = create_dqn_train_state(params, tx, epsilon_start=1.0)
        assert isinstance(state, DQNTrainState)
        assert float(state.epsilon) == 1.0

    def test_target_params_independent(self):
        """Target params start as same values but are an independent copy."""
        params = _dummy_params()
        tx = optax.adam(1e-3)
        state = create_dqn_train_state(params, tx)

        # Update online params, target should stay the same
        new_params = jax.tree.map(lambda p: p + 1.0, state.params)
        new_state = state._replace(params=new_params)

        # Target unchanged because NamedTuple _replace doesn't deep-copy
        # but the original jax arrays are immutable, so target_params
        # still points to the original values
        assert jnp.allclose(new_state.target_params["w"], jnp.ones((4, 2)))
        assert jnp.allclose(new_state.params["w"], jnp.full((4, 2), 2.0))

    def test_jit_target_update(self):
        """Polyak averaging of target params inside jit."""
        params = _dummy_params()
        tx = optax.adam(1e-3)
        state = create_dqn_train_state(params, tx)

        # Modify online params
        state = state._replace(
            params=jax.tree.map(lambda p: p * 10.0, state.params)
        )

        @jax.jit
        def hard_update(s: DQNTrainState) -> DQNTrainState:
            return s._replace(target_params=s.params)

        updated = hard_update(state)
        assert jnp.allclose(updated.target_params["w"], updated.params["w"])

    def test_lax_scan(self):
        """DQNTrainState works as scan carry."""
        params = _dummy_params()
        tx = optax.adam(1e-3)
        state = create_dqn_train_state(params, tx)

        def step_fn(s: DQNTrainState, _: None) -> tuple[DQNTrainState, jax.Array]:
            new_s = s._replace(
                step=s.step + 1,
                epsilon=s.epsilon * 0.99,
            )
            return new_s, s.epsilon

        final, epsilons = jax.lax.scan(step_fn, state, None, length=10)
        assert int(final.step) == 10
        assert epsilons.shape == (10,)
        assert float(final.epsilon) < 1.0
