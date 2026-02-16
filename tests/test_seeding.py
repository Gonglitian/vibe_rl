"""Tests for vibe_rl.seeding."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from vibe_rl.seeding import fold_in, make_rng, split_key, split_keys


class TestMakeRng:
    def test_returns_prng_key(self) -> None:
        rng = make_rng(42)
        assert rng.shape == (2,) or rng.shape == ()  # depends on jax config
        assert rng.dtype in (jnp.uint32, jnp.uint64)

    def test_deterministic(self) -> None:
        a = make_rng(42)
        b = make_rng(42)
        assert jnp.array_equal(a, b)

    def test_different_seeds_differ(self) -> None:
        a = make_rng(0)
        b = make_rng(1)
        assert not jnp.array_equal(a, b)


class TestSplitKey:
    def test_returns_two_keys(self) -> None:
        rng = make_rng(0)
        a, b = split_key(rng)
        assert not jnp.array_equal(a, b)

    def test_deterministic(self) -> None:
        rng = make_rng(0)
        a1, b1 = split_key(rng)
        a2, b2 = split_key(rng)
        assert jnp.array_equal(a1, a2)
        assert jnp.array_equal(b1, b2)


class TestSplitKeys:
    def test_returns_n_plus_1_keys(self) -> None:
        rng = make_rng(0)
        keys = split_keys(rng, n=3)
        assert len(keys) == 4  # rng + 3 subkeys

    def test_all_keys_different(self) -> None:
        rng = make_rng(0)
        keys = split_keys(rng, n=5)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                assert not jnp.array_equal(keys[i], keys[j])

    def test_n_equals_1(self) -> None:
        rng = make_rng(0)
        new_rng, sub = split_keys(rng, n=1)
        assert not jnp.array_equal(new_rng, sub)

    def test_unpacking_pattern(self) -> None:
        rng = make_rng(42)
        rng, agent_key, env_key = split_keys(rng, n=2)
        # All should be valid keys
        _ = jax.random.uniform(agent_key)
        _ = jax.random.uniform(env_key)


class TestFoldIn:
    def test_deterministic(self) -> None:
        rng = make_rng(0)
        a = fold_in(rng, 1)
        b = fold_in(rng, 1)
        assert jnp.array_equal(a, b)

    def test_different_data_gives_different_keys(self) -> None:
        rng = make_rng(0)
        a = fold_in(rng, 0)
        b = fold_in(rng, 1)
        assert not jnp.array_equal(a, b)

    def test_useful_for_worker_keys(self) -> None:
        rng = make_rng(0)
        worker_keys = [fold_in(rng, i) for i in range(4)]
        # All should be different
        for i in range(4):
            for j in range(i + 1, 4):
                assert not jnp.array_equal(worker_keys[i], worker_keys[j])
