"""Tests for vibe_rl.checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from vibe_rl.checkpoint import (
    load_checkpoint,
    load_eqx,
    load_metadata,
    save_checkpoint,
    save_eqx,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class SimpleState(NamedTuple):
    weights: jax.Array
    bias: jax.Array
    step: jax.Array


def _make_state() -> SimpleState:
    return SimpleState(
        weights=jnp.ones((3, 4)),
        bias=jnp.zeros(4),
        step=jnp.array(100, dtype=jnp.int32),
    )


class SimpleMLP(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(self, key: jax.Array) -> None:
        self.linear = eqx.nn.Linear(4, 2, key=key)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.linear(x)


# ---------------------------------------------------------------------------
# Equinox serialization tests
# ---------------------------------------------------------------------------


class TestSaveLoadEqx:
    def test_namedtuple_roundtrip(self, tmp_path: Path) -> None:
        state = _make_state()
        path = tmp_path / "state.eqx"
        save_eqx(path, state)

        restored = load_eqx(path, _make_state())
        assert jnp.array_equal(restored.weights, state.weights)
        assert jnp.array_equal(restored.bias, state.bias)
        assert int(restored.step) == 100

    def test_equinox_model_roundtrip(self, tmp_path: Path) -> None:
        key = jax.random.PRNGKey(0)
        model = SimpleMLP(key)
        path = tmp_path / "model.eqx"
        save_eqx(path, model)

        skeleton = SimpleMLP(jax.random.PRNGKey(1))  # different key = different weights
        restored = load_eqx(path, skeleton)

        x = jnp.ones(4)
        assert jnp.allclose(model(x), restored(x))

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        state = _make_state()
        path = tmp_path / "deep" / "nested" / "state.eqx"
        save_eqx(path, state)
        assert path.exists()

    def test_dict_pytree_roundtrip(self, tmp_path: Path) -> None:
        state = {"w": jnp.array([1.0, 2.0]), "b": jnp.array(0.5)}
        path = tmp_path / "dict.eqx"
        save_eqx(path, state)
        restored = load_eqx(path, {"w": jnp.zeros(2), "b": jnp.zeros(())})
        assert jnp.array_equal(restored["w"], state["w"])

    def test_composite_pytree_roundtrip(self, tmp_path: Path) -> None:
        """Save (model, opt_state, step, rng) as a tuple."""
        key = jax.random.PRNGKey(0)
        model = SimpleMLP(key)
        import optax

        tx = optax.adam(1e-3)
        opt_state = tx.init(eqx.filter(model, eqx.is_array))
        step = jnp.array(0, dtype=jnp.int32)
        rng = jax.random.PRNGKey(42)

        ckpt = (model, opt_state, step, rng)
        path = tmp_path / "full.eqx"
        save_eqx(path, ckpt)

        skeleton_model = SimpleMLP(jax.random.PRNGKey(99))
        skeleton_opt = tx.init(eqx.filter(skeleton_model, eqx.is_array))
        skeleton = (skeleton_model, skeleton_opt, jnp.int32(0), jax.random.PRNGKey(0))

        restored_model, restored_opt, restored_step, restored_rng = load_eqx(
            path, skeleton
        )
        x = jnp.ones(4)
        assert jnp.allclose(model(x), restored_model(x))


# ---------------------------------------------------------------------------
# Orbax checkpoint tests
# ---------------------------------------------------------------------------


class TestSaveLoadCheckpoint:
    def test_basic_roundtrip(self, tmp_path: Path) -> None:
        state = _make_state()
        ckpt_dir = tmp_path / "ckpt"
        save_checkpoint(ckpt_dir, state)

        restored = load_checkpoint(ckpt_dir, _make_state())
        assert jnp.array_equal(restored.weights, state.weights)
        assert int(restored.step) == 100

    def test_with_step(self, tmp_path: Path) -> None:
        state = _make_state()
        ckpt_dir = tmp_path / "ckpts"
        save_checkpoint(ckpt_dir, state, step=500)

        # step_500 subdirectory should exist
        assert (ckpt_dir / "step_500").is_dir()

        restored = load_checkpoint(ckpt_dir, _make_state(), step=500)
        assert jnp.array_equal(restored.weights, state.weights)

    def test_with_metadata(self, tmp_path: Path) -> None:
        state = _make_state()
        meta = {"lr": 1e-3, "algo": "dqn"}
        ckpt_dir = tmp_path / "ckpt"
        save_checkpoint(ckpt_dir, state, metadata=meta)

        loaded_meta = load_metadata(ckpt_dir)
        assert loaded_meta is not None
        assert loaded_meta["lr"] == 1e-3
        assert loaded_meta["algo"] == "dqn"

    def test_metadata_with_step(self, tmp_path: Path) -> None:
        state = _make_state()
        meta = {"epoch": 5}
        ckpt_dir = tmp_path / "ckpts"
        save_checkpoint(ckpt_dir, state, step=100, metadata=meta)

        loaded_meta = load_metadata(ckpt_dir, step=100)
        assert loaded_meta is not None
        assert loaded_meta["epoch"] == 5

    def test_no_metadata_returns_none(self, tmp_path: Path) -> None:
        state = _make_state()
        ckpt_dir = tmp_path / "ckpt"
        save_checkpoint(ckpt_dir, state)

        assert load_metadata(ckpt_dir) is None

    def test_equinox_model_checkpoint(self, tmp_path: Path) -> None:
        key = jax.random.PRNGKey(0)
        model = SimpleMLP(key)
        ckpt_dir = tmp_path / "model_ckpt"
        save_checkpoint(ckpt_dir, model)

        skeleton = SimpleMLP(jax.random.PRNGKey(99))
        restored = load_checkpoint(ckpt_dir, skeleton)

        x = jnp.ones(4)
        assert jnp.allclose(model(x), restored(x))


# ---------------------------------------------------------------------------
# MetricsLogger backend integration test
# ---------------------------------------------------------------------------


class TestMetricsLoggerBackends:
    def test_logger_with_no_backends(self, tmp_path: Path) -> None:
        """Existing behavior is preserved."""
        from vibe_rl.metrics import MetricsLogger, read_metrics

        path = tmp_path / "m.jsonl"
        with MetricsLogger(path) as logger:
            logger.write({"step": 1, "loss": 0.5})
        records = read_metrics(path)
        assert len(records) == 1

    def test_logger_closes_backends(self, tmp_path: Path) -> None:
        """Backends get their close() called."""
        from vibe_rl.metrics import MetricsLogger

        closed = []

        class DummyBackend:
            def log(self, record, step=None):
                pass

            def close(self):
                closed.append(True)

        path = tmp_path / "m.jsonl"
        with MetricsLogger(path, backends=[DummyBackend()]):
            pass

        assert len(closed) == 1

    def test_logger_fans_out_to_backend(self, tmp_path: Path) -> None:
        from vibe_rl.metrics import MetricsLogger

        received = []

        class RecordingBackend:
            def log(self, record, step=None):
                received.append((record, step))

            def close(self):
                pass

        path = tmp_path / "m.jsonl"
        with MetricsLogger(path, backends=[RecordingBackend()]) as logger:
            logger.write({"step": 42, "loss": 0.1})

        assert len(received) == 1
        assert received[0][0]["step"] == 42
        assert received[0][1] == 42  # step extracted from record
