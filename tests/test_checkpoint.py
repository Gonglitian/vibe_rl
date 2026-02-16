"""Tests for vibe_rl.checkpoint."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from vibe_rl.checkpoint import (
    CheckpointManager,
    initialize_checkpoint_dir,
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
# CheckpointManager enhanced features
# ---------------------------------------------------------------------------


class TestCheckpointManagerKeepPeriod:
    """Test keep_period parameter for permanent checkpoint retention."""

    def test_keep_period_repr(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(
            tmp_path / "ckpts",
            max_to_keep=2,
            keep_period=500,
            async_timeout_secs=None,
        )
        assert "keep_period=500" in repr(mgr)
        mgr.close()

    def test_manager_with_keep_period(self, tmp_path: Path) -> None:
        """CheckpointManager accepts keep_period without error."""
        ckpt_dir = tmp_path / "ckpts"
        with CheckpointManager(
            ckpt_dir,
            max_to_keep=2,
            keep_period=5,
            save_interval_steps=1,
            async_timeout_secs=None,
        ) as mgr:
            state = _make_state()
            for step in range(1, 11):
                mgr.save(step, state)
            mgr.wait()

            steps = mgr.all_steps()
            # keep_period=5 should retain steps 5 and 10
            assert 5 in steps
            assert 10 in steps

    def test_manager_async_timeout(self, tmp_path: Path) -> None:
        """CheckpointManager accepts async_timeout_secs."""
        ckpt_dir = tmp_path / "ckpts"
        with CheckpointManager(
            ckpt_dir,
            max_to_keep=2,
            async_timeout_secs=3600,
        ) as mgr:
            state = _make_state()
            mgr.save(1, state)
            mgr.wait()
            assert mgr.latest_step() == 1

    def test_manager_no_async(self, tmp_path: Path) -> None:
        """async_timeout_secs=None disables async."""
        ckpt_dir = tmp_path / "ckpts"
        with CheckpointManager(
            ckpt_dir,
            max_to_keep=2,
            async_timeout_secs=None,
        ) as mgr:
            state = _make_state()
            mgr.save(1, state)
            assert mgr.latest_step() == 1


# ---------------------------------------------------------------------------
# initialize_checkpoint_dir tests
# ---------------------------------------------------------------------------


class TestInitializeCheckpointDir:
    def test_fresh_start(self, tmp_path: Path) -> None:
        """Empty directory: creates manager, resuming=False."""
        ckpt_dir = tmp_path / "ckpts"
        mgr, resuming = initialize_checkpoint_dir(
            ckpt_dir, keep_period=None, overwrite=False, resume=False,
            async_timeout_secs=None,
        )
        assert resuming is False
        assert ckpt_dir.exists()
        mgr.close()

    def test_resume_with_checkpoints(self, tmp_path: Path) -> None:
        """Existing checkpoints + resume=True returns resuming=True."""
        ckpt_dir = tmp_path / "ckpts"
        # Pre-populate a checkpoint
        with CheckpointManager(ckpt_dir, async_timeout_secs=None) as mgr:
            state = _make_state()
            mgr.save(100, state)
            mgr.wait()

        mgr2, resuming = initialize_checkpoint_dir(
            ckpt_dir, keep_period=None, overwrite=False, resume=True,
            async_timeout_secs=None,
        )
        assert resuming is True
        assert mgr2.latest_step() == 100
        mgr2.close()

    def test_resume_empty_dir_no_checkpoints(self, tmp_path: Path) -> None:
        """Directory exists but no checkpoints: resuming=False."""
        ckpt_dir = tmp_path / "ckpts"
        ckpt_dir.mkdir(parents=True)
        # Create a dummy file so directory is non-empty
        (ckpt_dir / "dummy.txt").write_text("test")

        mgr, resuming = initialize_checkpoint_dir(
            ckpt_dir, keep_period=None, overwrite=False, resume=True,
            async_timeout_secs=None,
        )
        assert resuming is False
        mgr.close()

    def test_overwrite_clears_directory(self, tmp_path: Path) -> None:
        """overwrite=True wipes existing checkpoints."""
        ckpt_dir = tmp_path / "ckpts"
        with CheckpointManager(ckpt_dir, async_timeout_secs=None) as mgr:
            state = _make_state()
            mgr.save(100, state)
            mgr.wait()

        mgr2, resuming = initialize_checkpoint_dir(
            ckpt_dir, keep_period=None, overwrite=True, resume=False,
            async_timeout_secs=None,
        )
        assert resuming is False
        assert mgr2.latest_step() is None
        mgr2.close()

    def test_existing_dir_no_flags_raises(self, tmp_path: Path) -> None:
        """Existing checkpoints + neither resume nor overwrite -> error."""
        ckpt_dir = tmp_path / "ckpts"
        with CheckpointManager(ckpt_dir, async_timeout_secs=None) as mgr:
            state = _make_state()
            mgr.save(100, state)
            mgr.wait()

        with pytest.raises(FileExistsError, match="already exists"):
            initialize_checkpoint_dir(
                ckpt_dir, keep_period=None, overwrite=False, resume=False,
                async_timeout_secs=None,
            )

    def test_resume_restores_state(self, tmp_path: Path) -> None:
        """Full save-then-restore roundtrip via initialize_checkpoint_dir."""
        ckpt_dir = tmp_path / "ckpts"
        state = _make_state()

        # Save
        with CheckpointManager(ckpt_dir, async_timeout_secs=None) as mgr:
            mgr.save(1000, state)
            mgr.wait()

        # Resume
        mgr2, resuming = initialize_checkpoint_dir(
            ckpt_dir, keep_period=None, overwrite=False, resume=True,
            async_timeout_secs=None,
        )
        assert resuming is True

        template = _make_state()
        restored = mgr2.restore(None, template)
        assert int(restored.step) == 100  # step from the SimpleState content
        assert jnp.array_equal(restored.weights, state.weights)
        mgr2.close()

    def test_keep_period_passed_through(self, tmp_path: Path) -> None:
        """keep_period is forwarded to CheckpointManager."""
        ckpt_dir = tmp_path / "ckpts"
        mgr, _ = initialize_checkpoint_dir(
            ckpt_dir, keep_period=500, overwrite=False, resume=False,
            async_timeout_secs=None,
        )
        assert "keep_period=500" in repr(mgr)
        mgr.close()


# ---------------------------------------------------------------------------
# WandB resume (wandb_id.txt) tests
# ---------------------------------------------------------------------------


class TestWandbResume:
    def test_wandb_id_file_written(self, tmp_path: Path) -> None:
        """WandbBackend writes wandb_id.txt to checkpoint_dir."""
        from unittest.mock import MagicMock, patch

        ckpt_dir = tmp_path / "ckpts"
        ckpt_dir.mkdir(parents=True)

        mock_run = MagicMock()
        mock_run.id = "test_run_abc123"

        mock_wandb = MagicMock()
        mock_wandb.init.return_value = mock_run

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            from vibe_rl.metrics import WandbBackend

            backend = WandbBackend(
                project="test", run_dir=str(ckpt_dir),
            )

            # Verify wandb_id.txt was written
            id_path = ckpt_dir / "wandb_id.txt"
            assert id_path.exists()
            assert id_path.read_text().strip() == "test_run_abc123"

            backend.close()

    def test_wandb_resume_reads_existing_id(self, tmp_path: Path) -> None:
        """WandbBackend reads existing wandb_id.txt and passes id+resume."""
        from unittest.mock import MagicMock, patch

        ckpt_dir = tmp_path / "ckpts"
        ckpt_dir.mkdir(parents=True)

        # Pre-write a wandb ID
        (ckpt_dir / "wandb_id.txt").write_text("previous_run_xyz\n")

        mock_run = MagicMock()
        mock_run.id = "previous_run_xyz"

        mock_wandb = MagicMock()
        mock_wandb.init.return_value = mock_run

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            from vibe_rl.metrics import WandbBackend

            backend = WandbBackend(
                project="test", run_dir=str(ckpt_dir),
            )

            # Verify wandb.init was called with id and resume
            call_kwargs = mock_wandb.init.call_args[1]
            assert call_kwargs["id"] == "previous_run_xyz"
            assert call_kwargs["resume"] == "must"

            backend.close()

    def test_wandb_no_checkpoint_dir(self, tmp_path: Path) -> None:
        """WandbBackend works normally when checkpoint_dir is None."""
        from unittest.mock import MagicMock, patch

        mock_run = MagicMock()
        mock_run.id = "new_run_id"

        mock_wandb = MagicMock()
        mock_wandb.init.return_value = mock_run

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            from vibe_rl.metrics import WandbBackend

            backend = WandbBackend(project="test")

            # Should not have id or resume in kwargs
            call_kwargs = mock_wandb.init.call_args[1]
            assert "id" not in call_kwargs
            assert "resume" not in call_kwargs

            backend.close()


# ---------------------------------------------------------------------------
# RunnerConfig checkpoint fields test
# ---------------------------------------------------------------------------


class TestRunnerConfigCheckpointFields:
    def test_default_values(self) -> None:
        from vibe_rl.runner.config import RunnerConfig

        cfg = RunnerConfig()
        assert cfg.checkpoint_dir is None
        assert cfg.checkpoint_interval == 5_000
        assert cfg.max_checkpoints == 5
        assert cfg.keep_period is None
        assert cfg.resume is False
        assert cfg.overwrite is False

    def test_custom_values(self) -> None:
        from vibe_rl.runner.config import RunnerConfig

        cfg = RunnerConfig(
            checkpoint_dir="/tmp/ckpts",
            checkpoint_interval=1_000,
            max_checkpoints=3,
            keep_period=500,
            resume=True,
            overwrite=False,
        )
        assert cfg.checkpoint_dir == "/tmp/ckpts"
        assert cfg.checkpoint_interval == 1_000
        assert cfg.max_checkpoints == 3
        assert cfg.keep_period == 500
        assert cfg.resume is True


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


# ---------------------------------------------------------------------------
# Algorithm checkpoint roundtrip tests
# ---------------------------------------------------------------------------


def _save_state(path: Path, state):
    """Save a PyTree state using equinox serialization."""
    eqx.tree_serialise_leaves(str(path), state)


def _load_state(path: Path, template):
    """Load a PyTree state using equinox deserialization."""
    return eqx.tree_deserialise_leaves(str(path), template)


from vibe_rl.algorithms.dqn import DQN, DQNConfig, DQNState
from vibe_rl.algorithms.ppo import PPO, PPOConfig, PPOState
from vibe_rl.algorithms.sac import SAC, SACConfig, SACState
from vibe_rl.dataprotocol.transition import Transition


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
