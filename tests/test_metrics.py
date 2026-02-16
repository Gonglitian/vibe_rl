"""Tests for vibe_rl.metrics."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np

from vibe_rl.metrics import (
    MetricsLogger,
    WandbBackend,
    _load_wandb_id,
    _save_wandb_id,
    log_step_progress,
    read_metrics,
    resume_wandb,
    setup_logging,
)


class TestMetricsLogger:
    def test_write_and_read(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.jsonl"
        with MetricsLogger(path) as logger:
            logger.write({"step": 100, "loss": 0.5})
            logger.write({"step": 200, "loss": 0.3, "reward": 42.0})

        records = read_metrics(path)
        assert len(records) == 2
        assert records[0]["step"] == 100
        assert records[0]["loss"] == 0.5
        assert records[1]["reward"] == 42.0

    def test_auto_wall_time(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.jsonl"
        with MetricsLogger(path) as logger:
            logger.write({"step": 1})

        records = read_metrics(path)
        assert "wall_time" in records[0]
        assert isinstance(records[0]["wall_time"], float)

    def test_explicit_wall_time_not_overwritten(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.jsonl"
        with MetricsLogger(path) as logger:
            logger.write({"step": 1, "wall_time": 99.9})

        records = read_metrics(path)
        assert records[0]["wall_time"] == 99.9

    def test_jax_scalar_conversion(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.jsonl"
        with MetricsLogger(path) as logger:
            logger.write({"loss": jnp.float32(0.42)})

        records = read_metrics(path)
        assert isinstance(records[0]["loss"], float)

    def test_numpy_scalar_conversion(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.jsonl"
        with MetricsLogger(path) as logger:
            logger.write({"step": np.int64(100)})

        records = read_metrics(path)
        assert isinstance(records[0]["step"], int)

    def test_append_mode(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.jsonl"

        logger1 = MetricsLogger(path)
        logger1.write({"step": 1})
        logger1.close()

        logger2 = MetricsLogger(path)
        logger2.write({"step": 2})
        logger2.close()

        records = read_metrics(path)
        assert len(records) == 2

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "deep" / "nested" / "metrics.jsonl"
        with MetricsLogger(path) as logger:
            logger.write({"x": 1})

        assert path.exists()

    def test_read_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        assert read_metrics(path) == []

    def test_read_nonexistent_file(self, tmp_path: Path) -> None:
        assert read_metrics(tmp_path / "nope.jsonl") == []


# ---------------------------------------------------------------------------
# WandB ID persistence
# ---------------------------------------------------------------------------


class TestWandbIdPersistence:
    def test_save_and_load(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "my_run"
        run_dir.mkdir()
        _save_wandb_id(run_dir, "abc123")
        assert _load_wandb_id(run_dir) == "abc123"

    def test_load_missing_returns_none(self, tmp_path: Path) -> None:
        assert _load_wandb_id(tmp_path / "nonexistent") is None

    def test_load_empty_file_returns_none(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "wandb_id.txt").write_text("")
        assert _load_wandb_id(run_dir) is None

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "deep" / "nested" / "run"
        _save_wandb_id(run_dir, "xyz")
        assert _load_wandb_id(run_dir) == "xyz"

    def test_overwrite_existing_id(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _save_wandb_id(run_dir, "first")
        _save_wandb_id(run_dir, "second")
        assert _load_wandb_id(run_dir) == "second"


class TestWandbBackendResume:
    @patch("vibe_rl.metrics.WandbBackend.__init__", return_value=None)
    def test_resume_wandb_with_existing_id(self, mock_init: MagicMock, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _save_wandb_id(run_dir, "saved_id_123")

        resume_wandb(run_dir, project="test_proj")

        mock_init.assert_called_once()
        kwargs = mock_init.call_args
        assert kwargs[1]["id"] == "saved_id_123"
        assert kwargs[1]["resume"] == "must"
        assert kwargs[1]["project"] == "test_proj"

    @patch("vibe_rl.metrics.WandbBackend.__init__", return_value=None)
    def test_resume_wandb_no_id_starts_fresh(self, mock_init: MagicMock, tmp_path: Path) -> None:
        run_dir = tmp_path / "empty_run"
        run_dir.mkdir()

        resume_wandb(run_dir, project="fresh")

        mock_init.assert_called_once()
        kwargs = mock_init.call_args[1]
        assert "id" not in kwargs
        assert "resume" not in kwargs
        assert kwargs["project"] == "fresh"

    @patch("vibe_rl.metrics.WandbBackend.__init__", return_value=None)
    def test_resume_wandb_does_not_override_explicit_id(
        self, mock_init: MagicMock, tmp_path: Path,
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _save_wandb_id(run_dir, "saved_id")

        resume_wandb(run_dir, project="p", id="explicit_id")

        kwargs = mock_init.call_args[1]
        # Explicit kwarg should take precedence (setdefault won't overwrite)
        assert kwargs["id"] == "explicit_id"

    @patch("vibe_rl.metrics.WandbBackend.__init__", return_value=None)
    def test_resume_passes_config_and_entity(
        self, mock_init: MagicMock, tmp_path: Path,
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        resume_wandb(
            run_dir, project="proj", entity="team", config={"lr": 0.001},
        )

        kwargs = mock_init.call_args[1]
        assert kwargs["entity"] == "team"
        assert kwargs["config"] == {"lr": 0.001}


class TestWandbBackendPersistence:
    """Test that WandbBackend saves its ID when run_dir is provided."""

    def _make_mock_wandb(self) -> MagicMock:
        mock_wandb = MagicMock()
        mock_run = MagicMock()
        mock_run.id = "auto_generated_id"
        mock_wandb.init.return_value = mock_run
        return mock_wandb

    @patch.dict("sys.modules", {"wandb": MagicMock()})
    def test_saves_id_when_run_dir_provided(self, tmp_path: Path) -> None:
        import sys

        mock_wandb = sys.modules["wandb"]
        mock_run = MagicMock()
        mock_run.id = "auto_generated_id"
        mock_wandb.init.return_value = mock_run

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        backend = WandbBackend(project="test", run_dir=run_dir)

        assert _load_wandb_id(run_dir) == "auto_generated_id"
        assert backend.run_id == "auto_generated_id"
        backend.close()

    @patch.dict("sys.modules", {"wandb": MagicMock()})
    def test_no_id_file_without_run_dir(self, tmp_path: Path) -> None:
        import sys

        mock_wandb = sys.modules["wandb"]
        mock_run = MagicMock()
        mock_run.id = "some_id"
        mock_wandb.init.return_value = mock_run

        backend = WandbBackend(project="test")
        # No run_dir → no file written; just verify no crash
        backend.close()


# ---------------------------------------------------------------------------
# Console logging
# ---------------------------------------------------------------------------


class TestSetupLogging:
    def test_setup_creates_handler(self) -> None:
        setup_logging()
        logger = logging.getLogger("vibe_rl")
        assert len(logger.handlers) == 1
        assert logger.level == logging.INFO
        # Cleanup
        logger.handlers.clear()

    def test_setup_idempotent(self) -> None:
        setup_logging()
        setup_logging()
        logger = logging.getLogger("vibe_rl")
        assert len(logger.handlers) == 1
        logger.handlers.clear()

    def test_custom_level(self) -> None:
        setup_logging(level=logging.DEBUG)
        logger = logging.getLogger("vibe_rl")
        assert logger.level == logging.DEBUG
        logger.handlers.clear()

    def test_no_propagation(self) -> None:
        setup_logging()
        logger = logging.getLogger("vibe_rl")
        assert logger.propagate is False
        logger.handlers.clear()

    def test_formatter_output(self) -> None:
        setup_logging()
        logger = logging.getLogger("vibe_rl")
        handler = logger.handlers[0]
        record = logging.LogRecord(
            name="vibe_rl",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="hello world",
            args=(),
            exc_info=None,
        )
        formatted = handler.format(record)
        # Should start with abbreviated level "I"
        assert formatted.startswith("I ")
        assert "[vibe_rl]" in formatted
        assert "hello world" in formatted
        logger.handlers.clear()

    def test_formatter_levels(self) -> None:
        setup_logging()
        logger = logging.getLogger("vibe_rl")
        handler = logger.handlers[0]
        for level, abbrev in [
            (logging.DEBUG, "D"),
            (logging.INFO, "I"),
            (logging.WARNING, "W"),
            (logging.ERROR, "E"),
            (logging.CRITICAL, "C"),
        ]:
            record = logging.LogRecord(
                name="vibe_rl",
                level=level,
                pathname="",
                lineno=0,
                msg="test",
                args=(),
                exc_info=None,
            )
            formatted = handler.format(record)
            assert formatted.startswith(f"{abbrev} "), f"Expected '{abbrev}' prefix for level {level}"
        logger.handlers.clear()


class TestLogStepProgress:
    def test_basic_progress(self, caplog: object) -> None:
        setup_logging()
        with patch.object(logging.getLogger("vibe_rl"), "info") as mock_info:
            log_step_progress(500, 10000)
            mock_info.assert_called_once()
            msg = mock_info.call_args[0][0]
            assert "500/10000" in msg
            assert "5.0%" in msg
        logging.getLogger("vibe_rl").handlers.clear()

    def test_progress_with_metrics(self) -> None:
        setup_logging()
        with patch.object(logging.getLogger("vibe_rl"), "info") as mock_info:
            log_step_progress(1000, 10000, {"loss": 0.42, "reward": 195.0})
            msg = mock_info.call_args[0][0]
            assert "10.0%" in msg
            assert "loss=0.42" in msg
            assert "reward=195" in msg
        logging.getLogger("vibe_rl").handlers.clear()

    def test_skips_step_and_wall_time_in_metrics(self) -> None:
        setup_logging()
        with patch.object(logging.getLogger("vibe_rl"), "info") as mock_info:
            log_step_progress(100, 1000, {"step": 100, "wall_time": 1.0, "loss": 0.5})
            msg = mock_info.call_args[0][0]
            assert "loss=0.5" in msg
            assert "step=" not in msg.split("|")[-1]  # step not in metrics part
            assert "wall_time=" not in msg
        logging.getLogger("vibe_rl").handlers.clear()

    def test_zero_total_steps(self) -> None:
        setup_logging()
        with patch.object(logging.getLogger("vibe_rl"), "info") as mock_info:
            log_step_progress(0, 0)
            msg = mock_info.call_args[0][0]
            assert "0.0%" in msg
        logging.getLogger("vibe_rl").handlers.clear()
