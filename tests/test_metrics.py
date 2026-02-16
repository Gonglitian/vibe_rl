"""Tests for vibe_rl.metrics."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from vibe_rl.metrics import MetricsLogger, read_metrics


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
