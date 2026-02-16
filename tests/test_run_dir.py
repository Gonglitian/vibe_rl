"""Tests for vibe_rl.run_dir."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from vibe_rl.run_dir import RunDir, find_runs


class TestRunDir:
    """Core RunDir functionality."""

    def test_creates_standard_subdirs(self, tmp_path: Path) -> None:
        run = RunDir("exp", base_dir=tmp_path)
        assert run.checkpoints.is_dir()
        assert run.logs.is_dir()
        assert run.videos.is_dir()
        assert run.artifacts.is_dir()

    def test_dirname_contains_experiment_name(self, tmp_path: Path) -> None:
        run = RunDir("my_experiment", base_dir=tmp_path)
        assert run.root.name.startswith("my_experiment_")

    def test_explicit_run_id(self, tmp_path: Path) -> None:
        run = RunDir(base_dir=tmp_path, run_id="fixed_name")
        assert run.root.name == "fixed_name"
        assert run.root.parent == tmp_path

    def test_checkpoint_dir_creates_step_subdir(self, tmp_path: Path) -> None:
        run = RunDir(base_dir=tmp_path, run_id="r1")
        ckpt = run.checkpoint_dir(10000)
        assert ckpt == run.checkpoints / "step_10000"
        assert ckpt.is_dir()

    def test_log_path(self, tmp_path: Path) -> None:
        run = RunDir(base_dir=tmp_path, run_id="r1")
        assert run.log_path() == run.logs / "metrics.jsonl"
        assert run.log_path("custom.csv") == run.logs / "custom.csv"

    def test_video_path(self, tmp_path: Path) -> None:
        run = RunDir(base_dir=tmp_path, run_id="r1")
        assert run.video_path(step=5000) == run.videos / "eval_step_5000.mp4"
        expected = run.videos / "train_step_5000.mp4"
        assert run.video_path(step=5000, prefix="train") == expected

    def test_artifact_path(self, tmp_path: Path) -> None:
        run = RunDir(base_dir=tmp_path, run_id="r1")
        assert run.artifact_path("report.json") == run.artifacts / "report.json"

    def test_str_and_repr(self, tmp_path: Path) -> None:
        run = RunDir(base_dir=tmp_path, run_id="r1")
        assert str(run) == str(run.root)
        assert "RunDir" in repr(run)

    def test_fspath(self, tmp_path: Path) -> None:
        run = RunDir(base_dir=tmp_path, run_id="r1")
        # os.fspath should work
        import os

        assert os.fspath(run) == str(run.root)


class TestConfigSnapshot:
    """Config save/load round-trip."""

    def test_save_and_load_dict(self, tmp_path: Path) -> None:
        run = RunDir(base_dir=tmp_path, run_id="r1")
        cfg = {"lr": 1e-3, "gamma": 0.99, "hidden_sizes": [128, 128]}
        run.save_config(cfg)

        loaded = run.load_config()
        assert loaded["lr"] == 1e-3
        assert loaded["gamma"] == 0.99
        assert loaded["hidden_sizes"] == [128, 128]

    def test_save_frozen_dataclass(self, tmp_path: Path) -> None:
        @dataclass(frozen=True)
        class MyConfig:
            lr: float = 1e-3
            hidden_sizes: tuple[int, ...] = (64, 64)

        run = RunDir(base_dir=tmp_path, run_id="r1")
        run.save_config(MyConfig())

        loaded = run.load_config()
        assert loaded["lr"] == 1e-3
        assert loaded["hidden_sizes"] == [64, 64]  # tuple → list in JSON

    def test_save_nested_dataclass(self, tmp_path: Path) -> None:
        @dataclass(frozen=True)
        class Inner:
            x: int = 1

        @dataclass(frozen=True)
        class Outer:
            inner: Inner = Inner()
            name: str = "test"

        run = RunDir(base_dir=tmp_path, run_id="r1")
        run.save_config(Outer())

        loaded = run.load_config()
        assert loaded["inner"]["x"] == 1
        assert loaded["name"] == "test"

    def test_save_custom_filename(self, tmp_path: Path) -> None:
        run = RunDir(base_dir=tmp_path, run_id="r1")
        path = run.save_config({"a": 1}, filename="custom.json")
        assert path.name == "custom.json"
        assert json.loads(path.read_text()) == {"a": 1}

    def test_config_is_valid_json(self, tmp_path: Path) -> None:
        run = RunDir(base_dir=tmp_path, run_id="r1")
        run.save_config({"key": "value"})
        # Should parse without error
        raw = (run.root / "config.json").read_text()
        json.loads(raw)


class TestBestCheckpoint:
    """Best-checkpoint symlink management."""

    def test_mark_best_creates_symlink(self, tmp_path: Path) -> None:
        run = RunDir(base_dir=tmp_path, run_id="r1")
        run.checkpoint_dir(5000)
        link = run.mark_best(5000)

        assert link.is_symlink()
        assert link.resolve() == (run.checkpoints / "step_5000").resolve()

    def test_mark_best_updates_symlink(self, tmp_path: Path) -> None:
        run = RunDir(base_dir=tmp_path, run_id="r1")
        run.checkpoint_dir(5000)
        run.checkpoint_dir(10000)

        run.mark_best(5000)
        run.mark_best(10000)

        link = run.checkpoints / "best"
        # Should point to the newer one
        assert link.resolve() == (run.checkpoints / "step_10000").resolve()

    def test_best_checkpoint_property(self, tmp_path: Path) -> None:
        run = RunDir(base_dir=tmp_path, run_id="r1")
        assert run.best_checkpoint is None

        run.checkpoint_dir(5000)
        run.mark_best(5000)
        assert run.best_checkpoint is not None


class TestCheckpointCleanup:
    """Checkpoint retention strategy."""

    def _make_checkpoints(self, run: RunDir, steps: list[int]) -> None:
        for step in steps:
            d = run.checkpoint_dir(step)
            (d / "data.bin").write_bytes(b"x")  # non-empty so it's real

    def test_cleanup_keeps_latest(self, tmp_path: Path) -> None:
        run = RunDir(base_dir=tmp_path, run_id="r1")
        self._make_checkpoints(run, [1000, 2000, 3000, 4000, 5000])

        removed = run.cleanup_checkpoints(keep=2)
        assert len(removed) == 3

        remaining = run.list_checkpoints()
        assert [s for s, _ in remaining] == [4000, 5000]

    def test_cleanup_preserves_best(self, tmp_path: Path) -> None:
        run = RunDir(base_dir=tmp_path, run_id="r1")
        self._make_checkpoints(run, [1000, 2000, 3000, 4000, 5000])
        run.mark_best(1000)  # oldest but marked best

        run.cleanup_checkpoints(keep=2)

        remaining_steps = [s for s, _ in run.list_checkpoints()]
        # 1000 preserved (best), plus 4000 and 5000 (latest 2)
        assert 1000 in remaining_steps
        assert 4000 in remaining_steps
        assert 5000 in remaining_steps

    def test_cleanup_noop_when_fewer_than_keep(self, tmp_path: Path) -> None:
        run = RunDir(base_dir=tmp_path, run_id="r1")
        self._make_checkpoints(run, [1000, 2000])

        removed = run.cleanup_checkpoints(keep=5)
        assert len(removed) == 0
        assert len(run.list_checkpoints()) == 2

    def test_cleanup_rejects_keep_zero(self, tmp_path: Path) -> None:
        run = RunDir(base_dir=tmp_path, run_id="r1")
        with pytest.raises(ValueError, match="keep must be >= 1"):
            run.cleanup_checkpoints(keep=0)


class TestListCheckpoints:
    """Checkpoint discovery."""

    def test_list_empty(self, tmp_path: Path) -> None:
        run = RunDir(base_dir=tmp_path, run_id="r1")
        assert run.list_checkpoints() == []

    def test_list_sorted(self, tmp_path: Path) -> None:
        run = RunDir(base_dir=tmp_path, run_id="r1")
        for step in [30000, 10000, 20000]:
            run.checkpoint_dir(step)

        ckpts = run.list_checkpoints()
        assert [s for s, _ in ckpts] == [10000, 20000, 30000]

    def test_latest_checkpoint(self, tmp_path: Path) -> None:
        run = RunDir(base_dir=tmp_path, run_id="r1")
        assert run.latest_checkpoint is None

        run.checkpoint_dir(5000)
        run.checkpoint_dir(10000)
        assert run.latest_checkpoint == run.checkpoints / "step_10000"

    def test_ignores_non_step_dirs(self, tmp_path: Path) -> None:
        run = RunDir(base_dir=tmp_path, run_id="r1")
        run.checkpoint_dir(1000)
        (run.checkpoints / "best").mkdir()  # not a step_ dir
        (run.checkpoints / "temp_file.txt").write_text("x")

        ckpts = run.list_checkpoints()
        assert len(ckpts) == 1
        assert ckpts[0][0] == 1000


class TestFindRuns:
    """Run discovery from base_dir."""

    def test_find_all_runs(self, tmp_path: Path) -> None:
        RunDir("exp_a", base_dir=tmp_path, run_id="exp_a_20260101")
        RunDir("exp_b", base_dir=tmp_path, run_id="exp_b_20260102")

        runs = find_runs(base_dir=tmp_path)
        assert len(runs) == 2
        names = [r.root.name for r in runs]
        assert "exp_a_20260101" in names

    def test_filter_by_experiment_name(self, tmp_path: Path) -> None:
        RunDir(base_dir=tmp_path, run_id="dqn_20260101")
        RunDir(base_dir=tmp_path, run_id="dqn_20260102")
        RunDir(base_dir=tmp_path, run_id="ppo_20260101")

        runs = find_runs(base_dir=tmp_path, experiment_name="dqn")
        assert len(runs) == 2
        assert all("dqn_" in r.root.name for r in runs)

    def test_find_empty_dir(self, tmp_path: Path) -> None:
        assert find_runs(base_dir=tmp_path) == []

    def test_find_nonexistent_dir(self, tmp_path: Path) -> None:
        assert find_runs(base_dir=tmp_path / "nope") == []
