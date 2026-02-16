"""Tests for vibe_rl.plotting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vibe_rl.plotting.colors import (
    DEEPMIND,
    color_for,
    get_colors,
    reset_palette,
    set_palette,
)
from vibe_rl.plotting.config import PlotConfig
from vibe_rl.plotting.plot import (
    _detect_algo_name,
    _extract_series,
    _interpolate_to_common_x,
    plot_reward_curve,
    smooth_ema,
    smooth_window,
)

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # Non-interactive backend for CI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_metrics(path: Path, records: list[dict]) -> Path:
    """Write a list of dicts as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


def _make_run_dir(
    base: Path,
    name: str,
    n_steps: int = 100,
    algo: str | None = None,
    seed: int = 0,
) -> Path:
    """Create a fake run directory with synthetic reward data."""
    run_dir = base / name
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)
    records = []
    reward = 0.0
    for step in range(0, n_steps * 100, 100):
        reward += rng.randn() * 5 + 1  # upward trend + noise
        records.append({"step": step, "episode_return": round(reward, 2)})
    _write_metrics(run_dir / "logs" / "metrics.jsonl", records)

    if algo:
        (run_dir / "config.json").write_text(json.dumps({"algorithm": algo}))

    return run_dir


# ---------------------------------------------------------------------------
# Smoothing tests
# ---------------------------------------------------------------------------


class TestSmoothWindow:
    def test_identity_when_radius_zero(self) -> None:
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = smooth_window(y, radius=0)
        np.testing.assert_array_equal(result, y)

    def test_uniform_array_unchanged(self) -> None:
        y = np.ones(20) * 5.0
        result = smooth_window(y, radius=3)
        np.testing.assert_allclose(result, y, atol=1e-10)

    def test_reduces_noise(self) -> None:
        rng = np.random.RandomState(42)
        trend = np.linspace(0, 10, 200)
        noisy = trend + rng.randn(200) * 2
        smoothed = smooth_window(noisy, radius=10)
        # Smoothed should be closer to the true trend.
        assert np.std(smoothed - trend) < np.std(noisy - trend)

    def test_output_length_matches_input(self) -> None:
        y = np.random.randn(50)
        result = smooth_window(y, radius=5)
        assert len(result) == len(y)

    def test_negative_radius_no_op(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        result = smooth_window(y, radius=-1)
        np.testing.assert_array_equal(result, y)


class TestSmoothEMA:
    def test_identity_when_span_one(self) -> None:
        y = np.array([1.0, 3.0, 5.0])
        result = smooth_ema(y, span=1)
        np.testing.assert_array_equal(result, y)

    def test_first_element_unchanged(self) -> None:
        y = np.array([10.0, 20.0, 30.0])
        result = smooth_ema(y, span=5)
        assert result[0] == 10.0

    def test_reduces_noise(self) -> None:
        rng = np.random.RandomState(0)
        trend = np.linspace(0, 10, 200)
        noisy = trend + rng.randn(200) * 2
        smoothed = smooth_ema(noisy, span=20)
        assert np.std(smoothed - trend) < np.std(noisy - trend)

    def test_output_length_matches_input(self) -> None:
        y = np.random.randn(50)
        result = smooth_ema(y, span=10)
        assert len(result) == len(y)


# ---------------------------------------------------------------------------
# Color palette tests
# ---------------------------------------------------------------------------


class TestColors:
    def test_deepmind_has_ten_colors(self) -> None:
        assert len(DEEPMIND) == 10

    def test_color_for_cycles(self) -> None:
        assert color_for(0) == color_for(len(get_colors()))

    def test_set_and_reset_palette(self) -> None:
        original = get_colors()
        custom = ["#FF0000", "#00FF00"]
        set_palette(custom)
        assert get_colors() == custom
        reset_palette()
        assert get_colors() == original

    def test_get_colors_returns_copy(self) -> None:
        colors = get_colors()
        colors.append("#000000")
        assert len(get_colors()) == len(DEEPMIND)


# ---------------------------------------------------------------------------
# PlotConfig tests
# ---------------------------------------------------------------------------


class TestPlotConfig:
    def test_defaults(self) -> None:
        cfg = PlotConfig()
        assert cfg.smooth_radius == 10
        assert cfg.smooth_mode == "window"
        assert cfg.shaded == "std"
        assert cfg.figsize == (8, 6)
        assert cfg.dpi == 150
        assert cfg.save_format == "png"

    def test_frozen(self) -> None:
        cfg = PlotConfig()
        with pytest.raises(AttributeError):
            cfg.smooth_radius = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Data extraction tests
# ---------------------------------------------------------------------------


class TestExtractSeries:
    def test_basic(self) -> None:
        records = [
            {"step": 0, "episode_return": 1.0},
            {"step": 100, "episode_return": 2.0},
        ]
        x, y = _extract_series(records, "step", "episode_return")
        np.testing.assert_array_equal(x, [0, 100])
        np.testing.assert_array_equal(y, [1.0, 2.0])

    def test_missing_keys_skipped(self) -> None:
        records = [
            {"step": 0, "episode_return": 1.0},
            {"step": 100},  # missing y
            {"episode_return": 3.0},  # missing x
            {"step": 200, "episode_return": 4.0},
        ]
        x, y = _extract_series(records, "step", "episode_return")
        np.testing.assert_array_equal(x, [0, 200])
        np.testing.assert_array_equal(y, [1.0, 4.0])

    def test_empty_records(self) -> None:
        x, y = _extract_series([], "step", "episode_return")
        assert len(x) == 0
        assert len(y) == 0


# ---------------------------------------------------------------------------
# Algorithm detection tests
# ---------------------------------------------------------------------------


class TestDetectAlgoName:
    def test_from_config(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "my_run"
        run_dir.mkdir()
        (run_dir / "config.json").write_text(json.dumps({"algorithm": "PPO"}))
        assert _detect_algo_name(run_dir) == "PPO"

    def test_fallback_to_dirname(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "dqn_cartpole_20260215_143022"
        run_dir.mkdir()
        assert _detect_algo_name(run_dir) == "dqn_cartpole"

    def test_simple_dirname(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "experiment_x"
        run_dir.mkdir()
        assert _detect_algo_name(run_dir) == "experiment_x"


# ---------------------------------------------------------------------------
# Interpolation tests
# ---------------------------------------------------------------------------


class TestInterpolation:
    def test_identical_x_grids(self) -> None:
        x = np.arange(10, dtype=np.float64)
        y1 = np.ones(10)
        y2 = np.ones(10) * 2
        common_x, y_mat = _interpolate_to_common_x([x, x], [y1, y2])
        assert y_mat.shape == (2, 10)
        np.testing.assert_allclose(y_mat[0], 1.0)
        np.testing.assert_allclose(y_mat[1], 2.0)

    def test_different_x_ranges(self) -> None:
        x1 = np.array([0.0, 5.0, 10.0])
        x2 = np.array([2.0, 5.0, 8.0])
        y1 = np.array([0.0, 5.0, 10.0])
        y2 = np.array([0.0, 5.0, 10.0])
        common_x, y_mat = _interpolate_to_common_x([x1, x2], [y1, y2])
        # Common range should be [2, 8]
        assert common_x[0] == pytest.approx(2.0)
        assert common_x[-1] == pytest.approx(8.0)
        assert y_mat.shape[0] == 2


# ---------------------------------------------------------------------------
# End-to-end plot tests
# ---------------------------------------------------------------------------


class TestPlotRewardCurve:
    def test_single_run(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path, "ppo_seed0")
        fig = plot_reward_curve(run_dir)
        assert fig is not None

    def test_multi_seed_shading(self, tmp_path: Path) -> None:
        rd0 = _make_run_dir(tmp_path, "ppo_s0", algo="PPO", seed=0)
        rd1 = _make_run_dir(tmp_path, "ppo_s1", algo="PPO", seed=1)
        rd2 = _make_run_dir(tmp_path, "ppo_s2", algo="PPO", seed=2)
        fig = plot_reward_curve([rd0, rd1, rd2])
        ax = fig.axes[0]
        # Should have one line (mean) + one PolyCollection (shading)
        assert len(ax.lines) == 1
        assert len(ax.collections) >= 1

    def test_multi_algo_comparison(self, tmp_path: Path) -> None:
        rd_ppo = _make_run_dir(tmp_path, "ppo_run", algo="PPO", seed=0)
        rd_dqn = _make_run_dir(tmp_path, "dqn_run", algo="DQN", seed=1)
        fig = plot_reward_curve([rd_ppo, rd_dqn])
        ax = fig.axes[0]
        legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "PPO" in legend_labels
        assert "DQN" in legend_labels

    def test_save_png(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path, "run0")
        save_path = tmp_path / "reward.png"
        plot_reward_curve(run_dir, save_path=save_path)
        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_save_pdf(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path, "run0")
        save_path = tmp_path / "reward.pdf"
        plot_reward_curve(run_dir, save_path=save_path)
        assert save_path.exists()

    def test_save_svg(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path, "run0")
        save_path = tmp_path / "reward.svg"
        plot_reward_curve(run_dir, save_path=save_path)
        assert save_path.exists()

    def test_custom_keys(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "custom_run"
        (run_dir / "logs").mkdir(parents=True)
        records = [
            {"timestep": i, "reward": float(i) * 0.1} for i in range(50)
        ]
        _write_metrics(run_dir / "logs" / "metrics.jsonl", records)
        fig = plot_reward_curve(
            run_dir, x_key="timestep", y_key="reward"
        )
        assert fig is not None

    def test_smooth_override(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path, "run_smooth")
        fig = plot_reward_curve(run_dir, smooth=0)
        assert fig is not None

    def test_stderr_shading(self, tmp_path: Path) -> None:
        rd0 = _make_run_dir(tmp_path, "s0", algo="A", seed=0)
        rd1 = _make_run_dir(tmp_path, "s1", algo="A", seed=1)
        cfg = PlotConfig(shaded="stderr")
        fig = plot_reward_curve([rd0, rd1], config=cfg)
        ax = fig.axes[0]
        assert len(ax.collections) >= 1

    def test_no_shading(self, tmp_path: Path) -> None:
        rd0 = _make_run_dir(tmp_path, "s0", algo="A", seed=0)
        rd1 = _make_run_dir(tmp_path, "s1", algo="A", seed=1)
        cfg = PlotConfig(shaded="none")
        fig = plot_reward_curve([rd0, rd1], config=cfg)
        ax = fig.axes[0]
        assert len(ax.collections) == 0

    def test_ema_smoothing(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path, "ema_run")
        cfg = PlotConfig(smooth_mode="ema", smooth_radius=20)
        fig = plot_reward_curve(run_dir, config=cfg)
        assert fig is not None

    def test_save_to_artifacts(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path, "run_artifacts")
        save_path = run_dir / "artifacts" / "reward_curve.png"
        plot_reward_curve(run_dir, save_path=save_path)
        assert save_path.exists()

    def test_direct_jsonl_path(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "metrics.jsonl"
        records = [{"step": i, "episode_return": float(i)} for i in range(20)]
        _write_metrics(jsonl, records)
        fig = plot_reward_curve(jsonl)
        assert fig is not None

    def test_empty_run_no_crash(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "empty_run"
        (run_dir / "logs").mkdir(parents=True)
        _write_metrics(run_dir / "logs" / "metrics.jsonl", [])
        fig = plot_reward_curve(run_dir)
        assert fig is not None
