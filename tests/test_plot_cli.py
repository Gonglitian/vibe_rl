"""Tests for the plot_rewards CLI and train.py auto-plot integration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from scripts.plot_rewards import PlotArgs, main as plot_main  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_metrics(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


def _make_run(
    base: Path,
    name: str,
    n_steps: int = 50,
    algo: str | None = None,
    seed: int = 0,
) -> Path:
    """Create a synthetic run directory with JSONL metrics."""
    run_dir = base / name
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)
    records = []
    reward = 0.0
    for step in range(0, n_steps * 100, 100):
        reward += rng.randn() * 5 + 1
        records.append({"step": step, "episode_return": round(reward, 2)})
    _write_metrics(run_dir / "logs" / "metrics.jsonl", records)

    if algo:
        (run_dir / "config.json").write_text(json.dumps({"algorithm": algo}))

    return run_dir


# ---------------------------------------------------------------------------
# plot_rewards CLI tests
# ---------------------------------------------------------------------------


class TestPlotRewardsCLI:
    def test_plot_single_run(self, tmp_path: Path) -> None:
        run = _make_run(tmp_path, "ppo_run_001")
        output = tmp_path / "out.png"

        plot_main(PlotArgs(run_dirs=[str(run)], output=str(output)))

        assert output.exists()
        assert output.stat().st_size > 0

    def test_plot_multiple_runs(self, tmp_path: Path) -> None:
        r0 = _make_run(tmp_path, "ppo_s0", algo="PPO", seed=0)
        r1 = _make_run(tmp_path, "ppo_s1", algo="PPO", seed=1)
        output = tmp_path / "multi.png"

        plot_main(PlotArgs(
            run_dirs=[str(r0), str(r1)],
            output=str(output),
            compare=True,
        ))

        assert output.exists()

    def test_experiment_name_discovery(self, tmp_path: Path) -> None:
        runs_base = tmp_path / "runs"
        _make_run(runs_base, "cartpole_ppo_20260101_000000", seed=0)
        _make_run(runs_base, "cartpole_ppo_20260102_000000", seed=1)
        _make_run(runs_base, "pendulum_sac_20260101_000000", seed=2)

        output = tmp_path / "discovered.png"
        plot_main(PlotArgs(
            experiment_name="cartpole_ppo",
            base_dir=str(runs_base),
            output=str(output),
        ))

        assert output.exists()

    def test_default_output_in_artifacts(self, tmp_path: Path) -> None:
        run = _make_run(tmp_path, "ppo_run_002")

        plot_main(PlotArgs(run_dirs=[str(run)]))

        expected = run / "artifacts" / "reward_curve.png"
        assert expected.exists()

    def test_custom_y_key(self, tmp_path: Path) -> None:
        run = tmp_path / "custom_run"
        (run / "logs").mkdir(parents=True)
        (run / "artifacts").mkdir(parents=True)
        records = [{"step": i, "total_loss": float(i) * 0.01} for i in range(50)]
        _write_metrics(run / "logs" / "metrics.jsonl", records)

        output = tmp_path / "loss.png"
        plot_main(PlotArgs(
            run_dirs=[str(run)],
            y_key="total_loss",
            output=str(output),
        ))

        assert output.exists()

    def test_compare_mode_groups_algos(self, tmp_path: Path) -> None:
        ppo_run = _make_run(tmp_path, "ppo", algo="PPO", seed=0)
        dqn_run = _make_run(tmp_path, "dqn", algo="DQN", seed=1)
        output = tmp_path / "compare.png"

        plot_main(PlotArgs(
            run_dirs=[str(ppo_run), str(dqn_run)],
            compare=True,
            output=str(output),
        ))

        assert output.exists()

    def test_no_runs_exits(self) -> None:
        with pytest.raises(SystemExit):
            plot_main(PlotArgs())

    def test_smooth_and_style_overrides(self, tmp_path: Path) -> None:
        run = _make_run(tmp_path, "styled_run")
        output = tmp_path / "styled.png"

        plot_main(PlotArgs(
            run_dirs=[str(run)],
            smooth_radius=5,
            smooth_mode="ema",
            shaded="none",
            output=str(output),
        ))

        assert output.exists()


# ---------------------------------------------------------------------------
# train.py auto-plot integration
# ---------------------------------------------------------------------------


class TestTrainAutoPlot:
    def test_auto_plot_generates_image(self, tmp_path: Path) -> None:
        """Simulate what train.py does after training finishes."""
        from vibe_rl.plotting import plot_reward_curve
        import matplotlib.pyplot as plt

        run = _make_run(tmp_path, "auto_plot_run")
        save_path = run / "artifacts" / "reward_curve.png"

        fig = plot_reward_curve(run, save_path=save_path)
        plt.close(fig)

        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_plot_flag_in_runner_config(self) -> None:
        """RunnerConfig should have a plot field defaulting to True."""
        from vibe_rl.runner.config import RunnerConfig

        cfg = RunnerConfig()
        assert cfg.plot is True

        cfg_off = RunnerConfig(plot=False)
        assert cfg_off.plot is False
