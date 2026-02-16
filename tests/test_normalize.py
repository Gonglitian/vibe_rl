"""Tests for vibe_rl.data.normalize module."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest

from vibe_rl.data.normalize import (
    NormStats,
    compute_norm_stats,
    load_norm_stats,
    quantile_normalize,
    quantile_unnormalize,
    save_norm_stats,
    z_score_normalize,
    z_score_unnormalize,
)

# ── NormStats ─────────────────────────────────────────────────────────


class TestNormStats:
    def test_frozen(self):
        stats = NormStats(
            mean=np.array([0.0]),
            std=np.array([1.0]),
            q01=np.array([-2.0]),
            q99=np.array([2.0]),
        )
        with pytest.raises(AttributeError):
            stats.mean = np.array([1.0])  # type: ignore[misc]

    def test_to_dict_roundtrip(self):
        stats = NormStats(
            mean=np.array([1.0, 2.0]),
            std=np.array([0.5, 1.5]),
            q01=np.array([-1.0, -0.5]),
            q99=np.array([3.0, 4.5]),
        )
        d = stats.to_dict()
        restored = NormStats.from_dict(d)
        np.testing.assert_allclose(restored.mean, stats.mean)
        np.testing.assert_allclose(restored.std, stats.std)
        np.testing.assert_allclose(restored.q01, stats.q01)
        np.testing.assert_allclose(restored.q99, stats.q99)

    def test_from_dict_types(self):
        d = {
            "mean": [1.0, 2.0],
            "std": [0.5, 1.5],
            "q01": [-1.0, -0.5],
            "q99": [3.0, 4.5],
        }
        stats = NormStats.from_dict(d)
        assert stats.mean.dtype == np.float32
        assert stats.std.dtype == np.float32

    def test_scalar_stats(self):
        stats = NormStats(
            mean=np.float32(0.0),
            std=np.float32(1.0),
            q01=np.float32(-2.0),
            q99=np.float32(2.0),
        )
        d = stats.to_dict()
        assert isinstance(d["mean"], float)


# ── z-score normalize ─────────────────────────────────────────────────


class TestZScoreNormalize:
    def test_basic(self):
        stats = NormStats(
            mean=np.array([2.0, 4.0]),
            std=np.array([1.0, 2.0]),
            q01=np.zeros(2),
            q99=np.zeros(2),
        )
        x = np.array([3.0, 6.0])
        result = z_score_normalize(x, stats)
        np.testing.assert_allclose(result, [1.0, 1.0])

    def test_zero_std(self):
        stats = NormStats(
            mean=np.array([0.0]),
            std=np.array([0.0]),
            q01=np.zeros(1),
            q99=np.zeros(1),
        )
        x = np.array([5.0])
        result = z_score_normalize(x, stats, eps=1.0)
        np.testing.assert_allclose(result, [5.0])

    def test_roundtrip(self):
        stats = NormStats(
            mean=np.array([1.0, 2.0, 3.0]),
            std=np.array([0.5, 1.0, 2.0]),
            q01=np.zeros(3),
            q99=np.zeros(3),
        )
        x = np.array([1.5, 3.0, 5.0])
        normalized = z_score_normalize(x, stats)
        recovered = z_score_unnormalize(normalized, stats)
        np.testing.assert_allclose(recovered, x, atol=1e-6)

    def test_batched(self):
        stats = NormStats(
            mean=np.array([0.0, 0.0]),
            std=np.array([2.0, 4.0]),
            q01=np.zeros(2),
            q99=np.zeros(2),
        )
        x = np.array([[2.0, 4.0], [4.0, 8.0]])
        result = z_score_normalize(x, stats)
        np.testing.assert_allclose(result, [[1.0, 1.0], [2.0, 2.0]])


# ── quantile normalize ───────────────────────────────────────────────


class TestQuantileNormalize:
    def test_maps_q01_to_minus1(self):
        stats = NormStats(
            mean=np.zeros(1),
            std=np.ones(1),
            q01=np.array([10.0]),
            q99=np.array([20.0]),
        )
        result = quantile_normalize(np.array([10.0]), stats)
        np.testing.assert_allclose(result, [-1.0])

    def test_maps_q99_to_plus1(self):
        stats = NormStats(
            mean=np.zeros(1),
            std=np.ones(1),
            q01=np.array([10.0]),
            q99=np.array([20.0]),
        )
        result = quantile_normalize(np.array([20.0]), stats)
        np.testing.assert_allclose(result, [1.0])

    def test_maps_midpoint_to_zero(self):
        stats = NormStats(
            mean=np.zeros(1),
            std=np.ones(1),
            q01=np.array([0.0]),
            q99=np.array([10.0]),
        )
        result = quantile_normalize(np.array([5.0]), stats)
        np.testing.assert_allclose(result, [0.0])

    def test_outliers_not_clipped(self):
        stats = NormStats(
            mean=np.zeros(1),
            std=np.ones(1),
            q01=np.array([0.0]),
            q99=np.array([10.0]),
        )
        result = quantile_normalize(np.array([15.0]), stats)
        assert result[0] > 1.0  # beyond +1

    def test_zero_range(self):
        stats = NormStats(
            mean=np.zeros(1),
            std=np.ones(1),
            q01=np.array([5.0]),
            q99=np.array([5.0]),
        )
        # When q01 == q99, eps prevents division by zero
        result = quantile_normalize(np.array([5.0]), stats, eps=1.0)
        assert np.isfinite(result).all()

    def test_roundtrip(self):
        stats = NormStats(
            mean=np.zeros(2),
            std=np.ones(2),
            q01=np.array([0.0, 10.0]),
            q99=np.array([100.0, 50.0]),
        )
        x = np.array([25.0, 30.0])
        normalized = quantile_normalize(x, stats)
        recovered = quantile_unnormalize(normalized, stats)
        np.testing.assert_allclose(recovered, x, atol=1e-5)

    def test_multi_dim(self):
        stats = NormStats(
            mean=np.zeros(3),
            std=np.ones(3),
            q01=np.array([0.0, 0.0, 0.0]),
            q99=np.array([10.0, 20.0, 30.0]),
        )
        x = np.array([5.0, 10.0, 15.0])
        result = quantile_normalize(x, stats)
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0])


# ── compute_norm_stats ────────────────────────────────────────────────


class _DictDataset:
    """Minimal dict-returning dataset for testing."""

    def __init__(self, data: dict[str, np.ndarray]):
        self._data = data
        self._n = len(next(iter(data.values())))

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        return {k: v[index] for k, v in self._data.items()}


class _Transition(NamedTuple):
    obs: np.ndarray
    action: np.ndarray


class _NamedTupleDataset:
    """NamedTuple-returning dataset for testing _asdict fallback."""

    def __init__(self, n: int = 10, obs_dim: int = 3, act_dim: int = 2):
        rng = np.random.RandomState(42)
        self._obs = rng.randn(n, obs_dim).astype(np.float32)
        self._act = rng.randn(n, act_dim).astype(np.float32)
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int) -> _Transition:
        return _Transition(obs=self._obs[index], action=self._act[index])


class TestComputeNormStats:
    def test_basic(self):
        rng = np.random.RandomState(0)
        data = {
            "obs": rng.randn(100, 4).astype(np.float32),
            "action": rng.randn(100, 2).astype(np.float32),
        }
        ds = _DictDataset(data)
        stats = compute_norm_stats(ds, keys=["obs", "action"])

        assert "obs" in stats
        assert "action" in stats
        assert stats["obs"].mean.shape == (4,)
        assert stats["action"].std.shape == (2,)

    def test_stats_values(self):
        data = {
            "x": np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]]),
        }
        ds = _DictDataset(data)
        stats = compute_norm_stats(ds, keys=["x"])

        np.testing.assert_allclose(stats["x"].mean, [0.0, 10.0])
        np.testing.assert_allclose(stats["x"].std, [0.0, 0.0])

    def test_quantiles(self):
        rng = np.random.RandomState(42)
        values = rng.randn(1000, 1).astype(np.float32)
        ds = _DictDataset({"x": values})
        stats = compute_norm_stats(ds, keys=["x"])

        # q01 should be roughly -2.33, q99 roughly +2.33 for standard normal
        assert stats["x"].q01[0] < -1.5
        assert stats["x"].q99[0] > 1.5
        assert stats["x"].q01[0] < stats["x"].q99[0]

    def test_max_samples(self):
        data = {"x": np.arange(100, dtype=np.float32).reshape(100, 1)}
        ds = _DictDataset(data)
        stats = compute_norm_stats(ds, keys=["x"], max_samples=10)

        # Mean should be based on first 10 samples only: 0..9, mean=4.5
        np.testing.assert_allclose(stats["x"].mean, [4.5])

    def test_named_tuple_dataset(self):
        ds = _NamedTupleDataset(n=50, obs_dim=3, act_dim=2)
        stats = compute_norm_stats(ds, keys=["obs", "action"])

        assert stats["obs"].mean.shape == (3,)
        assert stats["action"].mean.shape == (2,)

    def test_missing_key_skipped(self):
        data = {"x": np.ones((10, 2), dtype=np.float32)}
        ds = _DictDataset(data)
        stats = compute_norm_stats(ds, keys=["x", "missing"])

        assert "x" in stats
        assert "missing" not in stats


# ── save / load ───────────────────────────────────────────────────────


class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        stats = {
            "obs": NormStats(
                mean=np.array([1.0, 2.0]),
                std=np.array([0.5, 1.5]),
                q01=np.array([-1.0, -0.5]),
                q99=np.array([3.0, 4.5]),
            ),
            "action": NormStats(
                mean=np.array([0.0]),
                std=np.array([1.0]),
                q01=np.array([-2.0]),
                q99=np.array([2.0]),
            ),
        }
        path = tmp_path / "stats.json"
        save_norm_stats(stats, path)

        loaded = load_norm_stats(path)
        assert set(loaded.keys()) == {"obs", "action"}
        np.testing.assert_allclose(loaded["obs"].mean, stats["obs"].mean)
        np.testing.assert_allclose(loaded["obs"].std, stats["obs"].std)
        np.testing.assert_allclose(loaded["obs"].q01, stats["obs"].q01)
        np.testing.assert_allclose(loaded["obs"].q99, stats["obs"].q99)
        np.testing.assert_allclose(loaded["action"].mean, stats["action"].mean)

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "stats.json"
        stats = {
            "x": NormStats(
                mean=np.zeros(1),
                std=np.ones(1),
                q01=np.zeros(1),
                q99=np.ones(1),
            )
        }
        save_norm_stats(stats, path)
        assert path.exists()

    def test_json_is_readable(self, tmp_path):
        """The JSON file should be human-readable."""
        import json

        stats = {
            "x": NormStats(
                mean=np.array([1.0]),
                std=np.array([2.0]),
                q01=np.array([0.0]),
                q99=np.array([3.0]),
            )
        }
        path = tmp_path / "stats.json"
        save_norm_stats(stats, path)

        data = json.loads(path.read_text())
        assert "x" in data
        assert data["x"]["mean"] == [1.0]
        assert data["x"]["std"] == [2.0]


# ── Integration: compute then normalize ──────────────────────────────


class TestIntegration:
    def test_compute_then_z_score(self):
        rng = np.random.RandomState(0)
        data = {"obs": rng.randn(200, 3).astype(np.float32) * 5 + 10}
        ds = _DictDataset(data)
        stats = compute_norm_stats(ds, keys=["obs"])

        # Normalize all samples and check distribution
        normalized = np.stack(
            [z_score_normalize(ds[i]["obs"], stats["obs"]) for i in range(len(ds))]
        )
        np.testing.assert_allclose(normalized.mean(axis=0), 0.0, atol=0.1)
        np.testing.assert_allclose(normalized.std(axis=0), 1.0, atol=0.1)

    def test_compute_then_quantile(self):
        rng = np.random.RandomState(0)
        data = {"obs": rng.randn(500, 2).astype(np.float32) * 3 + 5}
        ds = _DictDataset(data)
        stats = compute_norm_stats(ds, keys=["obs"])

        normalized = np.stack(
            [quantile_normalize(ds[i]["obs"], stats["obs"]) for i in range(len(ds))]
        )
        # Most values should be in [-1, 1] (by definition, 98% of them)
        in_range = np.logical_and(normalized >= -1.0, normalized <= 1.0)
        fraction_in_range = in_range.mean()
        assert fraction_in_range > 0.95

    def test_compute_save_load_normalize(self, tmp_path):
        rng = np.random.RandomState(42)
        data = {"obs": rng.randn(100, 4).astype(np.float32)}
        ds = _DictDataset(data)

        stats = compute_norm_stats(ds, keys=["obs"])
        path = tmp_path / "stats.json"
        save_norm_stats(stats, path)
        loaded = load_norm_stats(path)

        x = ds[0]["obs"]
        result_orig = z_score_normalize(x, stats["obs"])
        result_loaded = z_score_normalize(x, loaded["obs"])
        np.testing.assert_allclose(result_orig, result_loaded, atol=1e-6)
