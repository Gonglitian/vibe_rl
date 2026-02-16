"""Normalization statistics computation and application.

Computes per-feature statistics (mean, std, quantiles) over a dataset and
provides ``z_score_normalize`` / ``quantile_normalize`` functions that map
raw values into a standardized range.

Statistics are stored as a ``NormStats`` frozen dataclass and can be
serialized to / loaded from JSON for reproducibility.

Usage::

    from vibe_rl.data.normalize import compute_norm_stats, z_score_normalize

    stats = compute_norm_stats(dataset, keys=["obs", "action"])
    normalized_obs = z_score_normalize(obs, stats["obs"])

    # Persist
    from vibe_rl.data.normalize import save_norm_stats, load_norm_stats
    save_norm_stats(stats, "norm_stats.json")
    stats = load_norm_stats("norm_stats.json")
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class NormStats:
    """Per-feature normalization statistics.

    Attributes:
        mean: Mean of each feature.  Shape ``(D,)`` or scalar.
        std: Standard deviation of each feature.
        q01: 1st percentile (used for quantile normalization lower bound).
        q99: 99th percentile (used for quantile normalization upper bound).
    """

    mean: np.ndarray
    std: np.ndarray
    q01: np.ndarray
    q99: np.ndarray

    def to_dict(self) -> dict[str, list]:
        """Serialize to a JSON-compatible dict."""
        return {
            "mean": np.asarray(self.mean).tolist(),
            "std": np.asarray(self.std).tolist(),
            "q01": np.asarray(self.q01).tolist(),
            "q99": np.asarray(self.q99).tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NormStats:
        """Deserialize from a dict (e.g. loaded from JSON)."""
        return cls(
            mean=np.asarray(d["mean"], dtype=np.float32),
            std=np.asarray(d["std"], dtype=np.float32),
            q01=np.asarray(d["q01"], dtype=np.float32),
            q99=np.asarray(d["q99"], dtype=np.float32),
        )


# ---------------------------------------------------------------------------
# Compute statistics
# ---------------------------------------------------------------------------


def compute_norm_stats(
    dataset,
    keys: list[str],
    *,
    max_samples: int | None = None,
) -> dict[str, NormStats]:
    """Compute normalization statistics by iterating over a dataset.

    The dataset must support ``__len__`` and ``__getitem__`` returning a
    dict-like object (or a ``NamedTuple`` with ``_asdict``).

    Parameters:
        dataset: A dataset object (``__len__`` + ``__getitem__``).
        keys: Which keys to compute statistics for.
        max_samples: Cap the number of samples used (``None`` = all).

    Returns:
        A dict mapping each key to its ``NormStats``.
    """
    n = len(dataset)
    if max_samples is not None:
        n = min(n, max_samples)

    # Accumulate values per key
    accumulators: dict[str, list[np.ndarray]] = {k: [] for k in keys}

    for i in range(n):
        sample = dataset[i]
        # Support both dict and NamedTuple
        if hasattr(sample, "_asdict"):
            sample = sample._asdict()

        for key in keys:
            if key not in sample:
                continue
            arr = np.asarray(sample[key], dtype=np.float32)
            # Flatten to 1-D for per-feature accumulation
            accumulators[key].append(arr.reshape(-1) if arr.ndim == 0 else arr)

    stats: dict[str, NormStats] = {}
    for key in keys:
        if not accumulators[key]:
            continue
        stacked = np.stack(accumulators[key], axis=0)  # (N, D)
        stats[key] = NormStats(
            mean=stacked.mean(axis=0),
            std=stacked.std(axis=0),
            q01=np.percentile(stacked, 1, axis=0).astype(np.float32),
            q99=np.percentile(stacked, 99, axis=0).astype(np.float32),
        )

    return stats


# ---------------------------------------------------------------------------
# Normalize / Unnormalize
# ---------------------------------------------------------------------------


def z_score_normalize(
    x: np.ndarray, stats: NormStats, *, eps: float = 1e-8
) -> np.ndarray:
    """Z-score normalization: ``(x - mean) / max(std, eps)``."""
    x = np.asarray(x, dtype=np.float32)
    return (x - stats.mean) / np.maximum(stats.std, eps)


def z_score_unnormalize(
    x: np.ndarray, stats: NormStats, *, eps: float = 1e-8
) -> np.ndarray:
    """Inverse of ``z_score_normalize``."""
    x = np.asarray(x, dtype=np.float32)
    return x * np.maximum(stats.std, eps) + stats.mean


def quantile_normalize(
    x: np.ndarray, stats: NormStats, *, eps: float = 1e-8
) -> np.ndarray:
    """Quantile normalization: maps ``[q01, q99]`` to ``[-1, 1]``.

    Values outside the ``[q01, q99]`` range are *not* clipped, so
    outliers are preserved but the bulk of the distribution is in ``[-1, 1]``.
    """
    x = np.asarray(x, dtype=np.float32)
    q_range = stats.q99 - stats.q01
    safe_range = np.maximum(q_range, eps)
    return 2.0 * (x - stats.q01) / safe_range - 1.0


def quantile_unnormalize(
    x: np.ndarray, stats: NormStats, *, eps: float = 1e-8
) -> np.ndarray:
    """Inverse of ``quantile_normalize``."""
    x = np.asarray(x, dtype=np.float32)
    q_range = stats.q99 - stats.q01
    safe_range = np.maximum(q_range, eps)
    return (x + 1.0) / 2.0 * safe_range + stats.q01


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------


def save_norm_stats(
    stats: dict[str, NormStats], path: str | pathlib.Path
) -> None:
    """Save normalization statistics to a JSON file.

    Parameters:
        stats: A dict mapping key names to ``NormStats``.
        path: Output file path.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v.to_dict() for k, v in stats.items()}
    path.write_text(json.dumps(payload, indent=2))


def load_norm_stats(path: str | pathlib.Path) -> dict[str, NormStats]:
    """Load normalization statistics from a JSON file.

    Parameters:
        path: Path to a JSON file previously written by ``save_norm_stats``.

    Returns:
        A dict mapping key names to ``NormStats``.
    """
    path = pathlib.Path(path)
    payload = json.loads(path.read_text())
    return {k: NormStats.from_dict(v) for k, v in payload.items()}
