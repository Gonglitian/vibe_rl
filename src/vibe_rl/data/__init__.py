"""Offline dataset loading for RL and imitation learning.

This module provides a ``Dataset`` protocol and concrete adapters for
loading robotics demonstration data (e.g. LeRobot format) into
vibe-rl ``Transition`` batches suitable for JAX training loops.

Also includes a composable transform pipeline and normalization
statistics utilities for data preprocessing.

Requires the ``[data]`` optional dependency group::

    pip install 'vibe-rl[data]'
"""

from vibe_rl.data.data_loader import JaxDataLoader
from vibe_rl.data.dataset import Dataset
from vibe_rl.data.lerobot_dataset import LeRobotDatasetAdapter, LeRobotDatasetConfig
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
from vibe_rl.data.transforms import (
    LambdaTransform,
    Normalize,
    Pad,
    Resize,
    Tokenize,
    Transform,
    TransformGroup,
)

__all__ = [
    "Dataset",
    "JaxDataLoader",
    "LambdaTransform",
    "LeRobotDatasetAdapter",
    "LeRobotDatasetConfig",
    "Normalize",
    "NormStats",
    "Pad",
    "Resize",
    "Tokenize",
    "Transform",
    "TransformGroup",
    "compute_norm_stats",
    "load_norm_stats",
    "quantile_normalize",
    "quantile_unnormalize",
    "save_norm_stats",
    "z_score_normalize",
    "z_score_unnormalize",
]
