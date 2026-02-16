"""Offline dataset loading for RL and imitation learning.

This module provides a ``Dataset`` protocol and concrete adapters for
loading robotics demonstration data (e.g. LeRobot format) into
vibe-rl ``Transition`` batches suitable for JAX training loops.

Requires the ``[data]`` optional dependency group::

    pip install 'vibe-rl[data]'
"""

from vibe_rl.data.data_loader import JaxDataLoader
from vibe_rl.data.dataset import Dataset
from vibe_rl.data.lerobot_dataset import LeRobotDatasetAdapter, LeRobotDatasetConfig

__all__ = [
    "Dataset",
    "JaxDataLoader",
    "LeRobotDatasetAdapter",
    "LeRobotDatasetConfig",
]
