"""LeRobot dataset adapter for vibe-rl.

Wraps ``lerobot.datasets.lerobot_dataset.LeRobotDataset`` (a torch
Dataset) and adapts each sample to the vibe-rl ``Transition`` format.

LeRobot datasets store robotics demonstrations with keys like:
  - ``observation.state``          (float32 state vector)
  - ``observation.images.<cam>``   (float32 image tensor, C×H×W)
  - ``action``                     (float32 action vector)

This adapter maps them to ``Transition(obs, action, reward, next_obs, done)``.

Observation handling
--------------------
- **State-only**: ``obs`` = ``observation.state`` vector.
- **Image-only**: ``obs`` = single camera image (first camera found).
- **Mixed**: ``obs`` = dict-style pytree ``{"state": ..., "image": ...}``.

The ``obs_keys`` parameter gives full control over which LeRobot keys
to include and how to pack them.

Delta timestamps (action horizons)
-----------------------------------
Pass ``delta_timestamps={"action": [0/fps, 1/fps, ..., (H-1)/fps]}``
to get stacked action chunks of shape ``(H, action_dim)`` in the
``action`` field.  See LeRobot docs for details.

Usage::

    from vibe_rl.data import LeRobotDatasetAdapter

    ds = LeRobotDatasetAdapter("lerobot/aloha_sim_insertion_human")
    t = ds[0]  # Transition with jax arrays
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array

from vibe_rl.dataprotocol.transition import Transition


def _require_lerobot():
    """Import and return the LeRobotDataset class, with a clear error."""
    try:
        from lerobot.datasets.lerobot_dataset import (  # type: ignore[import-untyped]
            LeRobotDataset,
        )

        return LeRobotDataset
    except (ImportError, ModuleNotFoundError) as e:
        raise ImportError(
            "LeRobot is required for dataset loading. "
            "Install with: pip install 'vibe-rl[data]'"
        ) from e


@dataclass(frozen=True)
class LeRobotDatasetConfig:
    """Configuration for loading a LeRobot dataset.

    Attributes:
        repo_id: HuggingFace repo id (e.g. ``"lerobot/aloha_sim_insertion_human"``).
        root: Local cache directory. ``None`` uses default HF cache.
        episodes: Filter to specific episode indices.
        delta_timestamps: Dict mapping feature keys to lists of float
            time offsets for stacking (e.g. action horizons).
        obs_keys: LeRobot feature keys to include in ``obs``. When a
            single key is given, ``obs`` is the raw array. When multiple
            keys are given, ``obs`` is a dict mapping short names to
            arrays. Default: ``["observation.state"]``.
        action_key: LeRobot feature key for the action. Default: ``"action"``.
        reward_key: LeRobot feature key for reward. Default: ``"next.reward"``.
            Set to ``None`` if the dataset has no reward.
        image_transforms: Optional torchvision transforms for images.
        revision: HuggingFace dataset revision / branch.
    """

    repo_id: str
    root: str | None = None
    episodes: list[int] | None = None
    delta_timestamps: dict[str, list[float]] | None = None
    obs_keys: list[str] = field(default_factory=lambda: ["observation.state"])
    action_key: str = "action"
    reward_key: str | None = "next.reward"
    image_transforms: Any = None
    revision: str | None = None


def _to_numpy(x: Any) -> np.ndarray:
    """Convert a torch Tensor or any array-like to numpy."""
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def _pack_obs(item: dict, obs_keys: list[str]) -> Array:
    """Extract and pack observation from a LeRobot sample dict.

    If a single key is requested, return the array directly.
    If multiple keys are requested, concatenate all 1-D arrays into a
    single flat vector. Image keys (3-D or 4-D) are not supported in
    multi-key concat mode — use a single image key or state-only.
    """
    if len(obs_keys) == 1:
        return jnp.asarray(_to_numpy(item[obs_keys[0]]))

    # Multiple keys: concatenate flat vectors
    arrays = []
    for key in obs_keys:
        arr = _to_numpy(item[key])
        if arr.ndim > 1:
            # Flatten anything that's not already 1-D (e.g. stacked states)
            arr = arr.reshape(-1)
        arrays.append(arr)
    return jnp.asarray(np.concatenate(arrays, axis=0))


class LeRobotDatasetAdapter:
    """Adapts a LeRobot dataset to the vibe-rl ``Dataset`` protocol.

    Each ``__getitem__`` call returns a ``Transition`` with jax arrays.

    For ``next_obs``, the adapter looks up the next frame in the same
    episode. If the current frame is the last in its episode,
    ``next_obs = obs`` and ``done = True``.

    Parameters:
        config: A ``LeRobotDatasetConfig``, **or** a string ``repo_id``
            for quick construction with defaults.
    """

    def __init__(self, config: LeRobotDatasetConfig | str) -> None:
        if isinstance(config, str):
            config = LeRobotDatasetConfig(repo_id=config)
        self.config = config

        LeRobotDataset = _require_lerobot()
        self._dataset = LeRobotDataset(
            repo_id=config.repo_id,
            root=config.root,
            episodes=config.episodes,
            delta_timestamps=config.delta_timestamps,
            image_transforms=config.image_transforms,
            revision=config.revision,
        )

        # Build episode boundary lookup for next_obs / done computation.
        # _episode_ends[ep] = index of last frame (inclusive) in episode ep.
        self._build_episode_index()

    def _build_episode_index(self) -> None:
        """Pre-compute episode boundaries for fast next_obs lookup."""
        hf = self._dataset.hf_dataset
        ep_indices = np.array(hf["episode_index"])
        frame_indices = np.array(hf["frame_index"])

        # For each sample, check if it's the last frame in its episode
        # by comparing frame_index with max frame_index per episode.
        unique_eps = np.unique(ep_indices)
        ep_max_frame: dict[int, int] = {}
        for ep in unique_eps:
            mask = ep_indices == ep
            ep_max_frame[int(ep)] = int(frame_indices[mask].max())

        self._ep_indices = ep_indices
        self._frame_indices = frame_indices
        self._ep_max_frame = ep_max_frame

    @property
    def lerobot_dataset(self):
        """Access the underlying LeRobotDataset."""
        return self._dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Transition:
        item = self._dataset[index]

        obs = _pack_obs(item, self.config.obs_keys)
        action_np = _to_numpy(item[self.config.action_key])
        action = jnp.asarray(action_np)

        # Reward: default to 0 if not present in dataset
        if (
            self.config.reward_key is not None
            and self.config.reward_key in item
        ):
            reward = jnp.asarray(_to_numpy(item[self.config.reward_key]))
        else:
            reward = jnp.zeros((), dtype=jnp.float32)

        # Done + next_obs: check if this is the last frame in the episode
        ep_idx = int(self._ep_indices[index])
        frame_idx = int(self._frame_indices[index])
        is_last = frame_idx >= self._ep_max_frame[ep_idx]

        done = jnp.array(is_last, dtype=jnp.bool_)

        if is_last:
            next_obs = obs
        else:
            next_item = self._dataset[index + 1]
            next_obs = _pack_obs(next_item, self.config.obs_keys)

        return Transition(
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            done=done,
        )

    @property
    def metadata(self) -> dict[str, Any]:
        """Return dataset metadata (fps, features, stats, etc.)."""
        meta = self._dataset.meta
        return {
            "repo_id": self.config.repo_id,
            "fps": meta.fps,
            "features": dict(meta.features),
            "total_episodes": meta.total_episodes,
            "total_frames": meta.total_frames,
        }
