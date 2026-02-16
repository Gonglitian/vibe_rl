"""Tests for vibe_rl.data module.

Tests are structured to work without lerobot/torch installed (protocol
and collate tests) and to skip gracefully when those deps are missing.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vibe_rl.data.dataset import Dataset
from vibe_rl.dataprotocol.transition import Transition


def _has_real_torch() -> bool:
    """Check if real PyTorch (not a namespace stub) is available."""
    try:
        import torch.utils.data  # noqa: F401

        return True
    except (ImportError, AttributeError):
        return False


requires_torch = pytest.mark.skipif(
    not _has_real_torch(), reason="PyTorch not installed"
)


class _DummyDataset:
    """Minimal dataset for protocol conformance tests."""

    def __init__(self, n: int = 10, obs_dim: int = 4, act_dim: int = 2):
        self._n = n
        self._obs = np.random.randn(n, obs_dim).astype(np.float32)
        self._act = np.random.randn(n, act_dim).astype(np.float32)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int) -> Transition:
        is_last = index == self._n - 1
        next_idx = index if is_last else index + 1
        return Transition(
            obs=jnp.asarray(self._obs[index]),
            action=jnp.asarray(self._act[index]),
            reward=jnp.zeros(()),
            next_obs=jnp.asarray(self._obs[next_idx]),
            done=jnp.array(is_last),
        )


def test_dataset_protocol_conformance():
    ds = _DummyDataset()
    assert isinstance(ds, Dataset)


def test_dataset_protocol_rejects_non_conformant():
    class _Bad:
        pass

    assert not isinstance(_Bad(), Dataset)


# ── Collate function ──────────────────────────────────────────────────


def test_transition_collate_fn():
    from vibe_rl.data.data_loader import _transition_collate_fn

    samples = [
        Transition(
            obs=np.array([1.0, 2.0]),
            action=np.array([0.5]),
            reward=np.array(1.0),
            next_obs=np.array([3.0, 4.0]),
            done=np.array(False),
        )
        for _ in range(4)
    ]
    batch = _transition_collate_fn(samples)

    assert isinstance(batch, Transition)
    assert batch.obs.shape == (4, 2)
    assert batch.action.shape == (4, 1)
    assert batch.reward.shape == (4,)
    assert batch.done.shape == (4,)
    assert isinstance(batch.obs, np.ndarray)


def test_numpy_to_jax():
    from vibe_rl.data.data_loader import _numpy_to_jax

    np_batch = Transition(
        obs=np.ones((8, 3), dtype=np.float32),
        action=np.ones((8, 2), dtype=np.float32),
        reward=np.ones(8, dtype=np.float32),
        next_obs=np.ones((8, 3), dtype=np.float32),
        done=np.zeros(8, dtype=bool),
    )
    jax_batch = _numpy_to_jax(np_batch)

    assert isinstance(jax_batch, Transition)
    assert isinstance(jax_batch.obs, jax.Array)
    assert jax_batch.obs.shape == (8, 3)
    assert jax_batch.reward.shape == (8,)


# ── Shard batch ───────────────────────────────────────────────────────


def test_shard_batch_single_device():
    from vibe_rl.data.data_loader import _shard_batch

    devices = jax.devices()[:1]
    batch = Transition(
        obs=jnp.ones((4, 3)),
        action=jnp.ones((4, 2)),
        reward=jnp.ones(4),
        next_obs=jnp.ones((4, 3)),
        done=jnp.zeros(4, dtype=jnp.bool_),
    )
    sharded = _shard_batch(batch, devices)
    assert sharded.obs.shape == (1, 4, 3)
    assert sharded.reward.shape == (1, 4)


def test_shard_batch_rejects_indivisible():
    from vibe_rl.data.data_loader import _shard_batch

    # Simulate 2 devices when batch size is 3 (not divisible)
    devices = jax.devices()[:1] * 2  # duplicate single device for the test

    batch = Transition(
        obs=jnp.ones((3, 2)),
        action=jnp.ones((3,)),
        reward=jnp.ones(3),
        next_obs=jnp.ones((3, 2)),
        done=jnp.zeros(3, dtype=jnp.bool_),
    )
    with pytest.raises(ValueError, match="not divisible"):
        _shard_batch(batch, devices)


# ── pack_obs ──────────────────────────────────────────────────────────


def test_pack_obs_single_key():
    from vibe_rl.data.lerobot_dataset import _pack_obs

    item = {"observation.state": np.array([1.0, 2.0, 3.0])}
    obs = _pack_obs(item, ["observation.state"])
    assert isinstance(obs, jax.Array)
    np.testing.assert_allclose(obs, [1.0, 2.0, 3.0])


def test_pack_obs_multi_key_concat():
    from vibe_rl.data.lerobot_dataset import _pack_obs

    item = {
        "observation.state": np.array([1.0, 2.0]),
        "observation.gripper": np.array([3.0]),
    }
    obs = _pack_obs(item, ["observation.state", "observation.gripper"])
    assert obs.shape == (3,)
    np.testing.assert_allclose(obs, [1.0, 2.0, 3.0])


# ── LeRobotDatasetAdapter with mocked lerobot ────────────────────────


def _make_mock_lerobot_dataset(
    n_frames: int = 20,
    n_episodes: int = 2,
    state_dim: int = 4,
    action_dim: int = 2,
    fps: int = 10,
):
    """Create a mock that quacks like LeRobotDataset."""
    frames_per_ep = n_frames // n_episodes

    episode_indices = []
    frame_indices = []
    for ep in range(n_episodes):
        for f in range(frames_per_ep):
            episode_indices.append(ep)
            frame_indices.append(f)

    hf_data = {
        "episode_index": episode_indices,
        "frame_index": frame_indices,
    }

    rng = np.random.RandomState(42)
    states = rng.randn(n_frames, state_dim).astype(np.float32)
    actions = rng.randn(n_frames, action_dim).astype(np.float32)

    mock_ds = MagicMock()
    mock_ds.__len__ = MagicMock(return_value=n_frames)

    def getitem(idx):
        # Return numpy arrays (mimicking torch tensors with .numpy())
        class _FakeTensor:
            """Minimal mock of a torch Tensor with .numpy() support."""

            def __init__(self, arr):
                self._arr = np.asarray(arr)

            def numpy(self):
                return self._arr

            def item(self):
                return self._arr.item()

        return {
            "index": _FakeTensor(idx),
            "episode_index": _FakeTensor(episode_indices[idx]),
            "frame_index": _FakeTensor(frame_indices[idx]),
            "timestamp": _FakeTensor(frame_indices[idx] / fps),
            "observation.state": _FakeTensor(states[idx]),
            "action": _FakeTensor(actions[idx]),
        }

    mock_ds.__getitem__ = MagicMock(side_effect=getitem)
    mock_ds.hf_dataset = hf_data

    mock_ds.meta = MagicMock()
    mock_ds.meta.fps = fps
    mock_ds.meta.features = {
        "observation.state": {"dtype": "float32", "shape": (state_dim,)},
        "action": {"dtype": "float32", "shape": (action_dim,)},
    }
    mock_ds.meta.total_episodes = n_episodes
    mock_ds.meta.total_frames = n_frames

    return mock_ds, states, actions, episode_indices, frame_indices


def _make_adapter(**kwargs):
    from vibe_rl.data.lerobot_dataset import (
        LeRobotDatasetAdapter,
        LeRobotDatasetConfig,
    )

    mock_ds, states, actions, ep_idx, fr_idx = _make_mock_lerobot_dataset(
        **kwargs
    )

    config = LeRobotDatasetConfig(
        repo_id="test/mock",
        reward_key=None,
    )

    # Patch _require_lerobot to return a class that, when called, returns mock_ds
    with patch(
        "vibe_rl.data.lerobot_dataset._require_lerobot"
    ) as mock_require:
        mock_cls = MagicMock(return_value=mock_ds)
        mock_require.return_value = mock_cls
        adapter = LeRobotDatasetAdapter(config)

    return adapter, states, actions, ep_idx, fr_idx


class TestLeRobotDatasetAdapter:
    """Tests using a mocked LeRobot backend."""

    def test_len(self):
        adapter, *_ = _make_adapter(n_frames=20)
        assert len(adapter) == 20

    def test_getitem_returns_transition(self):
        adapter, *_ = _make_adapter()
        t = adapter[0]
        assert isinstance(t, Transition)
        assert isinstance(t.obs, jax.Array)
        assert isinstance(t.action, jax.Array)
        assert isinstance(t.reward, jax.Array)
        assert isinstance(t.done, jax.Array)

    def test_getitem_shapes(self):
        adapter, *_ = _make_adapter(state_dim=6, action_dim=3)
        t = adapter[0]
        assert t.obs.shape == (6,)
        assert t.action.shape == (3,)
        assert t.reward.shape == ()
        assert t.next_obs.shape == (6,)
        assert t.done.shape == ()

    def test_getitem_obs_values(self):
        adapter, states, _, _, _ = _make_adapter()
        t = adapter[0]
        np.testing.assert_allclose(t.obs, states[0], atol=1e-6)

    def test_getitem_action_values(self):
        adapter, _, actions, _, _ = _make_adapter()
        t = adapter[0]
        np.testing.assert_allclose(t.action, actions[0], atol=1e-6)

    def test_next_obs_within_episode(self):
        adapter, states, _, ep_idx, _ = _make_adapter(
            n_frames=20, n_episodes=2
        )
        # Index 0 is first frame of ep 0 -> next_obs should be states[1]
        t = adapter[0]
        np.testing.assert_allclose(t.next_obs, states[1], atol=1e-6)
        assert not t.done

    def test_done_at_episode_boundary(self):
        adapter, states, _, ep_idx, fr_idx = _make_adapter(
            n_frames=20, n_episodes=2
        )
        # Last frame of first episode is index 9 (10 frames per ep)
        t = adapter[9]
        assert t.done
        # next_obs should equal obs for terminal states
        np.testing.assert_allclose(t.next_obs, t.obs, atol=1e-6)

    def test_reward_defaults_to_zero(self):
        adapter, *_ = _make_adapter()
        t = adapter[0]
        assert float(t.reward) == 0.0

    def test_metadata(self):
        adapter, *_ = _make_adapter(n_frames=20, n_episodes=2)
        meta = adapter.metadata
        assert meta["repo_id"] == "test/mock"
        assert meta["fps"] == 10
        assert meta["total_episodes"] == 2
        assert meta["total_frames"] == 20

    def test_string_constructor(self):
        """Test that passing a string repo_id works."""
        from vibe_rl.data.lerobot_dataset import LeRobotDatasetAdapter

        mock_ds, *_ = _make_mock_lerobot_dataset()

        with patch(
            "vibe_rl.data.lerobot_dataset._require_lerobot"
        ) as mock_require:
            mock_cls = MagicMock(return_value=mock_ds)
            mock_require.return_value = mock_cls
            adapter = LeRobotDatasetAdapter("test/string_repo")

        assert adapter.config.repo_id == "test/string_repo"
        assert len(adapter) == 20


# ── JaxDataLoader with DummyDataset ──────────────────────────────────


@requires_torch
class TestJaxDataLoader:
    """Test the DataLoader using the _DummyDataset (needs real torch)."""

    def test_basic_iteration(self):
        from vibe_rl.data.data_loader import JaxDataLoader

        ds = _DummyDataset(n=16, obs_dim=3, act_dim=2)
        loader = JaxDataLoader(
            ds, batch_size=4, num_workers=0, shuffle=False, drop_last=True
        )
        batches = list(loader)
        assert len(batches) == 4

        b = batches[0]
        assert isinstance(b, Transition)
        assert isinstance(b.obs, jax.Array)
        assert b.obs.shape == (4, 3)
        assert b.action.shape == (4, 2)

    def test_len(self):
        from vibe_rl.data.data_loader import JaxDataLoader

        ds = _DummyDataset(n=16, obs_dim=3, act_dim=2)
        loader = JaxDataLoader(
            ds, batch_size=4, num_workers=0, drop_last=True
        )
        assert len(loader) == 4

    def test_drop_last_false(self):
        from vibe_rl.data.data_loader import JaxDataLoader

        ds = _DummyDataset(n=10, obs_dim=3, act_dim=2)
        loader = JaxDataLoader(
            ds, batch_size=4, num_workers=0, shuffle=False, drop_last=False
        )
        batches = list(loader)
        # 10 / 4 = 2 full + 1 partial = 3 batches
        assert len(batches) == 3
        assert batches[-1].obs.shape[0] == 2

    def test_sharding_single_device(self):
        from vibe_rl.data.data_loader import JaxDataLoader

        devices = jax.devices()[:1]
        ds = _DummyDataset(n=8, obs_dim=3, act_dim=2)
        loader = JaxDataLoader(
            ds,
            batch_size=4,
            num_workers=0,
            shuffle=False,
            drop_last=True,
            devices=devices,
        )
        batches = list(loader)
        assert len(batches) == 2
        # With 1 device, shape is (1, 4, 3)
        assert batches[0].obs.shape == (1, 4, 3)

    def test_multiple_epochs(self):
        from vibe_rl.data.data_loader import JaxDataLoader

        ds = _DummyDataset(n=8, obs_dim=3, act_dim=2)
        loader = JaxDataLoader(
            ds, batch_size=4, num_workers=0, shuffle=False, drop_last=True
        )
        for _epoch in range(3):
            batches = list(loader)
            assert len(batches) == 2
