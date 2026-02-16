"""Tests for vibe_rl.video."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vibe_rl.video import VideoRecorder, write_video


@pytest.fixture
def dummy_frames() -> list[np.ndarray]:
    """Generate a short sequence of random RGB frames."""
    rng = np.random.default_rng(42)
    return [rng.integers(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(10)]


class TestVideoRecorder:
    def test_bad_frame_shape_raises(self, tmp_path: Path) -> None:
        rec = VideoRecorder(tmp_path / "test.mp4")
        with pytest.raises(ValueError, match="Expected.*H, W, 3"):
            rec.add_frame(np.zeros((64, 64), dtype=np.uint8))  # missing channel
        rec.close()

    def test_bad_frame_shape_4_channels(self, tmp_path: Path) -> None:
        rec = VideoRecorder(tmp_path / "test.mp4")
        with pytest.raises(ValueError, match="Expected.*H, W, 3"):
            rec.add_frame(np.zeros((64, 64, 4), dtype=np.uint8))  # RGBA
        rec.close()

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "deep" / "nested" / "video.mp4"
        rec = VideoRecorder(path)
        assert path.parent.is_dir()
        rec.close()

    def test_frame_count_starts_zero(self, tmp_path: Path) -> None:
        rec = VideoRecorder(tmp_path / "test.mp4")
        assert rec.frame_count == 0
        rec.close()

    def test_context_manager(self, tmp_path: Path) -> None:
        path = tmp_path / "test.mp4"
        with VideoRecorder(path) as rec:
            assert rec.path == path

    def test_repr(self, tmp_path: Path) -> None:
        rec = VideoRecorder(tmp_path / "test.mp4")
        assert "VideoRecorder" in repr(rec)
        rec.close()


class TestWriteVideo:
    def test_bad_shape_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Expected"):
            write_video(
                tmp_path / "test.mp4",
                np.zeros((10,), dtype=np.uint8),  # 1D
            )


# Integration tests that require imageio[ffmpeg] are skipped if not installed.
try:
    import imageio  # noqa: F401

    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


@pytest.mark.skipif(not HAS_IMAGEIO, reason="imageio[ffmpeg] not installed")
class TestVideoIntegration:
    def test_write_and_read_video(
        self, tmp_path: Path, dummy_frames: list[np.ndarray]
    ) -> None:
        path = tmp_path / "test.mp4"
        write_video(path, dummy_frames, fps=10)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_recorder_produces_file(
        self, tmp_path: Path, dummy_frames: list[np.ndarray]
    ) -> None:
        path = tmp_path / "rec.mp4"
        with VideoRecorder(path, fps=10) as rec:
            for f in dummy_frames:
                rec.add_frame(f)

        assert path.exists()
        assert rec.frame_count == len(dummy_frames)

    def test_write_video_from_4d_array(self, tmp_path: Path) -> None:
        frames = np.random.randint(0, 255, (5, 32, 32, 3), dtype=np.uint8)
        path = write_video(tmp_path / "batch.mp4", frames, fps=5)
        assert path.exists()
