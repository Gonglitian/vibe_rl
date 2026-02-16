"""Video recording utilities for evaluation rollouts.

Records sequences of RGB frames into mp4 files. Works with any source
of ``(H, W, 3) uint8`` numpy arrays — JAX environments that produce
pixel observations, Gymnasium ``render()`` calls, or custom renderers.

Requires ``imageio[ffmpeg]`` at runtime (not a core dependency).

Usage::

    from vibe_rl.video import VideoRecorder

    recorder = VideoRecorder(run.video_path(step=10000), fps=30)
    for frame in rollout_frames:
        recorder.add_frame(frame)
    recorder.close()

    # Or as a context manager:
    with VideoRecorder(path, fps=30) as rec:
        for frame in frames:
            rec.add_frame(frame)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class VideoRecorder:
    """Write RGB frames to an mp4 file.

    Parameters
    ----------
    path:
        Output file path (should end in ``.mp4``).
    fps:
        Frames per second for the output video.
    """

    def __init__(self, path: str | Path, *, fps: int = 30) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fps = fps
        self._writer: object | None = None
        self._frame_count = 0

    def _ensure_writer(self) -> object:
        if self._writer is None:
            try:
                import imageio.v3 as iio  # noqa: F811
            except ImportError as exc:
                raise ImportError(
                    "Video recording requires imageio[ffmpeg]. "
                    "Install with: pip install 'imageio[ffmpeg]'"
                ) from exc

            self._iio = iio
            # imageio v3 uses imopen for streaming writes
            self._writer = iio.imopen(
                str(self._path),
                "w",
                plugin="pyav",
            )
        return self._writer

    def add_frame(self, frame: NDArray[np.uint8]) -> None:
        """Append a single ``(H, W, 3) uint8`` RGB frame."""
        frame = np.asarray(frame, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"Expected (H, W, 3) uint8 array, got shape {frame.shape}"
            )
        writer = self._ensure_writer()
        writer.write(frame, fps=self._fps)
        self._frame_count += 1

    def close(self) -> Path:
        """Finalize the video file. Returns the output path."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        return self._path

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def path(self) -> Path:
        return self._path

    def __enter__(self) -> VideoRecorder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"VideoRecorder({self._path}, frames={self._frame_count})"


def write_video(
    path: str | Path,
    frames: list[NDArray[np.uint8]] | NDArray[np.uint8],
    *,
    fps: int = 30,
) -> Path:
    """One-shot helper: write a list of frames to an mp4 file.

    Parameters
    ----------
    path:
        Output path (should end in ``.mp4``).
    frames:
        Sequence of ``(H, W, 3) uint8`` arrays, or a single
        ``(T, H, W, 3)`` array.
    fps:
        Frames per second.

    Returns
    -------
    Path to the written file.
    """
    frames_arr = np.asarray(frames, dtype=np.uint8)
    if frames_arr.ndim == 4:
        frame_list = [frames_arr[i] for i in range(frames_arr.shape[0])]
    elif frames_arr.ndim == 3:
        frame_list = [frames_arr]
    else:
        raise ValueError(
            f"Expected (T, H, W, 3) or (H, W, 3) array, got shape {frames_arr.shape}"
        )

    with VideoRecorder(path, fps=fps) as rec:
        for frame in frame_list:
            rec.add_frame(frame)

    return rec.path
