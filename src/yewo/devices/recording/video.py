"""Read the metadata of a captured screen recording."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Protocol

from imageio_ffmpeg import read_frames

from yewo.devices.recording.errors import ScreenRecordingError


MASK_LEAD_SECONDS = 0.25


@dataclass(frozen=True)
class ScreenRecordingVideoMetadata:
    width_px: int
    height_px: int
    frames_per_second: float
    duration_seconds: float


class _CommandRunner(Protocol):
    def __call__(
        self,
        command: Sequence[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: float,
        check: bool,
    ) -> subprocess.CompletedProcess[str]: ...


def _run_command(
    command: Sequence[str],
    *,
    capture_output: bool,
    text: bool,
    timeout: float,
    check: bool,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=capture_output,
        text=text,
        timeout=timeout,
        check=check,
    )


def read_screen_recording_metadata(path: Path) -> ScreenRecordingVideoMetadata:
    """Read validated geometry, frame rate, and duration from one MP4."""
    if not _is_mp4(path):
        raise ScreenRecordingError("The screen recording is not a valid MP4 container.")
    return _read_video_metadata(path)


def _is_mp4(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            header = stream.read(12)
    except OSError:
        return False
    return len(header) == 12 and header[4:8] == b"ftyp"


def _read_video_metadata(path: Path) -> ScreenRecordingVideoMetadata:
    reader = read_frames(str(path))
    try:
        values = next(reader)
    finally:
        reader.close()
    size = values.get("size")
    frames_per_second = values.get("fps")
    duration_seconds = values.get("duration")
    if (
        not isinstance(size, tuple)
        or len(size) != 2
        or not all(isinstance(value, int) and value > 0 for value in size)
        or isinstance(frames_per_second, bool)
        or not isinstance(frames_per_second, int | float)
        or frames_per_second <= 0
        or isinstance(duration_seconds, bool)
        or not isinstance(duration_seconds, int | float)
        or duration_seconds <= 0
    ):
        raise ScreenRecordingError("The recording has invalid video metadata.")
    return ScreenRecordingVideoMetadata(
        width_px=size[0],
        height_px=size[1],
        frames_per_second=float(frames_per_second),
        duration_seconds=float(duration_seconds),
    )
