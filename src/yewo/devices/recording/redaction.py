"""Apply time-bound region masks before publishing a screen recording."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from typing import Protocol

from imageio_ffmpeg import get_ffmpeg_exe, read_frames
from PIL import Image, ImageDraw

from yewo.artifacts import YEWO_REDACTION_POLICY_ID, draw_yewo_redaction_mask
from yewo.artifacts.records import ArtifactSensitivity
from yewo.devices.recording.errors import ScreenRecordingError
from yewo.devices.recording.records import (
    ScreenRecordingMask,
    ScreenRecordingRedactionProvenance,
)


MASK_LEAD_SECONDS = 0.25
REDACTION_TIMEOUT_SECONDS = 900.0


@dataclass(frozen=True)
class ScreenRecordingRedactionResult:
    """Metadata and evidence for one redacted MP4 file."""

    metadata: ScreenRecordingVideoMetadata
    provenance: ScreenRecordingRedactionProvenance


@dataclass(frozen=True)
class ScreenRecordingVideoMetadata:
    width_px: int
    height_px: int
    frames_per_second: float
    duration_seconds: float


class ScreenRecordingRedactor(Protocol):
    """Write MP4 pixels after applying approved time-bound masks."""

    def redact(
        self,
        source: Path,
        destination: Path,
        masks: Sequence[ScreenRecordingMask],
    ) -> ScreenRecordingRedactionResult: ...


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


class FfmpegScreenRecordingRedactor:
    """Render opaque time-bound masks with an isolated FFmpeg process."""

    def __init__(
        self,
        *,
        ffmpeg_executable: str | None = None,
        metadata_reader: Callable[[Path], ScreenRecordingVideoMetadata] | None = None,
        command_runner: _CommandRunner = _run_command,
    ) -> None:
        self._ffmpeg_executable = ffmpeg_executable
        self._metadata_reader = metadata_reader or _read_video_metadata
        self._command_runner = command_runner

    def redact(
        self,
        source: Path,
        destination: Path,
        masks: Sequence[ScreenRecordingMask],
    ) -> ScreenRecordingRedactionResult:
        ordered_masks = tuple(
            sorted(
                masks,
                key=lambda item: (
                    item.starts_at_seconds,
                    item.ends_at_seconds if item.ends_at_seconds is not None else float("inf"),
                    item.region.left,
                    item.region.top,
                    item.region.right,
                    item.region.bottom,
                    item.source_width_px,
                    item.source_height_px,
                ),
            )
        )
        if not ordered_masks:
            raise ScreenRecordingError("Recording redaction requires at least one mask.")
        if not _is_mp4(source):
            raise ScreenRecordingError("Recording redaction requires a valid MP4 container.")
        try:
            metadata = self._run_redaction(source, destination, ordered_masks)
        except ScreenRecordingError:
            raise
        except Exception as error:  # noqa: BLE001 - media decoder and process boundary
            raise ScreenRecordingError("Could not redact the screen recording.") from error
        return ScreenRecordingRedactionResult(
            metadata=metadata,
            provenance=ScreenRecordingRedactionProvenance(
                policy_id=YEWO_REDACTION_POLICY_ID,
                mask_count=len(ordered_masks),
                mask_set_sha256=recording_mask_set_digest(ordered_masks),
                input_sensitivity=ArtifactSensitivity.DECLARED_SENSITIVE,
            ),
        )

    def _run_redaction(
        self,
        source: Path,
        destination: Path,
        masks: tuple[ScreenRecordingMask, ...],
    ) -> ScreenRecordingVideoMetadata:
        with TemporaryDirectory(prefix="yewo-recording-overlays-") as directory:
            temporary_root = Path(directory)
            metadata = self._metadata_reader(source)
            overlay_paths = tuple(
                _write_overlay(mask, metadata, temporary_root, index)
                for index, mask in enumerate(masks, start=1)
            )
            graph, output_stream = _overlay_graph(
                masks,
                metadata.duration_seconds,
            )
            command = _build_command(
                self._ffmpeg_executable or get_ffmpeg_exe(),
                source,
                destination,
                overlay_paths,
                graph,
                output_stream,
                metadata,
            )
            completed = self._command_runner(
                command,
                capture_output=True,
                text=True,
                timeout=REDACTION_TIMEOUT_SECONDS,
                check=False,
            )
            if completed.returncode:
                raise ScreenRecordingError("FFmpeg could not redact the screen recording.")
        if not _is_mp4(destination):
            raise ScreenRecordingError("FFmpeg produced an invalid redacted recording.")
        output_metadata = self._metadata_reader(destination)
        tolerance_seconds = max(0.5, 2 / metadata.frames_per_second)
        if abs(output_metadata.duration_seconds - metadata.duration_seconds) > tolerance_seconds:
            raise ScreenRecordingError("FFmpeg produced an incomplete redacted recording.")
        return output_metadata


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


def _write_overlay(
    mask: ScreenRecordingMask,
    metadata: ScreenRecordingVideoMetadata,
    directory: Path,
    index: int,
) -> Path:
    target_size = (metadata.width_px, metadata.height_px)
    bounds = _scaled_bounds(mask, target_size)
    image = Image.new("RGBA", target_size, (0, 0, 0, 0))
    draw_yewo_redaction_mask(ImageDraw.Draw(image), bounds)
    path = directory / f"mask_{index:06d}.png"
    image.save(path)
    return path


def _scaled_bounds(
    mask: ScreenRecordingMask,
    target_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    target_width, target_height = target_size
    if mask.source_width_px * target_height != mask.source_height_px * target_width:
        return 0, 0, target_width - 1, target_height - 1
    region = mask.region
    scaled = (
        round(region.left * target_width / mask.source_width_px),
        round(region.top * target_height / mask.source_height_px),
        round(region.right * target_width / mask.source_width_px) - 1,
        round(region.bottom * target_height / mask.source_height_px) - 1,
    )
    clamped = (
        max(0, min(scaled[0], target_width - 1)),
        max(0, min(scaled[1], target_height - 1)),
        max(0, min(scaled[2], target_width - 1)),
        max(0, min(scaled[3], target_height - 1)),
    )
    if clamped[0] > clamped[2] or clamped[1] > clamped[3]:
        return 0, 0, target_width - 1, target_height - 1
    return clamped


def _overlay_graph(
    masks: tuple[ScreenRecordingMask, ...],
    duration_seconds: float,
) -> tuple[str, str]:
    filters: list[str] = []
    input_stream = "[0:v]"
    for index, mask in enumerate(masks, start=1):
        output_stream = f"[redacted{index}]"
        proposed_start = max(0.0, mask.starts_at_seconds - MASK_LEAD_SECONDS)
        start = 0.0 if proposed_start >= duration_seconds else proposed_start
        enable = f"gte(t,{start:.6f})"
        if mask.ends_at_seconds is not None:
            end = min(duration_seconds, mask.ends_at_seconds + MASK_LEAD_SECONDS)
            enable = f"between(t,{start:.6f},{end:.6f})"
        filters.append(f"{input_stream}[{index}:v]overlay=0:0:enable='{enable}'{output_stream}")
        input_stream = output_stream
    return ";".join(filters), input_stream


def _build_command(
    executable: str,
    raw_path: Path,
    safe_path: Path,
    overlay_paths: tuple[Path, ...],
    graph: str,
    output_stream: str,
    metadata: ScreenRecordingVideoMetadata,
) -> tuple[str, ...]:
    command = [executable, "-nostdin", "-v", "error", "-y", "-i", str(raw_path)]
    for overlay_path in overlay_paths:
        command.extend(
            [
                "-loop",
                "1",
                "-framerate",
                str(metadata.frames_per_second),
                "-i",
                str(overlay_path),
            ]
        )
    command.extend(
        [
            "-filter_complex",
            graph,
            "-map",
            output_stream,
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            "-t",
            str(metadata.duration_seconds),
            str(safe_path),
        ]
    )
    return tuple(command)


def recording_mask_set_digest(masks: Sequence[ScreenRecordingMask]) -> str:
    """Return a canonical digest for one ordered recording-mask set."""
    ordered_masks = sorted(
        masks,
        key=lambda item: (
            item.starts_at_seconds,
            item.ends_at_seconds if item.ends_at_seconds is not None else float("inf"),
            item.region.left,
            item.region.top,
            item.region.right,
            item.region.bottom,
            item.source_width_px,
            item.source_height_px,
        ),
    )
    canonical = json.dumps(
        [mask.model_dump(mode="json") for mask in ordered_masks],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return sha256(canonical).hexdigest()
