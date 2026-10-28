"""Apply time-bound secret masks before a screen recording is published."""

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from imageio_ffmpeg import get_ffmpeg_exe, read_frames
from PIL import Image, ImageDraw

from yewo.agent.artifacts.redaction import draw_secret_mask
from yewo.utils.secrets import ProtectedRegion

logger = logging.getLogger(__name__)

MASK_LEAD_SECONDS = 0.25
REDACTION_TIMEOUT_SECONDS = 900


@dataclass(frozen=True)
class RecordingMask:
    """One screen region that stays covered from a point in a video onward."""

    region: ProtectedRegion
    source_size: tuple[int, int]
    starts_at_seconds: float


def _video_metadata(raw_path: Path) -> tuple[tuple[int, int], float, float]:
    """Read the encoded size, cadence, and duration without decoding the video."""
    reader = read_frames(str(raw_path))
    try:
        metadata = next(reader)
    finally:
        reader.close()
    return metadata["size"], metadata["fps"] or 30.0, metadata["duration"]


def _scaled_bounds(
    bounds: tuple[int, int, int, int] | None,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Map screenshot coordinates into the encoded video dimensions."""
    if bounds is None:
        return 0, 0, target_size[0], target_size[1]
    source_width, source_height = source_size
    target_width, target_height = target_size
    if source_width * target_height != source_height * target_width:
        return 0, 0, target_width, target_height
    left, top, right, bottom = bounds
    scaled = (
        round(left * target_width / source_width),
        round(top * target_height / source_height),
        round(right * target_width / source_width),
        round(bottom * target_height / source_height),
    )
    return (
        max(0, min(scaled[0], target_width)),
        max(0, min(scaled[1], target_height)),
        max(0, min(scaled[2], target_width)),
        max(0, min(scaled[3], target_height)),
    )


def _write_overlay(
    mask: RecordingMask, path: Path, target_size: tuple[int, int]
) -> None:
    """Render the same branded overlay used for a persisted screenshot."""
    bounds = _scaled_bounds(mask.region.bounds, mask.source_size, target_size)
    image = Image.new("RGBA", target_size, (0, 0, 0, 0))
    draw_secret_mask(
        ImageDraw.Draw(image),
        bounds,
        mask.region.artifact_labels,
        mask.region.mask_type,
    )
    image.save(path)


def _overlay_graph(masks: tuple[RecordingMask, ...]) -> tuple[str, str]:
    """Apply each branded overlay from its secret-entry timestamp onward."""
    filters = []
    video = "[0:v]"
    for index, mask in enumerate(masks, start=1):
        redacted = f"redacted{index}"
        start = max(0.0, mask.starts_at_seconds - MASK_LEAD_SECONDS)
        filters.append(
            f"{video}[{index}:v]overlay=0:0:enable='gte(t,{start:.6f})'[{redacted}]"
        )
        video = f"[{redacted}]"
    return ";".join(filters), video


def redact_recording(
    raw_path: Path,
    safe_path: Path,
    masks: tuple[RecordingMask, ...],
) -> bool:
    """Write a masked MP4, returning False without publishing on any failure."""
    if not masks:
        raw_path.replace(safe_path)
        return True
    try:
        target_size, frame_rate, duration = _video_metadata(raw_path)
        with TemporaryDirectory(prefix=".overlays-", dir=safe_path.parent) as directory:
            overlay_paths = []
            for index, mask in enumerate(masks, start=1):
                overlay_path = Path(directory) / f"mask_{index}.png"
                _write_overlay(mask, overlay_path, target_size)
                overlay_paths.append(overlay_path)
            graph, video = _overlay_graph(masks)
            command = [
                get_ffmpeg_exe(), "-nostdin", "-v", "error", "-y", "-i", str(raw_path)
            ]
            for overlay_path in overlay_paths:
                command.extend(
                    ["-loop", "1", "-framerate", str(frame_rate), "-i", str(overlay_path)]
                )
            command.extend(
                [
                    "-filter_complex",
                    graph,
                    "-map",
                    video,
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
                    str(duration),
                    str(safe_path),
                ]
            )
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=REDACTION_TIMEOUT_SECONDS,
                check=False,
            )
    except Exception as exc:  # noqa: BLE001 - a secret-bearing video must fail closed
        logger.warning("Could not redact the screen recording: %s", exc)
        return False
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()[-1:] or ["unknown ffmpeg error"]
        logger.warning("Could not redact the screen recording: %s", detail[0])
        return False
    return safe_path.is_file() and safe_path.stat().st_size > 0
