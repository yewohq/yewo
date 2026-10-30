"""Tests for fail-closed recording redaction."""

from pathlib import Path
import subprocess

from PIL import Image
import pytest

from yewo.artifacts import YEWO_REDACTION_POLICY_ID
from yewo.brand import MOSS
from yewo.devices.recording import (
    FfmpegScreenRecordingRedactor,
    ScreenRecordingError,
    ScreenRecordingMask,
    ScreenRecordingVideoMetadata,
)
from yewo.observations import RedactionRegion


RAW_VIDEO = b"\x00\x00\x00\x18ftypmp42raw"
SAFE_VIDEO = b"\x00\x00\x00\x18ftypmp42safe"


class _RunnerStub:
    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self.command: tuple[str, ...] = ()
        self.last_overlay_pixel: tuple[int, ...] | None = None

    def __call__(
        self,
        command,
        *,
        capture_output: bool,
        text: bool,
        timeout: float,
        check: bool,
    ):
        del capture_output, text, timeout, check
        self.command = tuple(command)
        input_paths = [
            Path(command[index + 1]) for index, value in enumerate(command[:-1]) if value == "-i"
        ]
        with Image.open(input_paths[1]) as overlay:
            pixel = overlay.getpixel((15, 8))
            assert isinstance(pixel, tuple)
            self.last_overlay_pixel = tuple(int(value) for value in pixel)
        if self.returncode == 0:
            Path(command[-1]).write_bytes(SAFE_VIDEO)
        return subprocess.CompletedProcess(command, self.returncode, "", "")


def _metadata(path: Path) -> ScreenRecordingVideoMetadata:
    assert path.read_bytes() in {RAW_VIDEO, SAFE_VIDEO}
    return ScreenRecordingVideoMetadata(16, 9, 30.0, 3.0)


def test_redactor_masks_full_frame_when_source_aspect_ratio_differs(tmp_path: Path) -> None:
    runner = _RunnerStub()
    redactor = FfmpegScreenRecordingRedactor(
        ffmpeg_executable="ffmpeg",
        metadata_reader=_metadata,
        command_runner=runner,
    )
    mask = ScreenRecordingMask(
        region=RedactionRegion(left=1, top=1, right=2, bottom=2),
        source_width_px=10,
        source_height_px=10,
        starts_at_seconds=2.0,
    )

    source = tmp_path / "raw.mp4"
    destination = tmp_path / "safe.mp4"
    source.write_bytes(RAW_VIDEO)
    result = redactor.redact(source, destination, (mask,))

    assert destination.read_bytes() == SAFE_VIDEO
    assert result.metadata.duration_seconds == 3.0
    assert result.provenance.policy_id == YEWO_REDACTION_POLICY_ID
    assert result.provenance.mask_count == 1
    assert runner.last_overlay_pixel == (*MOSS, 255)
    graph = runner.command[runner.command.index("-filter_complex") + 1]
    assert "gte(t,1.750000)" in graph


def test_redactor_does_not_publish_output_after_ffmpeg_failure(tmp_path: Path) -> None:
    redactor = FfmpegScreenRecordingRedactor(
        ffmpeg_executable="ffmpeg",
        metadata_reader=_metadata,
        command_runner=_RunnerStub(returncode=1),
    )
    mask = ScreenRecordingMask(
        region=RedactionRegion(left=0, top=0, right=10, bottom=10),
        source_width_px=10,
        source_height_px=10,
        starts_at_seconds=0,
    )

    source = tmp_path / "raw.mp4"
    source.write_bytes(RAW_VIDEO)
    with pytest.raises(ScreenRecordingError, match="FFmpeg could not redact"):
        redactor.redact(source, tmp_path / "safe.mp4", (mask,))


def test_redactor_rejects_materially_shortened_output(tmp_path: Path) -> None:
    def metadata(path: Path) -> ScreenRecordingVideoMetadata:
        duration = 1.0 if path.name == "safe.mp4" else 3.0
        return ScreenRecordingVideoMetadata(16, 9, 30.0, duration)

    redactor = FfmpegScreenRecordingRedactor(
        ffmpeg_executable="ffmpeg",
        metadata_reader=metadata,
        command_runner=_RunnerStub(),
    )
    source = tmp_path / "raw.mp4"
    source.write_bytes(RAW_VIDEO)
    mask = ScreenRecordingMask(
        region=RedactionRegion(left=0, top=0, right=10, bottom=10),
        source_width_px=10,
        source_height_px=10,
        starts_at_seconds=0,
    )

    with pytest.raises(ScreenRecordingError, match="incomplete"):
        redactor.redact(source, tmp_path / "safe.mp4", (mask,))
