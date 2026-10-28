"""A secret-bearing video is masked before it becomes a run artifact."""

import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from imageio_ffmpeg import get_ffmpeg_exe, read_frames

from yewo.agent.actuation.actions import Coordinates
from yewo.agent.device.recording import DeviceRecording
from yewo.agent.device.recording_redaction import RecordingMask, redact_recording
from yewo.brand import MOSS
from yewo.utils.secrets import ProtectedRegion, ScreenshotMaskType, SecretScrubber


def _region(bounds: tuple[int, int, int, int] | None) -> ProtectedRegion:
    return ProtectedRegion(
        mask_type=ScreenshotMaskType.SECRET,
        bounds=bounds,
        artifact_labels=("$PASSWORD",),
        model_labels=("$PASSWORD",),
    )


def _make_video(path: Path) -> None:
    subprocess.run(
        [
            get_ffmpeg_exe(),
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=white:s=80x40:r=4:d=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        timeout=30,
    )


def _frames(path: Path) -> tuple[tuple[int, int], list[bytes]]:
    reader = read_frames(str(path), pix_fmt="rgb24")
    metadata = next(reader)
    return metadata["size"], list(reader)


def _pixel(frame: bytes, size: tuple[int, int], x: int, y: int) -> tuple[int, int, int]:
    offset = (y * size[0] + x) * 3
    return frame[offset], frame[offset + 1], frame[offset + 2]


class FinalizationTests(unittest.TestCase):
    def test_only_frames_after_secret_entry_receive_the_opaque_mask(self):
        with TemporaryDirectory() as directory:
            raw_path = Path(directory) / "raw.mp4"
            safe_path = Path(directory) / "safe.mp4"
            _make_video(raw_path)

            saved = redact_recording(
                raw_path,
                safe_path,
                (
                    RecordingMask(
                        region=_region((10, 5, 50, 30)),
                        source_size=(80, 40),
                        starts_at_seconds=1.0,
                    ),
                ),
            )

            self.assertTrue(saved)
            size, frames = _frames(safe_path)
            self.assertGreaterEqual(len(frames), 4)
            self.assertTrue(all(channel > 240 for channel in _pixel(frames[0], size, 20, 15)))
            mask_pixels = [
                _pixel(frames[-1], size, x, y)
                for y in range(5, 30)
                for x in range(10, 50)
            ]
            moss_pixels = sum(
                all(
                    abs(actual - wanted) < 18
                    for actual, wanted in zip(pixel, MOSS, strict=True)
                )
                for pixel in mask_pixels
            )
            self.assertGreater(moss_pixels, len(mask_pixels) // 2)
            self.assertTrue(all(channel > 240 for channel in _pixel(frames[-1], size, 70, 35)))

    def test_a_failed_transcode_publishes_no_video(self):
        with TemporaryDirectory() as directory:
            raw_path = Path(directory) / "raw.mp4"
            safe_path = Path(directory) / "safe.mp4"
            _make_video(raw_path)
            mask = RecordingMask(_region((0, 0, 10, 10)), (10, 10), 0.0)

            with patch(
                "yewo.agent.device.recording_redaction.subprocess.run",
                return_value=SimpleNamespace(returncode=1, stderr="encode failed"),
            ):
                saved = redact_recording(raw_path, safe_path, (mask,))

            self.assertFalse(saved)
            self.assertFalse(safe_path.exists())


class MaskTimingTests(unittest.TestCase):
    def test_the_segment_records_bounds_and_time_before_typing(self):
        recorder = DeviceRecording()
        recorder.redact_recording = True
        recorder.is_recording = True
        recorder._recording_started_at = 10.0
        recorder._recording_masks = []

        with patch("yewo.agent.device.recording.time.monotonic", return_value=12.5):
            recorder.start_recording_mask(_region((2, 3, 20, 30)), (80, 40))

        self.assertEqual(
            recorder._recording_masks,
            [RecordingMask(_region((2, 3, 20, 30)), (80, 40), 2.5)],
        )

    def test_an_unknown_field_location_masks_the_whole_frame(self):
        recorder = DeviceRecording()
        recorder.redact_recording = True
        recorder.is_recording = True
        recorder._recording_started_at = 1.0
        recorder._recording_masks = []

        with patch("yewo.agent.device.recording.time.monotonic", return_value=2.0):
            recorder.start_recording_mask(_region(None), (80, 40))

        self.assertIsNone(recorder._recording_masks[0].region.bounds)

    def test_an_orientation_mismatch_masks_the_whole_encoded_frame(self):
        from yewo.agent.device.recording_redaction import _scaled_bounds

        self.assertEqual(
            _scaled_bounds((2, 3, 20, 30), (40, 80), (80, 40)),
            (0, 0, 80, 40),
        )


class ProtectedRegionNotificationTests(unittest.TestCase):
    def test_a_declared_value_notifies_the_recorder_with_its_field_bounds(self):
        scrubber = SecretScrubber()
        scrubber.learn_secrets({"password": "hunter2"})
        regions = []
        scrubber.protected_region_callback = regions.append

        scrubber.protect_typed_region(
            "hunter2",
            Coordinates(x=20, y=20),
            [],
            bounds=SimpleNamespace(x_min=5, y_min=6, x_max=70, y_max=30),
        )

        self.assertEqual(regions[0].bounds, (5, 6, 70, 30))

    def test_ordinary_typed_text_does_not_notify_the_recorder(self):
        scrubber = SecretScrubber()
        scrubber.learn_secrets({"password": "hunter2"})
        regions = []
        scrubber.protected_region_callback = regions.append

        scrubber.protect_typed_region("hello", Coordinates(x=20, y=20), [])

        self.assertEqual(regions, [])
