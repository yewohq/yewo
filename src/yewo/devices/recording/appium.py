"""Appium adapter for platform-neutral screen recording."""

from __future__ import annotations

import base64
import binascii
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol, cast

from yewo.devices.recording.capture import (
    ScreenRecordingCapture,
    ScreenRecordingStart,
)
from yewo.devices.recording.errors import ScreenRecordingError
from yewo.devices.recording.records import ScreenRecordingCaptureMethod


class _AppiumRecordingDriver(Protocol):
    def start_recording_screen(self, **options: Any) -> None: ...

    def stop_recording_screen(self) -> str: ...


class AppiumScreenRecorder:
    """Normalize Appium recording calls and returned base64 video."""

    def __init__(self, driver: object) -> None:
        self._driver = cast(_AppiumRecordingDriver, driver)
        self._active = False
        self._start: ScreenRecordingStart | None = None
        self._can_start_after_failure = True

    @property
    def capture_method(self) -> ScreenRecordingCaptureMethod:
        """Return that Appium supplies the completed video bytes."""
        return ScreenRecordingCaptureMethod.DRIVER_RESPONSE

    @property
    def can_start_after_failure(self) -> bool:
        """Return whether the prior recorder is confirmed inactive."""
        return self._can_start_after_failure

    def start(
        self,
        maximum_duration_seconds: int,
        maximum_stop_seconds: float = 10.0,
        maximum_pull_seconds: float = 30.0,
    ) -> ScreenRecordingStart:
        if self._active:
            raise ScreenRecordingError("A screen-recording segment is already active.")
        start = ScreenRecordingStart(
            requested_maximum_seconds=maximum_duration_seconds,
            effective_maximum_seconds=maximum_duration_seconds,
            maximum_stop_seconds=maximum_stop_seconds,
            maximum_pull_seconds=maximum_pull_seconds,
        )
        try:
            self._driver.start_recording_screen(timeLimit=start.effective_maximum_seconds)
        except Exception as error:  # noqa: BLE001 - external driver boundary
            raise ScreenRecordingError("Appium could not start screen recording.") from error
        self._active = True
        self._start = start
        self._can_start_after_failure = False
        return start

    def stop(self) -> ScreenRecordingCapture:
        start = self._start
        if not self._active or start is None:
            raise ScreenRecordingError("No screen-recording segment is active.")
        self._active = False
        self._start = None
        try:
            encoded_video = self._driver.stop_recording_screen()
        except Exception as error:  # noqa: BLE001 - external driver boundary
            raise ScreenRecordingError("Appium could not stop screen recording.") from error
        self._can_start_after_failure = True
        if not isinstance(encoded_video, str) or not encoded_video.strip():
            raise ScreenRecordingError("Appium returned an empty screen recording.")
        directory = TemporaryDirectory(prefix="yewo-appium-recording-")
        path = Path(directory.name) / "capture.mp4"
        try:
            _decode_base64_file(encoded_video.strip(), path)
            with path.open("rb") as stream:
                header = stream.read(12)
            if len(header) != 12 or header[4:8] != b"ftyp":
                raise ScreenRecordingError("Appium returned an invalid MP4 recording.")
        except Exception:
            directory.cleanup()
            raise
        return ScreenRecordingCapture(path, start, cleanup=directory.cleanup)


def _decode_base64_file(encoded: str, destination: Path) -> None:
    chunk_size = 1024 * 1024
    try:
        with destination.open("wb") as stream:
            for offset in range(0, len(encoded), chunk_size):
                chunk = encoded[offset : offset + chunk_size]
                stream.write(base64.b64decode(chunk, validate=True))
    except (OSError, ValueError, binascii.Error) as error:
        raise ScreenRecordingError("Appium returned invalid recording data.") from error
    if destination.stat().st_size == 0:
        raise ScreenRecordingError("Appium returned an empty screen recording.")
