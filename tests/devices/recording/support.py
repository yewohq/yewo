"""Reusable recorder fakes for screen-recording tests."""

from collections import deque
from pathlib import Path
from tempfile import TemporaryDirectory

from yewo.devices.recording import (
    ScreenRecordingCapture,
    ScreenRecordingCaptureMethod,
    ScreenRecordingError,
    ScreenRecordingStart,
)


class RecorderStub:
    capture_method = ScreenRecordingCaptureMethod.DRIVER_RESPONSE

    def __init__(self, videos: tuple[bytes | Exception, ...] = ()) -> None:
        self.videos = deque(videos)
        self.start_failures = 0
        self.start_error: Exception | None = None
        self.start_limits: list[int] = []
        self.stop_count = 0
        self.can_start_after_failure = True
        self._start: ScreenRecordingStart | None = None
        self.acknowledgement_error: Exception | None = None
        self.acknowledgement_count = 0

    def start(
        self,
        maximum_duration_seconds: int,
        maximum_stop_seconds: float = 10.0,
        maximum_pull_seconds: float = 120.0,
    ) -> ScreenRecordingStart:
        self.start_limits.append(maximum_duration_seconds)
        if self.start_error is not None:
            raise self.start_error
        if self.start_failures:
            self.start_failures -= 1
            raise ScreenRecordingError("start failed")
        self._start = ScreenRecordingStart(
            maximum_duration_seconds,
            maximum_duration_seconds,
            maximum_stop_seconds,
            maximum_pull_seconds,
        )
        return self._start

    def stop(self) -> ScreenRecordingCapture:
        self.stop_count += 1
        value = self.videos.popleft()
        if isinstance(value, Exception):
            self.can_start_after_failure = False
            raise value
        assert self._start is not None
        directory = TemporaryDirectory(prefix="yewo-recording-test-")
        path = Path(directory.name) / "capture.mp4"
        path.write_bytes(value)
        self.can_start_after_failure = True

        def acknowledge() -> None:
            self.acknowledgement_count += 1
            if self.acknowledgement_error is not None:
                raise self.acknowledgement_error

        return ScreenRecordingCapture(
            path,
            self._start,
            acknowledge=(
                acknowledge
                if self.capture_method is ScreenRecordingCaptureMethod.REMOTE_FILE
                else None
            ),
            cleanup=directory.cleanup,
        )


class RemoteRecorderStub(RecorderStub):
    capture_method = ScreenRecordingCaptureMethod.REMOTE_FILE
