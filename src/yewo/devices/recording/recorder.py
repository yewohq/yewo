"""Validated boundary for device-side screen recorders."""

from typing import Protocol

from yewo.devices.recording.capture import ScreenRecordingCapture, ScreenRecordingStart
from yewo.devices.recording.errors import ScreenRecordingError
from yewo.devices.recording.records import ScreenRecordingCaptureMethod


class ScreenRecorder(Protocol):
    """Start and stop one device-side recording segment."""

    @property
    def capture_method(self) -> ScreenRecordingCaptureMethod: ...

    @property
    def can_start_after_failure(self) -> bool: ...

    def start(
        self,
        maximum_duration_seconds: int,
        maximum_stop_seconds: float = 10.0,
        maximum_pull_seconds: float = 30.0,
    ) -> ScreenRecordingStart: ...

    def stop(self) -> ScreenRecordingCapture: ...


class ScreenRecorderBoundary:
    """Normalize injected recorder failures into the recording error contract."""

    def __init__(self, recorder: ScreenRecorder) -> None:
        self._recorder = recorder

    @property
    def capture_method(self) -> ScreenRecordingCaptureMethod:
        try:
            return self._recorder.capture_method
        except Exception as error:  # noqa: BLE001 - injected recorder boundary
            raise ScreenRecordingError("The screen recorder method is unavailable.") from error

    @property
    def can_start_after_failure(self) -> bool:
        try:
            return self._recorder.can_start_after_failure
        except Exception:  # noqa: BLE001 - an unknown recorder state is unsafe
            return False

    def start(
        self,
        maximum_duration_seconds: int,
        maximum_stop_seconds: float,
        maximum_pull_seconds: float,
    ) -> ScreenRecordingStart:
        try:
            return self._recorder.start(
                maximum_duration_seconds,
                maximum_stop_seconds,
                maximum_pull_seconds,
            )
        except ScreenRecordingError:
            raise
        except Exception as error:  # noqa: BLE001 - injected recorder boundary
            raise ScreenRecordingError("The screen recorder could not start.") from error

    def stop(self) -> ScreenRecordingCapture:
        try:
            return self._recorder.stop()
        except ScreenRecordingError:
            raise
        except Exception as error:  # noqa: BLE001 - injected recorder boundary
            raise ScreenRecordingError("The screen recorder could not stop.") from error
