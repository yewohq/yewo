"""Tests for the Appium screen-recorder adapter."""

import base64

import pytest

from yewo.devices.recording import AppiumScreenRecorder, ScreenRecordingError


VIDEO = b"\x00\x00\x00\x18ftypmp42video"


class _DriverStub:
    def __init__(self) -> None:
        self.start_failures = 0
        self.start_options: list[dict[str, int]] = []
        self.encoded_video: object = base64.b64encode(VIDEO).decode()
        self.stop_error: Exception | None = None

    def start_recording_screen(self, **options: int) -> None:
        self.start_options.append(options)
        if self.start_failures:
            self.start_failures -= 1
            raise RuntimeError("driver detail")

    def stop_recording_screen(self) -> object:
        if self.stop_error is not None:
            raise self.stop_error
        return self.encoded_video


def test_recorder_preserves_the_declared_limit_without_a_policy_downgrade() -> None:
    driver = _DriverStub()
    recorder = AppiumScreenRecorder(driver)

    start = recorder.start(1_800)

    assert start.requested_maximum_seconds == 1_800
    assert start.effective_maximum_seconds == 1_800
    assert driver.start_options == [{"timeLimit": 1_800}]


def test_recorder_decodes_nonempty_video() -> None:
    recorder = AppiumScreenRecorder(_DriverStub())
    recorder.start(30)

    capture = recorder.stop()
    try:
        assert capture.path.read_bytes() == VIDEO
    finally:
        capture.close()


def test_failed_stop_does_not_claim_that_restart_is_safe() -> None:
    driver = _DriverStub()
    driver.stop_error = RuntimeError("stop failed")
    recorder = AppiumScreenRecorder(driver)
    recorder.start(30)

    with pytest.raises(ScreenRecordingError):
        recorder.stop()

    assert recorder.can_start_after_failure is False


def test_invalid_payload_after_confirmed_stop_allows_a_fresh_segment() -> None:
    driver = _DriverStub()
    driver.encoded_video = "not base64"
    recorder = AppiumScreenRecorder(driver)
    recorder.start(30)

    with pytest.raises(ScreenRecordingError):
        recorder.stop()

    assert recorder.can_start_after_failure is True


@pytest.mark.parametrize(
    "encoded_video",
    ["", "not base64", base64.b64encode(b"not mp4").decode(), 4],
)
def test_recorder_rejects_invalid_video_without_leaking_driver_values(
    encoded_video: object,
) -> None:
    driver = _DriverStub()
    driver.encoded_video = encoded_video
    recorder = AppiumScreenRecorder(driver)
    recorder.start(30)

    with pytest.raises(ScreenRecordingError, match="recording") as captured:
        recorder.stop()

    assert "not base64" not in str(captured.value)


def test_recorder_rejects_overlapping_segments() -> None:
    recorder = AppiumScreenRecorder(_DriverStub())
    recorder.start(30)

    with pytest.raises(ScreenRecordingError, match="already active"):
        recorder.start(30)
