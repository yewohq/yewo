"""Tests for application-scoped device interaction behavior."""

from io import BytesIO

from PIL import Image
import pytest

from devices._interaction_stubs import DriverStub
from devices._interaction_stubs import application
from yewo.devices import DevicePlatform
from yewo.devices.android.interaction import AndroidInteractionBackend
from yewo.devices.errors import ApplicationFirstPaintError, DeviceInteractionError
from yewo.devices.interaction import DeviceInteraction


class _ClockStub:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _timed_interaction(driver: DriverStub, clock: _ClockStub) -> DeviceInteraction:
    return DeviceInteraction(
        driver,
        DevicePlatform.ANDROID,
        application(DevicePlatform.ANDROID),
        AndroidInteractionBackend(),
        application_transition_timeout_seconds=0.3,
        application_poll_interval_seconds=0.1,
        clock=clock,
    )


@pytest.mark.parametrize("duration_seconds", [-0.1, 31])
def test_wait_limits_fail_without_sleeping(duration_seconds: float) -> None:
    clock = _ClockStub()
    interaction = _timed_interaction(DriverStub(), clock)

    with pytest.raises(DeviceInteractionError, match="Wait duration"):
        interaction.wait(duration_seconds)

    assert clock.now == 0


def test_first_paint_does_not_capture_another_application() -> None:
    class ScreenshotDriver(DriverStub):
        def __init__(self) -> None:
            super().__init__()
            self.current_package = "com.example.launcher"
            self.screenshot_calls = 0

        def get_screenshot_as_png(self) -> bytes:
            self.screenshot_calls += 1
            return _image_png([0, 255, 0, 255])

    clock = _ClockStub()
    driver = ScreenshotDriver()
    interaction = _timed_interaction(driver, clock)

    with pytest.raises(ApplicationFirstPaintError, match="did not become active"):
        interaction.wait_for_first_paint()

    assert driver.screenshot_calls == 0
    assert clock.now == pytest.approx(0.3)


def test_first_paint_waits_without_capturing_another_application() -> None:
    class DepartingDriver(DriverStub):
        def __init__(self) -> None:
            super().__init__()
            self.state_reads = 0
            self.screenshot_calls = 0

        def query_app_state(self, application_id: str) -> int:
            del application_id
            self.state_reads += 1
            if self.state_reads >= 3:
                self.current_package = "com.example.launcher"
            return self.state

        def get_screenshot_as_png(self) -> bytes:
            self.screenshot_calls += 1
            return _image_png([0, 0, 0, 0])

        def get_window_size(self) -> dict[str, int]:
            return {"width": 2, "height": 2}

    clock = _ClockStub()
    driver = DepartingDriver()
    interaction = _timed_interaction(driver, clock)

    with pytest.raises(ApplicationFirstPaintError, match="did not remain in the foreground"):
        interaction.wait_for_first_paint(timeout_seconds=1, poll_interval_seconds=0.1)

    assert driver.screenshot_calls == 1
    assert clock.now == pytest.approx(1)


def test_first_paint_reports_blank_when_foreground_identity_is_unknown() -> None:
    class IdentityUnavailableDriver(DriverStub):
        def __init__(self) -> None:
            super().__init__()
            self.current_package = None

        def get_screenshot_as_png(self) -> bytes:
            return _image_png([0, 0, 0, 0])

        def get_window_size(self) -> dict[str, int]:
            return {"width": 2, "height": 2}

    clock = _ClockStub()
    interaction = _timed_interaction(IdentityUnavailableDriver(), clock)

    with pytest.raises(ApplicationFirstPaintError, match="remained blank"):
        interaction.wait_for_first_paint(
            timeout_seconds=0.3,
            poll_interval_seconds=0.1,
        )

    assert clock.now == pytest.approx(0.3)


def _image_png(pixels: list[int]) -> bytes:
    image = Image.new("L", (2, 2))
    image.putdata(pixels)
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def test_first_paint_waits_through_a_blank_frame() -> None:
    class PaintDriver(DriverStub):
        def __init__(self) -> None:
            super().__init__()
            self.frames = [_image_png([0, 0, 0, 0]), _image_png([0, 255, 0, 255])]

        def get_screenshot_as_png(self) -> bytes:
            return self.frames.pop(0)

        def get_window_size(self) -> dict[str, int]:
            return {"width": 2, "height": 2}

    clock = _ClockStub()
    interaction = DeviceInteraction(
        PaintDriver(),
        DevicePlatform.ANDROID,
        application(DevicePlatform.ANDROID),
        AndroidInteractionBackend(),
        clock=clock,
    )

    painted = interaction.wait_for_first_paint(
        timeout_seconds=1,
        poll_interval_seconds=0.1,
    )

    assert painted == _image_png([0, 255, 0, 255])
    assert clock.now == 0.1
