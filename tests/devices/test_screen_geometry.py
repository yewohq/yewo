"""Screenshot geometry timing and bounded driver diagnostics."""

from pathlib import Path

import pytest

from devices._interaction_stubs import DriverStub, png
from yewo.devices.errors import DeviceInteractionError
from yewo.devices.screen import ScreenGeometry
from yewo.run_narration import RunNarrator, narration_scope


def test_screen_geometry_traces_screenshot_and_window_size_duration(tmp_path: Path) -> None:
    moments = iter((10.0, 10.125, 20.0, 20.05))
    screen = ScreenGeometry(DriverStub(), monotonic=lambda: next(moments))

    with narration_scope(RunNarrator(tmp_path)):
        screen.capture()

    trace = (tmp_path / "runtime/trace.log").read_text()
    assert "appium-command: operation=screenshot duration_ms=125.0" in trace
    assert "appium-command: operation=window-size duration_ms=50.0" in trace


def test_screen_geometry_traces_bounded_driver_failure_detail(tmp_path: Path) -> None:
    class FailingScreenshotDriver(DriverStub):
        def get_screenshot_as_png(self) -> bytes:
            error = RuntimeError("session unavailable\nconnection closed")
            error.msg = "invalid session id"  # type: ignore[attr-defined]
            raise error

    screen = ScreenGeometry(FailingScreenshotDriver(), sleep=lambda unused: None)

    with (
        narration_scope(RunNarrator(tmp_path)),
        pytest.raises(DeviceInteractionError, match="Could not capture"),
    ):
        screen.capture()

    trace = (tmp_path / "runtime/trace.log").read_text()
    assert (
        "appium-command: operation=screenshot state=failed "
        "attempt=1/3 error=RuntimeError detail=invalid session id"
    ) in trace


def test_screen_geometry_retries_a_transient_screenshot_failure() -> None:
    class TransientScreenshotDriver(DriverStub):
        def __init__(self) -> None:
            super().__init__()
            self.screenshot_calls = 0

        def get_screenshot_as_png(self) -> bytes:
            self.screenshot_calls += 1
            if self.screenshot_calls == 1:
                raise ConnectionError("socket hang up")
            return super().get_screenshot_as_png()

    driver = TransientScreenshotDriver()
    delays: list[float] = []
    screen = ScreenGeometry(driver, sleep=delays.append)

    result = screen.capture()

    assert result == png(200, 400)
    assert driver.screenshot_calls == 2
    assert delays == [0.25]
