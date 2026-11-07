"""Tests for iOS-specific interaction evidence."""

from typing import Any, cast

from yewo.devices.application_lifecycle import ApplicationLifecycleDriver
from yewo.devices.ios.interaction import IosInteractionBackend
from yewo.devices.screen import ScreenCalibration, ScreenCoordinateSpace


class _ViewportDriver:
    def execute_script(self, script: str, params: dict[str, Any]) -> object:
        assert script == "mobile: viewportRect"
        assert params == {}
        return {"left": 0, "top": 40, "width": 200, "height": 320}


def test_backend_derives_system_regions_from_the_application_viewport() -> None:
    driver = cast(ApplicationLifecycleDriver, cast(Any, _ViewportDriver()))

    regions = IosInteractionBackend().system_regions(
        driver,
        ScreenCalibration(200, 400, 100, 200),
    )

    assert regions is not None
    assert all(item.coordinate_space is ScreenCoordinateSpace.SCREENSHOT for item in regions)
    assert [item.kind.value for item in regions] == ["status_bar", "navigation_bar"]
    assert [(item.left, item.top, item.right, item.bottom) for item in regions] == [
        (0, 0, 200, 40),
        (0, 360, 200, 400),
    ]


class _LegacyShapeDriver:
    def execute_script(self, script: str, params: dict[str, Any]) -> object:
        del script, params
        return {"x": 0, "y": 20, "width": 100, "height": 160}


def test_backend_preserves_unknown_for_an_undocumented_viewport_shape() -> None:
    driver = cast(ApplicationLifecycleDriver, cast(Any, _LegacyShapeDriver()))

    regions = IosInteractionBackend().system_regions(
        driver,
        ScreenCalibration(200, 400, 100, 200),
    )

    assert regions is None
