"""Tests for Android-specific interaction evidence."""

from typing import Any, cast

from yewo.devices.application_lifecycle import ApplicationLifecycleDriver
from yewo.devices.android.interaction import AndroidInteractionBackend
from yewo.devices.screen import ScreenCalibration, ScreenCoordinateSpace


class _SystemBarDriver:
    def get_system_bars(self) -> dict[str, object]:
        return {
            "statusBar": {
                "visible": True,
                "x": 0,
                "y": 0,
                "width": 100,
                "height": 20,
            },
        }


def test_backend_reads_system_regions_from_the_driver() -> None:
    driver = cast(ApplicationLifecycleDriver, cast(Any, _SystemBarDriver()))

    regions = AndroidInteractionBackend().system_regions(
        driver,
        ScreenCalibration(200, 400, 100, 200),
    )

    assert regions is not None
    assert regions[0].coordinate_space is ScreenCoordinateSpace.DRIVER
    assert [region.kind.value for region in regions] == ["status_bar"]
    assert (regions[0].left, regions[0].top, regions[0].right, regions[0].bottom) == (
        0,
        0,
        100,
        20,
    )


class _HiddenSystemBarDriver:
    def get_system_bars(self) -> dict[str, object]:
        return {
            "statusBar": {
                "visible": False,
                "x": 0,
                "y": 0,
                "width": 100,
                "height": 20,
            },
        }


def test_backend_excludes_hidden_system_regions() -> None:
    driver = cast(ApplicationLifecycleDriver, cast(Any, _HiddenSystemBarDriver()))

    regions = AndroidInteractionBackend().system_regions(
        driver,
        ScreenCalibration(200, 400, 100, 200),
    )

    assert regions == ()


class _MalformedSystemBarDriver:
    def get_system_bars(self) -> dict[str, object]:
        return {"statusBar": {"visible": True, "height": 20}}


def test_backend_preserves_unknown_for_unparsable_system_regions() -> None:
    driver = cast(ApplicationLifecycleDriver, cast(Any, _MalformedSystemBarDriver()))

    regions = AndroidInteractionBackend().system_regions(
        driver,
        ScreenCalibration(200, 400, 100, 200),
    )

    assert regions is None
