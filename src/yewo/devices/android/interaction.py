"""Android gesture dispatch through UiAutomator2 extensions."""

from __future__ import annotations

from collections.abc import Mapping

from yewo.devices.application_lifecycle import ApplicationLifecycleDriver
from yewo.devices.interaction import InteractionDriver
from yewo.devices.screen import (
    ScreenCalibration,
    ScreenCoordinateSpace,
    ScreenRegionReading,
    SystemRegionKind,
    bounded_rectangle,
)


class AndroidInteractionBackend:
    """Dispatch Android interactions and read platform context."""

    def tap(self, driver: InteractionDriver, x: int, y: int) -> None:
        driver.execute_script("mobile: clickGesture", {"x": x, "y": y})

    def double_tap(self, driver: InteractionDriver, x: int, y: int) -> None:
        driver.execute_script("mobile: doubleClickGesture", {"x": x, "y": y})

    def long_press(
        self, driver: InteractionDriver, x: int, y: int, duration_seconds: float
    ) -> None:
        driver.execute_script(
            "mobile: longClickGesture",
            {"x": x, "y": y, "duration": round(duration_seconds * 1000)},
        )

    def drag(
        self,
        driver: InteractionDriver,
        start: tuple[int, int],
        end: tuple[int, int],
        duration_seconds: float,
    ) -> None:
        speed = max(1, round(_distance(start, end) / duration_seconds))
        driver.execute_script(
            "mobile: dragGesture",
            {
                "startX": start[0],
                "startY": start[1],
                "endX": end[0],
                "endY": end[1],
                "speed": speed,
            },
        )

    def active_application_id(
        self,
        driver: ApplicationLifecycleDriver,
    ) -> str | None:
        try:
            value = getattr(driver, "current_package", None)
        except Exception:  # noqa: BLE001 - an unavailable read is unknown
            return None
        return value if isinstance(value, str) and value else None

    def application_context(
        self,
        driver: ApplicationLifecycleDriver,
    ) -> str | None:
        try:
            value = getattr(driver, "current_activity", None)
        except Exception:  # noqa: BLE001 - external driver boundary
            return None
        return value if isinstance(value, str) and value else None

    def device_orientation(
        self,
        driver: ApplicationLifecycleDriver,
    ) -> str | None:
        try:
            value = getattr(driver, "orientation", None)
        except Exception:  # noqa: BLE001 - external driver boundary
            return None
        return value.strip().lower() if isinstance(value, str) and value.strip() else None

    def system_regions(
        self,
        driver: ApplicationLifecycleDriver,
        calibration: ScreenCalibration,
    ) -> tuple[ScreenRegionReading, ...] | None:
        try:
            reader = getattr(driver, "get_system_bars", None)
            answer = reader() if callable(reader) else None
        except Exception:  # noqa: BLE001 - external driver boundary
            return None
        if not isinstance(answer, dict) or not answer:
            return None
        regions: list[ScreenRegionReading] = []
        for name, value in answer.items():
            if not isinstance(value, Mapping) or not isinstance(value.get("visible"), bool):
                return None
            if value["visible"] is False:
                continue
            region = _region_from_mapping(
                name,
                value,
                (calibration.driver_width_px, calibration.driver_height_px),
            )
            if region is None:
                return None
            regions.append(region)
        return tuple(regions)

    def clear_application_data(
        self,
        driver: ApplicationLifecycleDriver,
        application_id: str,
    ) -> None:
        driver.execute_script("mobile: clearApp", {"appId": application_id})

    def open_deep_link(
        self,
        driver: ApplicationLifecycleDriver,
        application_id: str,
        url: str,
    ) -> None:
        driver.execute_script(
            "mobile: deepLink",
            {"url": url, "package": application_id},
        )


def _distance(start: tuple[int, int], end: tuple[int, int]) -> float:
    return ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5


def _region_from_mapping(
    name: object,
    value: object,
    driver_size: tuple[int, int],
) -> ScreenRegionReading | None:
    bounds = bounded_rectangle(value, driver_size)
    if bounds is None:
        return None
    normalized_name = str(name).lower()
    if "status" in normalized_name:
        kind = SystemRegionKind.STATUS_BAR
    elif "navigation" in normalized_name:
        kind = SystemRegionKind.NAVIGATION_BAR
    else:
        kind = SystemRegionKind.OTHER
    return ScreenRegionReading(kind, ScreenCoordinateSpace.DRIVER, *bounds)
