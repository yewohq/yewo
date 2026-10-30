"""iOS gesture dispatch through XCUITest extensions."""

from __future__ import annotations

from yewo.devices.application_lifecycle import ApplicationLifecycleDriver
from yewo.devices.interaction import InteractionDriver
from yewo.devices.screen import (
    ScreenCalibration,
    ScreenCoordinateSpace,
    ScreenRegionReading,
    SystemRegionKind,
    bounded_rectangle,
)


class IosInteractionBackend:
    """Dispatch iOS interactions and read platform context."""

    def tap(self, driver: InteractionDriver, x: int, y: int) -> None:
        driver.execute_script("mobile: tap", {"x": x, "y": y})

    def double_tap(self, driver: InteractionDriver, x: int, y: int) -> None:
        driver.execute_script("mobile: doubleTap", {"x": x, "y": y})

    def long_press(
        self, driver: InteractionDriver, x: int, y: int, duration_seconds: float
    ) -> None:
        driver.execute_script(
            "mobile: touchAndHold",
            {"x": x, "y": y, "duration": duration_seconds},
        )

    def drag(
        self,
        driver: InteractionDriver,
        start: tuple[int, int],
        end: tuple[int, int],
        duration_seconds: float,
    ) -> None:
        driver.execute_script(
            "mobile: dragFromToForDuration",
            {
                "fromX": start[0],
                "fromY": start[1],
                "toX": end[0],
                "toY": end[1],
                "duration": duration_seconds,
            },
        )

    def active_application_id(
        self,
        driver: ApplicationLifecycleDriver,
    ) -> str | None:
        try:
            answer = driver.execute_script("mobile: activeAppInfo", {})
        except Exception:  # noqa: BLE001 - an unavailable read is unknown
            return None
        if not isinstance(answer, dict):
            return None
        bundle_id = answer.get("bundleId")
        return bundle_id if isinstance(bundle_id, str) and bundle_id else None

    def application_context(
        self,
        driver: ApplicationLifecycleDriver,
    ) -> str | None:
        del driver
        return None

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
            answer = driver.execute_script("mobile: viewportRect", {})
        except Exception:  # noqa: BLE001 - external driver boundary
            return None
        if isinstance(answer, dict) and isinstance(answer.get("value"), dict):
            answer = answer["value"]
        screenshot_size = (
            calibration.screenshot_width_px,
            calibration.screenshot_height_px,
        )
        viewport = bounded_rectangle(
            answer,
            screenshot_size,
            horizontal_key="left",
            vertical_key="top",
        )
        if viewport is None:
            return None
        left, top, right, bottom = viewport
        width, height = screenshot_size
        candidates = (
            (SystemRegionKind.STATUS_BAR, (0, 0, width, top)),
            (SystemRegionKind.NAVIGATION_BAR, (0, bottom, width, height)),
            (SystemRegionKind.OTHER, (0, top, left, bottom)),
            (SystemRegionKind.OTHER, (right, top, width, bottom)),
        )
        return tuple(
            ScreenRegionReading(kind, ScreenCoordinateSpace.SCREENSHOT, *bounds)
            for kind, bounds in candidates
            if bounds[0] < bounds[2] and bounds[1] < bounds[3]
        )

    def clear_application_data(
        self,
        driver: ApplicationLifecycleDriver,
        application_id: str,
    ) -> None:
        driver.execute_script("mobile: clearApp", {"bundleId": application_id})

    def open_deep_link(
        self,
        driver: ApplicationLifecycleDriver,
        application_id: str,
        url: str,
    ) -> None:
        driver.execute_script(
            "mobile: deepLink",
            {"url": url, "bundleId": application_id},
        )
