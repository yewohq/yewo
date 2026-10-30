"""Screenshot capture, geometry calibration, and blank-frame analysis."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from io import BytesIO
from typing import Protocol

from PIL import Image, ImageStat, UnidentifiedImageError

from yewo.devices.errors import DeviceInteractionError


class ScreenDriver(Protocol):
    """Driver reads required for screenshot geometry."""

    def get_screenshot_as_png(self) -> bytes: ...

    def get_window_size(self) -> dict[str, int]: ...


class SystemRegionKind(StrEnum):
    """A platform-owned region visible in screenshot coordinates."""

    STATUS_BAR = "status_bar"
    NAVIGATION_BAR = "navigation_bar"
    KEYBOARD = "keyboard"
    DISPLAY_CUTOUT = "display_cutout"
    OTHER = "other"


class ScreenCoordinateSpace(StrEnum):
    """Coordinate space used by one platform geometry reading."""

    DRIVER = "driver"
    SCREENSHOT = "screenshot"


@dataclass(frozen=True)
class ScreenRegionReading:
    """One platform-owned rectangle with an explicit coordinate space."""

    kind: SystemRegionKind
    coordinate_space: ScreenCoordinateSpace
    left: int
    top: int
    right: int
    bottom: int


def bounded_rectangle(
    value: object,
    bounds_size: tuple[int, int],
    *,
    horizontal_key: str = "x",
    vertical_key: str = "y",
) -> tuple[int, int, int, int] | None:
    """Normalize one rectangle and clamp it to its declared coordinate space."""
    if not isinstance(value, Mapping):
        return None
    coordinates = tuple(value.get(key) for key in (horizontal_key, vertical_key, "width", "height"))
    if not all(isinstance(item, int | float) for item in coordinates):
        return None
    numeric_coordinates = tuple(
        float(item) for item in coordinates if isinstance(item, int | float)
    )
    x, y, width, height = (round(item) for item in numeric_coordinates)
    bounds_width, bounds_height = bounds_size
    left = max(0, min(bounds_width, x))
    top = max(0, min(bounds_height, y))
    right = max(left, min(bounds_width, x + width))
    bottom = max(top, min(bounds_height, y + height))
    if left >= right or top >= bottom:
        return None
    return left, top, right, bottom


@dataclass(frozen=True)
class ScreenCalibration:
    """Observed screenshot and Appium viewport geometry."""

    screenshot_width_px: int
    screenshot_height_px: int
    driver_width_px: int
    driver_height_px: int

    @property
    def scale_x(self) -> float:
        return self.driver_width_px / self.screenshot_width_px

    @property
    def scale_y(self) -> float:
        return self.driver_height_px / self.screenshot_height_px


class ScreenGeometry:
    """Map screenshot pixels to the Appium gesture viewport."""

    def __init__(self, driver: ScreenDriver) -> None:
        self._driver = driver
        self._screenshot_size: tuple[int, int] | None = None
        self._driver_size: tuple[int, int] | None = None

    def capture(self) -> bytes:
        """Capture PNG bytes and update coordinate calibration."""
        try:
            png = self._driver.get_screenshot_as_png()
        except Exception as error:  # noqa: BLE001 - external driver boundary
            raise DeviceInteractionError("Could not capture a device screenshot.") from error
        size = png_size(png)
        if size is None:
            raise DeviceInteractionError("The driver returned an invalid PNG screenshot.")
        window = self._driver.get_window_size()
        width = window.get("width")
        height = window.get("height")
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            raise DeviceInteractionError("The driver returned an invalid window size.")
        self._screenshot_size = size
        self._driver_size = (width, height)
        return png

    def screenshot_size(self) -> tuple[int, int]:
        if self._screenshot_size is None:
            self.capture()
        if self._screenshot_size is None:
            raise DeviceInteractionError("Screenshot geometry is unavailable after capture.")
        return self._screenshot_size

    def calibration(self) -> ScreenCalibration:
        """Return geometry from the most recent screenshot capture."""
        screenshot_width, screenshot_height = self.screenshot_size()
        if self._driver_size is None:
            raise DeviceInteractionError("Driver geometry is unavailable after capture.")
        driver_width, driver_height = self._driver_size
        return ScreenCalibration(
            screenshot_width_px=screenshot_width,
            screenshot_height_px=screenshot_height,
            driver_width_px=driver_width,
            driver_height_px=driver_height,
        )

    def driver_point(self, x: int, y: int) -> tuple[int, int]:
        width, height = self.screenshot_size()
        if x < 0 or y < 0 or x >= width or y >= height:
            raise DeviceInteractionError(
                f"Point ({x}, {y}) is outside the {width}x{height} screenshot."
            )
        if self._driver_size is None:
            raise DeviceInteractionError("Driver geometry is unavailable after capture.")
        driver_width, driver_height = self._driver_size
        return round(x * driver_width / width), round(y * driver_height / height)

    def validate_region(self, left: int, top: int, right: int, bottom: int) -> None:
        width, height = self.screenshot_size()
        if left < 0 or top < 0 or right >= width or bottom >= height:
            raise DeviceInteractionError("The scroll region is outside the screenshot.")
        if left >= right or top >= bottom:
            raise DeviceInteractionError("The scroll region must have positive width and height.")


def png_size(png: bytes) -> tuple[int, int] | None:
    """Return PNG dimensions from its fixed header."""
    if len(png) < 24 or png[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    width = int.from_bytes(png[16:20], "big")
    height = int.from_bytes(png[20:24], "big")
    return (width, height) if width > 0 and height > 0 else None


def is_blank(png: bytes, threshold: float) -> bool:
    """Return whether luminance variation stays below the blank threshold."""
    try:
        with Image.open(BytesIO(png)) as image:
            standard_deviation = ImageStat.Stat(image.convert("L")).stddev[0]
    except (OSError, UnidentifiedImageError) as error:
        raise DeviceInteractionError("Could not decode the PNG screenshot.") from error
    return standard_deviation <= threshold
