"""Screenshot capture, geometry calibration, and blank-frame analysis."""

from __future__ import annotations

from math import ceil, floor

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from io import BytesIO
import time
from typing import TYPE_CHECKING, Protocol

from PIL import Image, ImageStat, UnidentifiedImageError
from selenium.common.exceptions import (
    InvalidSessionIdException,
    TimeoutException,
    WebDriverException,
)

from yewo.devices.errors import DeviceActionRejectedError, DeviceInteractionError

if TYPE_CHECKING:
    from yewo.run_narration import RunNarrator


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


def _current_narrator() -> RunNarrator | None:
    """Return the active narrator, importing the narrator module on use.

    Deferred because `run_narration` reaches the artifact records, those reach device records,
    and this module is one of them. A module-level import closes that loop, and the loop only
    resolves when some other module happens to import first.
    """
    from yewo.run_narration import current_narrator

    return current_narrator()


def _exception_detail(error: Exception, *, maximum_length: int = 1_000) -> str:
    """Retain bounded external-driver detail for run diagnostics."""
    driver_message = getattr(error, "msg", None)
    detail = driver_message if isinstance(driver_message, str) else str(error)
    escaped_detail = detail.replace("\r", "\\r").replace("\n", "\\n").strip()
    if not escaped_detail:
        return "No exception message was provided."
    if len(escaped_detail) <= maximum_length:
        return escaped_detail
    return f"{escaped_detail[: maximum_length - 3]}..."


def _is_retryable_screenshot_failure(error: Exception) -> bool:
    """Return whether a failed read can safely be repeated on the same session."""
    if isinstance(error, InvalidSessionIdException | TimeoutException):
        return False
    return isinstance(error, ConnectionError | WebDriverException)


class ScreenGeometry:
    """Map screenshot pixels to the Appium gesture viewport."""

    def __init__(
        self,
        driver: ScreenDriver,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        maximum_screenshot_attempts: int = 3,
        screenshot_retry_interval_seconds: float = 0.25,
    ) -> None:
        if maximum_screenshot_attempts < 1:
            raise ValueError("The maximum screenshot attempt count must be positive.")
        if screenshot_retry_interval_seconds < 0:
            raise ValueError("The screenshot retry interval cannot be negative.")
        self._driver = driver
        self._monotonic = monotonic
        self._sleep = sleep
        self._maximum_screenshot_attempts = maximum_screenshot_attempts
        self._screenshot_retry_interval_seconds = screenshot_retry_interval_seconds
        self._screenshot_size: tuple[int, int] | None = None
        self._driver_size: tuple[int, int] | None = None

    def capture(self) -> bytes:
        """Capture PNG bytes and update coordinate calibration."""
        png = self._capture_png()
        size = png_size(png)
        if size is None:
            raise DeviceInteractionError("The driver returned an invalid PNG screenshot.")
        window_started = self._monotonic()
        try:
            window = self._driver.get_window_size()
        finally:
            self._trace_duration("window-size", window_started)
        width = window.get("width")
        height = window.get("height")
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            raise DeviceInteractionError("The driver returned an invalid window size.")
        self._screenshot_size = size
        self._driver_size = (width, height)
        return png

    def _capture_png(self) -> bytes:
        """Retry one read-only screenshot operation without redispatching an action."""
        last_error: Exception | None = None
        for attempt_number in range(1, self._maximum_screenshot_attempts + 1):
            if attempt_number > 1 and self._screenshot_retry_interval_seconds:
                self._sleep(self._screenshot_retry_interval_seconds)
            try:
                return self._capture_png_once(attempt_number)
            except Exception as error:  # noqa: BLE001 - external driver boundary
                if not _is_retryable_screenshot_failure(error):
                    raise DeviceInteractionError(
                        "Could not capture a device screenshot."
                    ) from error
                last_error = error
        raise DeviceInteractionError("Could not capture a device screenshot.") from last_error

    def _capture_png_once(self, attempt_number: int) -> bytes:
        screenshot_started = self._monotonic()
        try:
            return self._driver.get_screenshot_as_png()
        except Exception as error:  # noqa: BLE001 - external driver boundary
            narrator = _current_narrator()
            if narrator is not None:
                narrator.trace(
                    "appium-command",
                    "operation=screenshot state=failed "
                    f"attempt={attempt_number}/{self._maximum_screenshot_attempts} "
                    f"error={type(error).__name__} detail={_exception_detail(error)}",
                )
            raise
        finally:
            self._trace_duration("screenshot", screenshot_started)

    def _trace_duration(self, operation: str, started: float) -> None:
        narrator = _current_narrator()
        if narrator is not None:
            from yewo.run_narration import elapsed_milliseconds

            duration_ms = elapsed_milliseconds(started, self._monotonic())
            narrator.trace(
                "appium-command",
                f"operation={operation} duration_ms={duration_ms:.1f}",
            )

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
            raise DeviceActionRejectedError(
                f"Point ({x}, {y}) is outside the {width}x{height} screenshot."
            )
        if self._driver_size is None:
            raise DeviceInteractionError("Driver geometry is unavailable after capture.")
        driver_width, driver_height = self._driver_size
        # Rounding can carry the largest accepted screenshot point one step past
        # the driver edge when the spaces differ. Keep the result addressable.
        return (
            min(round(x * driver_width / width), driver_width - 1),
            min(round(y * driver_height / height), driver_height - 1),
        )

    def driver_region(
        self,
        left: int,
        top: int,
        right: int,
        bottom: int,
    ) -> tuple[int, int, int, int]:
        """Map an inclusive screenshot rectangle to an inclusive driver rectangle.

        Converting the two corner points with driver_point loses a pixel of extent, because the
        inclusive right edge is one short of the rectangle's width. The exclusive edge is
        converted instead, and driver_point refuses it as a point outside the screenshot.
        """
        self.validate_region(left, top, right, bottom)
        width, height = self.screenshot_size()
        if self._driver_size is None:
            raise DeviceInteractionError("Driver geometry is unavailable after capture.")
        driver_width, driver_height = self._driver_size
        driver_left = min(floor(left * driver_width / width), driver_width - 1)
        driver_top = min(floor(top * driver_height / height), driver_height - 1)
        return (
            driver_left,
            driver_top,
            max(driver_left, min(ceil((right + 1) * driver_width / width) - 1, driver_width - 1)),
            max(
                driver_top, min(ceil((bottom + 1) * driver_height / height) - 1, driver_height - 1)
            ),
        )

    def validate_region(self, left: int, top: int, right: int, bottom: int) -> None:
        width, height = self.screenshot_size()
        if left < 0 or top < 0 or right >= width or bottom >= height:
            raise DeviceActionRejectedError("The scroll region is outside the screenshot.")
        if left >= right or top >= bottom:
            raise DeviceActionRejectedError(
                "The scroll region must have positive width and height."
            )


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
