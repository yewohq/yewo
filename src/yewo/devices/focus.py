"""Bounded, secret-free focused-field verification."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from yewo.devices.errors import DeviceTargetRejectedError
from yewo.devices.screen import ScreenCalibration, ScreenGeometry


DEFAULT_FOCUS_TIMEOUT_SECONDS = 1.5
ANDROID_FOCUS_TIMEOUT_SECONDS = 8.0
DEFAULT_FOCUS_POLL_SECONDS = 0.05
_TEXT_INPUT_CLASSES = (
    "android.widget.EditText",
    "XCUIElementTypeTextField",
    "XCUIElementTypeSecureTextField",
    "XCUIElementTypeSearchField",
    "XCUIElementTypeTextView",
)


class TextElement(Protocol):
    @property
    def rect(self) -> dict[str, object]: ...

    def send_keys(self, value: str) -> None: ...

    def click(self) -> None: ...

    def clear(self) -> None: ...

    def get_attribute(self, name: str) -> str | None: ...


class _SwitchTo(Protocol):
    @property
    def active_element(self) -> TextElement: ...


class FocusDriver(Protocol):
    @property
    def switch_to(self) -> _SwitchTo: ...

    def is_keyboard_shown(self) -> bool: ...


class FocusClock(Protocol):
    def monotonic(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


@dataclass(frozen=True)
class FocusedFieldReading:
    """Secret-free platform evidence about the active text field."""

    is_focused: bool | None
    driver_bounds: tuple[int, int, int, int] | None
    keyboard_visible: bool | None


class FocusService:
    """Focus one grounded field and expose secret-free evidence."""

    def __init__(
        self,
        driver: FocusDriver,
        screen: ScreenGeometry,
        clock: FocusClock,
        tap: Callable[[int, int], None],
        *,
        timeout_seconds: float = DEFAULT_FOCUS_TIMEOUT_SECONDS,
        poll_seconds: float = DEFAULT_FOCUS_POLL_SECONDS,
    ) -> None:
        if timeout_seconds <= 0 or poll_seconds <= 0:
            raise ValueError("Focus timing values must be greater than zero.")
        self._driver = driver
        self._screen = screen
        self._clock = clock
        self._tap = tap
        self._timeout_seconds = timeout_seconds
        self._poll_seconds = poll_seconds

    def focus(self, x: int, y: int) -> TextElement:
        """Return the field that accepts focus at one screenshot point."""
        driver_point = self._screen.driver_point(x, y)
        calibration = self._screen.calibration()
        before = _active_element(self._driver)
        if before is not None:
            before_bounds = _element_bounds(before)
            if _boolean_attribute(before.get_attribute("focused")) is True and _is_target_field(
                before_bounds, driver_point, calibration
            ):
                return before
        target_element = _target_text_element(self._driver, driver_point, calibration)
        if target_element is not None:
            try:
                target_element.click()
            except Exception:  # noqa: BLE001 - coordinate focus remains available
                target_element = None
        if target_element is None and self._keyboard_visible() is True:
            raise DeviceTargetRejectedError(
                "The keyboard is visible, but no editable target field contains the grounded point."
            )
        if target_element is None:
            self._tap(x, y)
        deadline = self._clock.monotonic() + self._timeout_seconds
        while True:
            field = _active_element(self._driver)
            if field is None:
                field = _focused_target_element(self._driver, driver_point, calibration)
            if field is None:
                field = target_element or _target_text_element(
                    self._driver,
                    driver_point,
                    calibration,
                )
                if (
                    field is not None
                    and _boolean_attribute(field.get_attribute("focused")) is True
                    and _is_target_field(_element_bounds(field), driver_point, calibration)
                ):
                    return field
            if field is not None:
                focused = _boolean_attribute(field.get_attribute("focused"))
                bounds = _element_bounds(field)
                if focused is not False and (
                    _is_target_field(bounds, driver_point, calibration)
                    or (focused is True and bounds is None)
                ):
                    return field
            remaining_seconds = deadline - self._clock.monotonic()
            if remaining_seconds <= 0:
                raise DeviceTargetRejectedError(
                    "The grounded target field did not accept keyboard focus. It may be a label or "
                    "an editable field obscured by the keyboard."
                )
            self._clock.sleep(min(self._poll_seconds, remaining_seconds))

    def reading(self) -> FocusedFieldReading:
        """Return focus and keyboard state without the field value."""
        try:
            element = self.focused_field()
        except DeviceTargetRejectedError:
            element = None
        if element is None:
            return FocusedFieldReading(
                is_focused=None,
                driver_bounds=None,
                keyboard_visible=self._keyboard_visible(),
            )
        return FocusedFieldReading(
            is_focused=_boolean_attribute(element.get_attribute("focused")),
            driver_bounds=_element_bounds(element),
            keyboard_visible=self._keyboard_visible(),
        )

    def focused_field(self) -> TextElement:
        """Return the active editable field without changing focus."""
        active = _active_element(self._driver)
        if active is not None and _boolean_attribute(active.get_attribute("focused")) is True:
            return active
        focused = _focused_elements(self._driver)
        if focused:
            return focused[0]
        if (
            active is not None
            and _boolean_attribute(active.get_attribute("focused")) is not False
            and self._keyboard_visible() is True
        ):
            return active
        raise DeviceTargetRejectedError(
            "No editable field has verified focus. Focus a field before editing its text."
        )

    def _keyboard_visible(self) -> bool | None:
        try:
            return self._driver.is_keyboard_shown()
        except Exception:  # noqa: BLE001 - an unavailable driver read is unknown
            return None


def _active_element(driver: FocusDriver) -> TextElement | None:
    try:
        return driver.switch_to.active_element
    except Exception:  # noqa: BLE001 - Appium reports no element before first focus
        return None


def _focused_elements(driver: FocusDriver) -> tuple[TextElement, ...]:
    return _find_elements(driver, "xpath", "//*[@focused='true']")


def _text_elements(driver: FocusDriver) -> tuple[TextElement, ...]:
    return tuple(
        element
        for class_name in _TEXT_INPUT_CLASSES
        for element in _find_elements(driver, "class name", class_name)
    )


def _find_elements(
    driver: FocusDriver,
    strategy: str,
    selector: str,
) -> tuple[TextElement, ...]:
    reader = getattr(driver, "find_elements", None)
    if not callable(reader):
        return ()
    try:
        elements = reader(strategy, selector)
    except Exception:  # noqa: BLE001 - Appium element lookup is best-effort evidence
        return ()
    return tuple(elements) if isinstance(elements, list | tuple) else ()


def _focused_target_element(
    driver: FocusDriver,
    point: tuple[int, int],
    calibration: ScreenCalibration,
) -> TextElement | None:
    return _unique_target_element(
        _focused_elements(driver),
        point,
        calibration,
    )


def _target_text_element(
    driver: FocusDriver,
    point: tuple[int, int],
    calibration: ScreenCalibration,
) -> TextElement | None:
    return _unique_target_element(
        _text_elements(driver),
        point,
        calibration,
    )


def _unique_target_element(
    elements: tuple[TextElement, ...],
    point: tuple[int, int],
    calibration: ScreenCalibration,
) -> TextElement | None:
    matching = tuple(
        element
        for element in elements
        if _is_target_field(_element_bounds(element), point, calibration)
    )
    return matching[0] if len(matching) == 1 else None


def _boolean_attribute(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    return None


def _element_bounds(element: TextElement) -> tuple[int, int, int, int] | None:
    try:
        rect = element.rect
    except Exception:  # noqa: BLE001 - external driver boundary
        return None
    left = rect.get("x")
    top = rect.get("y")
    width = rect.get("width")
    height = rect.get("height")
    if not (
        isinstance(left, int | float)
        and isinstance(top, int | float)
        and isinstance(width, int | float)
        and isinstance(height, int | float)
    ):
        return None
    if width <= 0 or height <= 0:
        return None
    return (round(left), round(top), round(left + width), round(top + height))


def _contains(
    bounds: tuple[int, int, int, int] | None,
    point: tuple[int, int],
) -> bool:
    if bounds is None:
        return False
    left, top, right, bottom = bounds
    return left <= point[0] < right and top <= point[1] < bottom


def _covers_driver_screen(
    bounds: tuple[int, int, int, int],
    calibration: ScreenCalibration,
) -> bool:
    left, top, right, bottom = bounds
    area = (right - left) * (bottom - top)
    screen_area = calibration.driver_width_px * calibration.driver_height_px
    return area >= screen_area * 0.9


def _is_target_field(
    bounds: tuple[int, int, int, int] | None,
    point: tuple[int, int],
    calibration: ScreenCalibration,
) -> bool:
    return (
        bounds is not None
        and _contains(bounds, point)
        and not _covers_driver_screen(bounds, calibration)
    )
