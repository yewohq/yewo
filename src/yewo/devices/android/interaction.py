"""Android gesture dispatch through UiAutomator2 extensions."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Protocol

from yewo.devices.application_lifecycle import ApplicationLifecycleDriver
from yewo.devices.focus import TextElement
from yewo.devices.interaction import (
    ANDROID_SELECT_ALL_META_STATE,
    AndroidKey,
    InteractionDriver,
    TextInputConstraint,
)
from yewo.devices.screen import (
    ScreenCalibration,
    ScreenCoordinateSpace,
    ScreenRegionReading,
    SystemRegionKind,
    bounded_rectangle,
)


_CURRENT_INPUT_TYPE_PATTERN = re.compile(
    r"\bmCurAttribute=EditorInfo\{[^\n]*?\binputType=(?:0x)?([0-9A-Fa-f]+)"
)
_START_INPUT_HISTORY_PATTERN = re.compile(
    r"^\s*mStartInputHistory:\s*$\n(?P<history>.*?)(?=^\s*mSoftInputShowHideHistory:)",
    re.MULTILINE | re.DOTALL,
)
_HISTORY_INPUT_TYPE_PATTERN = re.compile(r"^\s*inputType=0x([0-9A-Fa-f]+)\b", re.MULTILINE)
_INPUT_SHOWN_PATTERN = re.compile(r"\bmInputShown=(true|false)\b")
_INPUT_VIEW_SHOWN_PATTERN = re.compile(r"\bmIsInputViewShown=(true|false)\b")
_CURRENT_FOCUS_PACKAGE_PATTERN = re.compile(
    r"^\s*mCurrentFocus=Window\{[^\n]*?\s"
    r"(?P<package>[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)+)(?:/[^\s}]+)?",
    re.MULTILINE,
)
_CURRENT_FOCUS_TITLE_PATTERN = re.compile(
    r"^\s*mCurrentFocus=Window\{[^\n]*?\su\d+\s(?P<title>[A-Za-z][A-Za-z0-9_]*)\}",
    re.MULTILINE,
)
_SYSTEM_WINDOW_APPLICATION_IDS = {
    "NotificationShade": "com.android.systemui",
    "StatusBar": "com.android.systemui",
}
_INPUT_TYPE_CLASS_MASK = 0xF
_INPUT_TYPE_CLASS_NUMBER = 0x2
_INPUT_TYPE_NUMBER_SIGNED = 0x1000
_INPUT_TYPE_NUMBER_DECIMAL = 0x2000


class _AndroidInputReader(Protocol):
    def ask(self, connection_id: str, *args: str, timeout_seconds: float = 20) -> str | None: ...


class AndroidInteractionBackend:
    """Dispatch Android interactions and read platform context."""

    def __init__(
        self,
        input_reader: _AndroidInputReader | None = None,
        connection_id: str | None = None,
    ) -> None:
        self._input_reader = input_reader
        self._connection_id = connection_id

    def dismiss_keyboard(self, driver: InteractionDriver) -> bool:
        """Dismiss the IME without issuing unguarded Android back navigation."""
        if not driver.is_keyboard_shown():
            return True
        source_context = self.application_context(driver)
        if source_context is None:
            return False
        if self._keyboard_back_is_authorized():
            driver.press_keycode(AndroidKey.BACK.value)
            if not driver.is_keyboard_shown():
                return _context_is_unchanged(self.application_context(driver), source_context)
        if not _context_is_unchanged(self.application_context(driver), source_context):
            return False
        if not driver.is_keyboard_shown():
            return True
        driver.hide_keyboard()
        return not driver.is_keyboard_shown() and _context_is_unchanged(
            self.application_context(driver), source_context
        )

    def _keyboard_back_is_authorized(self) -> bool:
        if self._input_reader is None or self._connection_id is None:
            return False
        output = self._input_reader.ask(
            self._connection_id,
            "dumpsys",
            "input_method",
            timeout_seconds=3,
        )
        return _keyboard_back_is_authorized(output or "")

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

    def clear_text(self, driver: InteractionDriver, element: TextElement) -> None:
        """Clear the focused Android field with a WebDriver fallback."""
        try:
            driver.press_keycode(AndroidKey.A, metastate=ANDROID_SELECT_ALL_META_STATE)
            driver.press_keycode(AndroidKey.DELETE)
            remaining = element.get_attribute("text")
            if remaining:
                element.clear()
        except Exception:  # noqa: BLE001 - platform fallback owns driver failures
            element.clear()

    def focused_text_constraint(
        self,
        driver: InteractionDriver,
    ) -> TextInputConstraint | None:
        """Read Android's current editor type without reading the field value."""
        del driver
        if self._input_reader is None or self._connection_id is None:
            return None
        output = self._input_reader.ask(
            self._connection_id,
            "dumpsys",
            "input_method",
            timeout_seconds=3,
        )
        input_type = _current_input_type(output or "")
        if input_type is None:
            return None
        if input_type & _INPUT_TYPE_CLASS_MASK != _INPUT_TYPE_CLASS_NUMBER:
            return TextInputConstraint(is_numeric=False)
        return TextInputConstraint(
            is_numeric=True,
            allows_sign=bool(input_type & _INPUT_TYPE_NUMBER_SIGNED),
            allows_decimal=bool(input_type & _INPUT_TYPE_NUMBER_DECIMAL),
        )

    def active_application_id(
        self,
        driver: ApplicationLifecycleDriver,
    ) -> str | None:
        focused = self._focused_window_application_id()
        if focused is not None:
            return focused
        try:
            value = getattr(driver, "current_package", None)
        except Exception:  # noqa: BLE001 - an unavailable read is unknown
            return None
        return value if isinstance(value, str) and value else None

    def _focused_window_application_id(self) -> str | None:
        """Read the package that owns Android's focused window."""
        if self._input_reader is None or self._connection_id is None:
            return None
        output = self._input_reader.ask(
            self._connection_id,
            "dumpsys",
            "window",
            timeout_seconds=3,
        )
        match = _CURRENT_FOCUS_PACKAGE_PATTERN.search(output or "")
        if match is not None:
            return match.group("package")
        title_match = _CURRENT_FOCUS_TITLE_PATTERN.search(output or "")
        return (
            _SYSTEM_WINDOW_APPLICATION_IDS.get(title_match.group("title"))
            if title_match is not None
            else None
        )

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


def _current_input_type(output: str) -> int | None:
    current = _CURRENT_INPUT_TYPE_PATTERN.search(output)
    if current is not None:
        return int(current.group(1), 16)
    history = _START_INPUT_HISTORY_PATTERN.search(output)
    if history is None:
        return None
    entries = _HISTORY_INPUT_TYPE_PATTERN.findall(history.group("history"))
    return int(entries[-1], 16) if entries else None


def _distance(start: tuple[int, int], end: tuple[int, int]) -> float:
    return ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5


def _context_is_unchanged(current: str | None, source: str | None) -> bool:
    return source is not None and current == source


def _keyboard_back_is_authorized(output: str) -> bool:
    input_shown = _INPUT_SHOWN_PATTERN.search(output)
    input_view_shown = _INPUT_VIEW_SHOWN_PATTERN.search(output)
    return (
        input_shown is not None
        and input_shown.group(1) == "true"
        and input_view_shown is not None
        and input_view_shown.group(1) == "true"
    )


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
    elif normalized_name == "ime" or any(
        token in normalized_name for token in ("keyboard", "inputmethod")
    ):
        kind = SystemRegionKind.KEYBOARD
    else:
        kind = SystemRegionKind.OTHER
    return ScreenRegionReading(kind, ScreenCoordinateSpace.DRIVER, *bounds)
