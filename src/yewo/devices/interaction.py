"""Validated cross-platform application interaction."""

from __future__ import annotations

from typing import Self, cast

from selenium.common.exceptions import NoAlertPresentException

from yewo.applications.records import ApplicationDescriptor
from yewo.automation.session import AutomationSession
from yewo.devices.application_interaction import ApplicationInteraction
from yewo.devices.application_lifecycle import (
    ApplicationState,
    DEFAULT_MAXIMUM_BACKGROUND_SECONDS,
)
from yewo.devices.errors import (
    DeviceActionRejectedError,
    DeviceInputRejectedError,
    DeviceInteractionError,
)
from yewo.devices.focus import (
    ANDROID_FOCUS_TIMEOUT_SECONDS,
    DEFAULT_FOCUS_POLL_SECONDS,
    DEFAULT_FOCUS_TIMEOUT_SECONDS,
    FocusService,
    FocusedFieldReading,
    TextElement,
    text_field_context,
)
from yewo.devices.interaction_contracts import (
    InteractionClock,
    InteractionDriver,
    PlatformInteractionBackend,
    SystemInteractionClock,
)
from yewo.devices.inventory import DeviceKind, DevicePlatform
from yewo.devices.interaction_records import AndroidKey, TextInputConstraint
from yewo.devices.screen import (
    ScreenCalibration,
    ScreenGeometry,
    ScreenRegionReading,
)
from yewo.devices.scroll import ScrollDirection, scroll_gesture, validate_scroll_distance
from yewo.observations.service import ScreenCapture


MINIMUM_WAIT_SECONDS = 0.1
DEFAULT_MAXIMUM_WAIT_SECONDS = 30.0
MINIMUM_LONG_PRESS_SECONDS = 0.1
MINIMUM_ANDROID_DRAG_SECONDS = 0.1
MINIMUM_IOS_DRAG_SECONDS = 0.5
MAXIMUM_GESTURE_SECONDS = 10.0
# Android's aggregate Ctrl mask accepted by UiAutomator2 for Ctrl+A.
ANDROID_SELECT_ALL_META_STATE = 0x7000
_REFRESH_START_FRACTION = 0.1
_REFRESH_TRAVEL_FRACTION = 0.6
_REFRESH_SECONDS = 0.6


class DeviceInteraction:
    """Validate and dispatch UI and application lifecycle operations."""

    def __init__(
        self,
        driver: InteractionDriver,
        platform: DevicePlatform,
        application: ApplicationDescriptor,
        backend: PlatformInteractionBackend,
        *,
        device_kind: DeviceKind | None = None,
        maximum_wait_seconds: float = DEFAULT_MAXIMUM_WAIT_SECONDS,
        maximum_background_seconds: int = DEFAULT_MAXIMUM_BACKGROUND_SECONDS,
        application_transition_timeout_seconds: float = 10,
        application_poll_interval_seconds: float = 0.25,
        focus_timeout_seconds: float | None = None,
        focus_poll_seconds: float = DEFAULT_FOCUS_POLL_SECONDS,
        clock: InteractionClock | None = None,
    ) -> None:
        self._driver = driver
        self.platform = platform
        self.application = application
        self._backend = backend
        self._device_kind = device_kind
        self._maximum_wait_seconds = maximum_wait_seconds
        self._clock = clock or SystemInteractionClock()
        self._screen = ScreenGeometry(driver)
        self._focus = FocusService(
            driver,
            self._screen,
            self._clock,
            self.tap,
            timeout_seconds=(
                focus_timeout_seconds
                if focus_timeout_seconds is not None
                else (
                    ANDROID_FOCUS_TIMEOUT_SECONDS
                    if platform is DevicePlatform.ANDROID
                    else DEFAULT_FOCUS_TIMEOUT_SECONDS
                )
            ),
            poll_seconds=focus_poll_seconds,
        )
        self._application_interaction = ApplicationInteraction(
            driver,
            platform,
            device_kind,
            application,
            backend,
            self._clock,
            self.screenshot,
            transition_timeout_seconds=application_transition_timeout_seconds,
            poll_interval_seconds=application_poll_interval_seconds,
            maximum_background_seconds=maximum_background_seconds,
        )

    @classmethod
    def from_session(cls, session: AutomationSession) -> Self:
        """Create interaction with the adapter for the session platform."""
        platform = session.device.descriptor.platform
        if platform is DevicePlatform.ANDROID:
            from yewo.devices.android.interaction import AndroidInteractionBackend
            from yewo.devices.android.adb import AdbClient

            descriptor = session.device.descriptor
            backend: PlatformInteractionBackend = AndroidInteractionBackend(
                AdbClient(), descriptor.connection_id or descriptor.target_id
            )
        else:
            from yewo.devices.ios.interaction import IosInteractionBackend

            backend = IosInteractionBackend()
        return cls(
            cast(InteractionDriver, session.driver.driver),
            platform,
            session.application,
            backend,
            device_kind=session.device.descriptor.kind,
        )

    def screenshot(self) -> ScreenCapture:
        """Capture the screen once. A device alters nothing, so both views are the same bytes."""
        png = self._screen.capture()
        return ScreenCapture(image=png, comparable=png)

    def comparable_screenshot(self) -> bytes:
        """Capture the screen for comparison. A device has nothing cheaper to offer."""
        return self._screen.capture()

    def tap(
        self,
        x: int,
        y: int,
        *,
        count: int = 1,
        interval_seconds: float = 0.1,
    ) -> None:
        if count < 1 or count > 10:
            raise DeviceActionRejectedError("Tap count must be between 1 and 10.")
        if interval_seconds < 0 or interval_seconds > 1:
            raise DeviceActionRejectedError("Tap interval must be between 0 and 1 second.")
        point = self._driver_point(x, y)
        for index in range(count):
            self._backend.tap(self._driver, *point)
            if index + 1 < count:
                self._clock.sleep(interval_seconds)

    def double_tap(self, x: int, y: int) -> None:
        self._backend.double_tap(self._driver, *self._driver_point(x, y))

    def long_press(self, x: int, y: int, duration_seconds: float) -> None:
        if (
            duration_seconds < MINIMUM_LONG_PRESS_SECONDS
            or duration_seconds > MAXIMUM_GESTURE_SECONDS
        ):
            raise DeviceActionRejectedError(
                "Long-press duration must be between 0.1 and 10 seconds."
            )
        self._backend.long_press(
            self._driver,
            *self._driver_point(x, y),
            duration_seconds,
        )

    def drag(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        *,
        duration_seconds: float = 1,
    ) -> None:
        self._validate_gesture_duration(duration_seconds)
        self._backend.drag(
            self._driver,
            self._driver_point(start_x, start_y),
            self._driver_point(end_x, end_y),
            duration_seconds,
        )

    def refresh(
        self,
        *,
        left: int = 0,
        top: int = 0,
        right: int | None = None,
        bottom: int | None = None,
    ) -> None:
        """Pull one region downward far enough to request a content refresh."""
        width, height = self._screen_size()
        right = width - 1 if right is None else right
        bottom = height - 1 if bottom is None else bottom
        self._validate_region(left, top, right, bottom)
        center_x = (left + right) // 2
        span = bottom - top
        # The pull starts near the top edge, because a refresh control only appears while the
        # region is already scrolled to its start.
        start_y = top + max(1, round(span * _REFRESH_START_FRACTION))
        end_y = min(bottom, start_y + max(2, round(span * _REFRESH_TRAVEL_FRACTION)))
        self.swipe(center_x, start_y, center_x, end_y, duration_seconds=_REFRESH_SECONDS)

    def swipe(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        *,
        duration_seconds: float = 0.5,
    ) -> None:
        self.drag(
            start_x,
            start_y,
            end_x,
            end_y,
            duration_seconds=duration_seconds,
        )

    def scroll(
        self,
        direction: ScrollDirection,
        *,
        left: int = 0,
        top: int = 0,
        right: int | None = None,
        bottom: int | None = None,
        distance_fraction: float = 0.6,
    ) -> bool | None:
        """Scroll and report whether the surface can scroll further, when known."""
        width, height = self._screen_size()
        right = width - 1 if right is None else right
        bottom = height - 1 if bottom is None else bottom
        validate_scroll_distance(distance_fraction)
        # Converted like every other gesture region. A backend receives driver coordinates, and
        # the screenshot and the Appium gesture viewport differ whenever the device scales.
        driver_left, driver_top, driver_right, driver_bottom = self._screen.driver_region(
            left, top, right, bottom
        )
        dispatch = self._backend.scroll(
            self._driver,
            left=driver_left,
            top=driver_top,
            right=driver_right,
            bottom=driver_bottom,
            direction=direction,
            distance_fraction=distance_fraction,
        )
        if not dispatch.dispatched:
            self.swipe(*scroll_gesture(direction, left, top, right, bottom, distance_fraction))
        return dispatch.can_scroll_more

    def type_text(self, x: int, y: int, text: str) -> bool | None:
        return self._type_in_field(self.focus(x, y), text)

    def type_focused_text(self, text: str) -> bool | None:
        return self._type_in_field(self._focus.focused_field(), text)

    def _type_in_field(self, field: TextElement, text: str) -> bool | None:
        constraint = self._backend.focused_text_constraint(self._driver)
        if constraint is not None and not constraint.accepts(text):
            raise DeviceInputRejectedError(f"{constraint.describe()} Choose a compatible value.")
        field_context = text_field_context(field)
        before = None if field_context.is_secure is True else self._text_value(field)
        field.send_keys(text)
        if field_context.is_secure is True:
            return None
        after = self._text_value(field)
        if after is None:
            return None
        return after == text or (before is not None and after == before + text)

    def clear_text(self, x: int, y: int) -> bool | None:
        return self._clear_field(self.focus(x, y))

    def clear_focused_text(self) -> bool | None:
        return self._clear_field(self._focus.focused_field())

    def _clear_field(self, field: TextElement) -> bool | None:
        self._backend.clear_text(self._driver, field)
        value = self._text_value(field)
        return value == "" if value is not None else None

    def focus(self, x: int, y: int) -> TextElement:
        """Focus the field at one validated screenshot point."""
        return self._focus.focus(x, y)

    def focused_field_state(self) -> bool | None:
        """Return whether the active element reports keyboard focus."""
        value = self._driver.switch_to.active_element.get_attribute("focused")
        return _boolean_attribute(value)

    def focused_field_reading(self) -> FocusedFieldReading:
        """Return secret-free focus and keyboard evidence."""
        return self._focus.reading()

    def _text_value(self, field: TextElement) -> str | None:
        names = ("value", "text") if self.platform is DevicePlatform.IOS else ("text",)
        values = tuple(field.get_attribute(name) for name in names)
        return next((value for value in values if value is not None), None)

    def press_key(self, key: AndroidKey | str) -> None:
        if self.platform is DevicePlatform.ANDROID:
            if not isinstance(key, AndroidKey):
                raise DeviceActionRejectedError("Android keys must use a supported AndroidKey.")
            self._driver.press_keycode(key.value)
            return
        if not isinstance(key, str) or not key:
            raise DeviceActionRejectedError("An iOS button name must be nonempty.")
        supported_buttons = self._ios_buttons()
        if key not in supported_buttons:
            names = ", ".join(sorted(supported_buttons)) or "none"
            raise DeviceActionRejectedError(f"The device supports these iOS buttons: {names}.")
        self._driver.execute_script("mobile: pressButton", {"name": key})

    def _ios_buttons(self) -> set[str]:
        try:
            answer = self._driver.execute_script("mobile: getPressableButtons", {})
        except Exception as error:  # noqa: BLE001 - external driver boundary
            raise DeviceInteractionError("Could not read the supported iOS buttons.") from error
        if not isinstance(answer, dict):
            raise DeviceInteractionError("The iOS driver did not report pressable buttons.")
        values = answer.get("xctest")
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise DeviceInteractionError("The iOS driver returned invalid pressable buttons.")
        return set(values)

    def back(self) -> None:
        if self.platform is not DevicePlatform.ANDROID:
            raise DeviceActionRejectedError("Back navigation is available only on Android.")
        self._driver.back()

    def submit(self) -> None:
        if self.platform is DevicePlatform.ANDROID:
            self._driver.press_keycode(AndroidKey.ENTER.value)
        else:
            self._driver.switch_to.active_element.send_keys("\n")

    def keyboard_visible(self) -> bool | None:
        try:
            return self._driver.is_keyboard_shown()
        except Exception:  # noqa: BLE001 - an unavailable read is unknown
            return None

    def dismiss_keyboard(self) -> bool:
        try:
            return self._backend.dismiss_keyboard(self._driver)
        except Exception:  # noqa: BLE001 - verification determines the result
            return False

    def wait(self, duration_seconds: float) -> None:
        if duration_seconds < MINIMUM_WAIT_SECONDS or duration_seconds > self._maximum_wait_seconds:
            raise DeviceActionRejectedError(
                "Wait duration must be between "
                f"{MINIMUM_WAIT_SECONDS:g} and {self._maximum_wait_seconds:g} seconds."
            )
        self._clock.sleep(duration_seconds)

    def application_state(self) -> ApplicationState:
        return self._application_interaction.state()

    def is_installed(self) -> bool:
        return self._application_interaction.is_installed()

    def is_foreground(self) -> bool:
        return self._application_interaction.is_foreground()

    def activate(self) -> None:
        self._application_interaction.activate()

    def terminate(self) -> None:
        self._application_interaction.terminate()

    def clear_application_data(self) -> None:
        self._application_interaction.clear_data()

    def relaunch(self) -> None:
        self._application_interaction.relaunch()

    def background_and_resume(self, duration_seconds: int) -> None:
        self._application_interaction.background_and_resume(duration_seconds)

    def open_deep_link(self, url: str) -> None:
        self._application_interaction.open_deep_link(url)

    def recover_to_application(self) -> bool:
        return self._application_interaction.recover()

    def active_application_id(self) -> str | None:
        """Return the platform identity of the application receiving input."""
        return self._application_interaction.active_application_id()

    def application_context(self) -> str | None:
        """Return the platform activity or scene identity when available."""
        return self._backend.application_context(self._driver)

    def device_orientation(self) -> str | None:
        """Return the normalized platform orientation value when available."""
        return self._backend.device_orientation(self._driver)

    def screen_calibration(self) -> ScreenCalibration:
        """Return geometry from the most recent screenshot capture."""
        return self._screen.calibration()

    def system_regions(self) -> tuple[ScreenRegionReading, ...] | None:
        """Return platform-owned rectangles with explicit coordinate spaces."""
        calibration = self.screen_calibration()
        return self._backend.system_regions(self._driver, calibration)

    def is_application_active(self) -> bool | None:
        return self._application_interaction.is_active()

    def showing_external_ui(self) -> bool | None:
        """Return whether another application currently receives input."""
        return self._application_interaction.showing_external_ui()

    def system_prompt_visible(self) -> bool | None:
        """Return whether WebDriver exposes a native alert."""
        try:
            _ = self._driver.switch_to.alert
        except NoAlertPresentException:
            return False
        except Exception:  # noqa: BLE001 - an unavailable alert read is unknown
            return None
        return True

    def wait_for_first_paint(
        self,
        *,
        timeout_seconds: float = 15,
        poll_interval_seconds: float = 0.3,
        blank_standard_deviation: float = 2,
    ) -> bytes:
        return self._application_interaction.wait_for_first_paint(
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            blank_standard_deviation=blank_standard_deviation,
        )

    def _screen_size(self) -> tuple[int, int]:
        return self._screen.screenshot_size()

    def _driver_point(self, x: int, y: int) -> tuple[int, int]:
        return self._screen.driver_point(x, y)

    def _validate_region(self, left: int, top: int, right: int, bottom: int) -> None:
        self._screen.validate_region(left, top, right, bottom)

    def _validate_gesture_duration(self, duration_seconds: float) -> None:
        minimum = (
            MINIMUM_IOS_DRAG_SECONDS
            if self.platform is DevicePlatform.IOS
            else MINIMUM_ANDROID_DRAG_SECONDS
        )
        if duration_seconds < minimum or duration_seconds > MAXIMUM_GESTURE_SECONDS:
            raise DeviceActionRejectedError(
                f"Gesture duration must be between {minimum:g} and 10 seconds."
            )


def _boolean_attribute(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized_value = value.strip().lower()
    if normalized_value in {"true", "1"}:
        return True
    if normalized_value in {"false", "0"}:
        return False
    return None


# Re-exported because callers address the interaction boundary, not the records and gesture
# modules behind it.
__all__ = [
    "ANDROID_SELECT_ALL_META_STATE",
    "DEFAULT_MAXIMUM_WAIT_SECONDS",
    "MAXIMUM_GESTURE_SECONDS",
    "MINIMUM_ANDROID_DRAG_SECONDS",
    "MINIMUM_IOS_DRAG_SECONDS",
    "MINIMUM_LONG_PRESS_SECONDS",
    "MINIMUM_WAIT_SECONDS",
    "AndroidKey",
    "DeviceInteraction",
    "InteractionDriver",
    "PlatformInteractionBackend",
    "ScrollDirection",
    "TextInputConstraint",
]
