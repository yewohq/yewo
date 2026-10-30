"""Validated cross-platform application interaction."""

from __future__ import annotations

from enum import IntEnum, StrEnum
import time
from typing import Protocol, Self, cast

from selenium.common.exceptions import NoAlertPresentException

from yewo.applications.records import ApplicationDescriptor
from yewo.automation.session import AutomationSession
from yewo.devices.application_interaction import ApplicationInteraction
from yewo.devices.application_lifecycle import (
    ApplicationLifecycleDriver,
    ApplicationState,
    DEFAULT_MAXIMUM_BACKGROUND_SECONDS,
)
from yewo.devices.errors import DeviceInteractionError
from yewo.devices.inventory import DeviceKind, DevicePlatform
from yewo.devices.screen import (
    ScreenCalibration,
    ScreenGeometry,
    ScreenRegionReading,
)



class AndroidKey(IntEnum):
    """Supported Android hardware and navigation keys."""

    HOME = 3
    BACK = 4
    DPAD_UP = 19
    DPAD_DOWN = 20
    DPAD_LEFT = 21
    DPAD_RIGHT = 22
    DPAD_CENTER = 23
    TAB = 61
    ENTER = 66
    DELETE = 67
    ESCAPE = 111


class ScrollDirection(StrEnum):
    """A semantic direction for moving visible content."""

    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


class _Element(Protocol):
    def send_keys(self, value: str) -> None: ...

    def clear(self) -> None: ...

    def get_attribute(self, name: str) -> str | None: ...


class _SwitchTo(Protocol):
    @property
    def active_element(self) -> _Element: ...

    @property
    def alert(self) -> object: ...


class _Clock(Protocol):
    def monotonic(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


class _SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class InteractionDriver(ApplicationLifecycleDriver, Protocol):
    """Driver methods used by core interaction and lifecycle operations."""

    @property
    def switch_to(self) -> _SwitchTo: ...

    def get_screenshot_as_png(self) -> bytes: ...

    def get_window_size(self) -> dict[str, int]: ...

    def is_keyboard_shown(self) -> bool: ...

    def hide_keyboard(self) -> object: ...

    def press_keycode(self, keycode: int, metastate: int | None = None) -> object: ...

    def back(self) -> None: ...


class PlatformInteractionBackend(Protocol):
    """Dispatch platform-specific interaction and context operations."""

    def tap(self, driver: InteractionDriver, x: int, y: int) -> None: ...

    def double_tap(self, driver: InteractionDriver, x: int, y: int) -> None: ...

    def long_press(
        self, driver: InteractionDriver, x: int, y: int, duration_seconds: float
    ) -> None: ...

    def drag(
        self,
        driver: InteractionDriver,
        start: tuple[int, int],
        end: tuple[int, int],
        duration_seconds: float,
    ) -> None: ...

    def active_application_id(
        self,
        driver: ApplicationLifecycleDriver,
    ) -> str | None: ...

    def application_context(
        self,
        driver: ApplicationLifecycleDriver,
    ) -> str | None: ...

    def device_orientation(
        self,
        driver: ApplicationLifecycleDriver,
    ) -> str | None: ...

    def system_regions(
        self,
        driver: ApplicationLifecycleDriver,
        calibration: ScreenCalibration,
    ) -> tuple[ScreenRegionReading, ...] | None: ...

    def clear_application_data(
        self,
        driver: ApplicationLifecycleDriver,
        application_id: str,
    ) -> None: ...

    def open_deep_link(
        self,
        driver: ApplicationLifecycleDriver,
        application_id: str,
        url: str,
    ) -> None: ...


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
        maximum_wait_seconds: float = 30,
        maximum_background_seconds: int = DEFAULT_MAXIMUM_BACKGROUND_SECONDS,
        application_transition_timeout_seconds: float = 10,
        application_poll_interval_seconds: float = 0.25,
        clock: _Clock | None = None,
    ) -> None:
        self._driver = driver
        self.platform = platform
        self.application = application
        self._backend = backend
        self._device_kind = device_kind
        self._maximum_wait_seconds = maximum_wait_seconds
        self._clock = clock or _SystemClock()
        self._screen = ScreenGeometry(driver)
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

            backend: PlatformInteractionBackend = AndroidInteractionBackend()
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

    def screenshot(self) -> bytes:
        """Capture PNG bytes and calibrate screenshot-to-driver coordinates."""
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
            raise DeviceInteractionError("Tap count must be between 1 and 10.")
        if interval_seconds < 0 or interval_seconds > 1:
            raise DeviceInteractionError("Tap interval must be between 0 and 1 second.")
        point = self._driver_point(x, y)
        for index in range(count):
            self._backend.tap(self._driver, *point)
            if index + 1 < count:
                self._clock.sleep(interval_seconds)

    def double_tap(self, x: int, y: int) -> None:
        self._backend.double_tap(self._driver, *self._driver_point(x, y))

    def long_press(self, x: int, y: int, duration_seconds: float) -> None:
        if duration_seconds < 0.1 or duration_seconds > 10:
            raise DeviceInteractionError("Long-press duration must be between 0.1 and 10 seconds.")
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
    ) -> None:
        width, height = self._screen_size()
        right = width - 1 if right is None else right
        bottom = height - 1 if bottom is None else bottom
        self._validate_region(left, top, right, bottom)
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2
        inset_x = max(1, (right - left) // 5)
        inset_y = max(1, (bottom - top) // 5)
        points = {
            ScrollDirection.UP: (center_x, bottom - inset_y, center_x, top + inset_y),
            ScrollDirection.DOWN: (center_x, top + inset_y, center_x, bottom - inset_y),
            ScrollDirection.LEFT: (right - inset_x, center_y, left + inset_x, center_y),
            ScrollDirection.RIGHT: (left + inset_x, center_y, right - inset_x, center_y),
        }
        self.swipe(*points[direction])

    def type_text(self, x: int, y: int, text: str) -> str | None:
        field = self.focus(x, y)
        field.send_keys(text)
        return field.get_attribute("value") or field.get_attribute("text")

    def clear_text(self, x: int, y: int) -> None:
        self.focus(x, y).clear()

    def focus(self, x: int, y: int) -> _Element:
        """Focus the field at one validated screenshot point."""
        self.tap(x, y)
        field = self._driver.switch_to.active_element
        focused = _boolean_attribute(field.get_attribute("focused"))
        if focused is False:
            raise DeviceInteractionError("The active element did not accept keyboard focus.")
        return field

    def focused_field_state(self) -> bool | None:
        """Return whether the active element reports keyboard focus."""
        value = self._driver.switch_to.active_element.get_attribute("focused")
        return _boolean_attribute(value)

    def press_key(self, key: AndroidKey | str) -> None:
        if self.platform is DevicePlatform.ANDROID:
            if not isinstance(key, AndroidKey):
                raise DeviceInteractionError("Android keys must use a supported AndroidKey.")
            self._driver.press_keycode(key.value)
            return
        if not isinstance(key, str) or not key:
            raise DeviceInteractionError("An iOS button name must be nonempty.")
        supported = self._ios_buttons()
        if key not in supported:
            names = ", ".join(sorted(supported)) or "none"
            raise DeviceInteractionError(f"The device supports these iOS buttons: {names}.")
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
            raise DeviceInteractionError("Back navigation is available only on Android.")
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
        if self.keyboard_visible() is False:
            return True
        try:
            self._driver.hide_keyboard()
        except Exception:  # noqa: BLE001 - verification determines the result
            return False
        return self.keyboard_visible() is False

    def wait(self, duration_seconds: float) -> None:
        if duration_seconds < 0 or duration_seconds > self._maximum_wait_seconds:
            raise DeviceInteractionError(
                f"Wait duration must be between 0 and {self._maximum_wait_seconds:g} seconds."
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
        minimum = 0.5 if self.platform is DevicePlatform.IOS else 0.1
        if duration_seconds < minimum or duration_seconds > 10:
            raise DeviceInteractionError(
                f"Gesture duration must be between {minimum:g} and 10 seconds."
            )


def _boolean_attribute(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    return None
