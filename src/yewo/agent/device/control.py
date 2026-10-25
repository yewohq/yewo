"""
High-level control interface for interacting with mobile devices (Android/iOS) via Appium.
Encapsulates driver initialization, screenshots, and gesture/input execution.
"""

import logging
import time
from typing import TYPE_CHECKING, Optional

from appium import webdriver
from selenium.common.exceptions import WebDriverException
from appium.options.android import UiAutomator2Options
from appium.options.ios import XCUITestOptions
from selenium.webdriver.common.actions.action_builder import ActionBuilder
from selenium.webdriver.common.actions.pointer_input import PointerInput

from yewo.agent.device.adb_control import AdbControl
from yewo.agent.device.focus import Focus
from yewo.agent.device.diagnostics import diagnose_avd_launch
from yewo.agent.device.platform import Platform
from yewo.agent.device.ios_app import bundle_identifier
from yewo.defaults import (
    AVD_LAUNCH_TIMEOUT_MS,
    IDLE_SESSION_TIMEOUT_SECONDS,
)
from yewo.errors import DeviceUnavailable, ObservationUnavailable

if TYPE_CHECKING:
    from yewo.agent.device.state import DeviceState

logger = logging.getLogger(__name__)

ANDROID_KEYCODE_BACK = 4


def _listed_udid(device_name: str) -> Optional[str]:
    """The udid of the listed iOS target with this name, or None.

    Imported here because device enumeration is a command-line concern, and
    this module is the driver's. A name that resolves to nothing still starts
    a session, because a simulator is reachable by name alone.

    Raises:
        UsageError: Two listed targets carry this name, so neither can be
            chosen for the person who typed it.
    """
    from yewo.devices.discovery import identifier_for

    return identifier_for(device_name, "ios")


class DeviceControl:
    """Wrapper for executing gesture actions and capturing state from mobile devices."""

    screen_width = 0
    screen_height = 0
    """The size the driver reported, once it has answered.

    Class-level so the field always exists. Zero means nothing has asked
    the device yet. A caller that keys on size reads that as unknown,
    rather than as a device one pixel wide.
    """

    def __init__(
        self,
        device_name: str,
        app_path: str,
        adb_control: AdbControl = None,
        is_headless: bool = False,
        platform: Platform = Platform.ANDROID,
        keep_device: bool = False,
        reuse_app: bool = False,
        server_url: str = "http://localhost:4723",
    ):
        self.platform = platform
        self.device_name = device_name
        self.app_path = app_path
        # Only an Android session resolves one, and teardown asks either way.
        self.package_id: Optional[str] = None
        self.app_id: Optional[str] = None
        """The app under test, as its own platform names it.

        An Android package id or an iOS bundle id. A capability acts on the
        app rather than on a platform, so it asks for this and not for one of
        the two. `package_id` stays beside it for the adb paths, which take a
        package and nothing else.
        """
        self.serial: Optional[str] = None
        self.keep_device = keep_device
        self.reuse_app = reuse_app
        self.server_url = server_url
        if platform == Platform.IOS:
            options = XCUITestOptions()
            options.platform_name = "iOS"
            options.automation_name = "XCUITest"
            options.device_name = device_name
            # A name selects a target only while one target carries it. A
            # physical iPhone and a simulator can share a name, and the udid
            # is what separates them. XCUITest wants it for a real device.
            udid = _listed_udid(device_name)
            if udid:
                options.set_capability("appium:udid", udid)
                logger.info("Targeting iOS device %s (%s)", device_name, udid)
            options.app = app_path
            options.is_headless = is_headless
            options.new_command_timeout = IDLE_SESSION_TIMEOUT_SECONDS
            options.set_capability("appium:enforceAppInstall", not reuse_app)
            # This session starts WDA, so it must also stop its xcodebuild.
            # XCUITest otherwise caches that process after driver.quit(),
            # which accumulated across runs and exhausted the host.
            options.set_capability("appium:useNewWDA", True)
            self.adb_control = None
        else:
            self._device_was_running = bool(
                AdbControl.resolve_emulator_serial(device_name)
            )
            options = UiAutomator2Options()
            options.app = app_path
            options.platform_name = "Android"
            options.device_name = device_name
            options.is_headless = is_headless
            options.avd = device_name
            options.avd_launch_timeout = AVD_LAUNCH_TIMEOUT_MS
            options.avd_ready_timeout = AVD_LAUNCH_TIMEOUT_MS
            options.new_command_timeout = IDLE_SESSION_TIMEOUT_SECONDS
            options.set_capability("appium:enforceAppInstall", not reuse_app)
            if keep_device:
                self._skip_redundant_setup(options)
            self.adb_control = adb_control
            adb_runner = adb_control if adb_control is not None else AdbControl
            package_id, _ = adb_runner.get_apk_info(app_path)
            self.package_id = package_id
            options.app_package = package_id
            options.app_activity = adb_runner.resolve_launch_activity(
                app_path, package_id, AdbControl.resolve_emulator_serial(device_name)
            )

        self.driver = self._start_session(options)
        self.device_info = self._init_device_info()

    def _skip_redundant_setup(self, options) -> None:
        """Reuses the Appium helper apps already on a warm device.

        Session creation spends most of its time reinstalling and restarting
        the UiAutomator2 server and settings app. Skipping that is only safe
        once they are actually present, so check before claiming they are.
        """
        serial = AdbControl.resolve_emulator_serial(self.device_name) or self.device_name
        packages = AdbControl.installed_packages(serial)
        required = {"io.appium.uiautomator2.server", "io.appium.uiautomator2.server.test"}
        if not required.issubset(packages):
            logger.info("No Appium server on %s yet. Doing a full device setup.", serial)
            return
        logger.info("Reusing the Appium server already installed on %s.", serial)
        options.set_capability("appium:skipServerInstallation", True)
        if "io.appium.settings" in packages:
            options.set_capability("appium:skipDeviceInitialization", True)

    def _start_session(self, options):
        """Creates the Appium session, translating a failed AVD launch into a reason.

        Appium reports any launch problem as 'Error getting AVD with retry', hiding
        whatever the emulator actually complained about.
        """
        try:
            return webdriver.Remote(self.server_url, options=options)
        except WebDriverException as exc:
            message = str(exc)
            if "getting AVD" not in message and "Condition unmet" not in message:
                raise
            raise DeviceUnavailable(
                f"The emulator '{self.device_name}' did not start within "
                f"{AVD_LAUNCH_TIMEOUT_MS // 1000}s.\n{diagnose_avd_launch(self.device_name)}"
            ) from exc

    def _resolve_serial(self):
        # Try to resolve emulator/device serial from capabilities
        serial = self.driver.capabilities.get(
            "deviceUDID"
        ) or self.driver.capabilities.get("udid")
        # If not in capabilities, fall back to adb resolution on Android
        original_device_name = self.device_name
        self.device_name = self.driver.capabilities.get("deviceName")
        adb_runner = self.adb_control if self.adb_control is not None else AdbControl
        if not serial and self.platform == Platform.ANDROID:
            serial = adb_runner.resolve_emulator_serial(original_device_name)
        return serial

    def _init_device_info(self):

        size = self.driver.get_window_size()
        raw = self.driver.capabilities.get("pixelRatio", 1)
        pixel_ratio = float(str(raw)[:5])
        logger.info(
            "Screen: %sx%s dp | Pixel ratio: %s",
            size["width"],
            size["height"],
            pixel_ratio,
        )
        self.serial = self._resolve_serial()
        self.app_id = self._resolve_app_id()

        self.platform_name = self.driver.capabilities.get("platformName")
        self.os_version = self.driver.capabilities.get("platformVersion")
        self.model = self.driver.capabilities.get("deviceModel")
        self.screen_width = size["width"]
        self.screen_height = size["height"]
        self.pixel_ratio = pixel_ratio
        logger.info(
            "Device connected. Name: %s, Platform: %s, OS Version: %s, Model: %s, Serial: %s",
            self.device_name,
            self.platform_name,
            self.os_version,
            self.model,
            self.serial,
        )

    def _resolve_app_id(self) -> Optional[str]:
        """The app under test, whichever name this platform gives it.

        The driver answers for both. Android reports `appPackage` and iOS
        reports `bundleId`, each worked out from the app it installed. The
        fallback is the id read off the apk before the session, which is the
        same value by another route and is Android's only.
        """
        capabilities = self.driver.capabilities
        for name in ("appPackage", "bundleId", "appium:appPackage", "appium:bundleId"):
            value = capabilities.get(name)
            if isinstance(value, str) and value:
                return value
        if self.platform == Platform.IOS:
            return bundle_identifier(self.app_path)
        return self.package_id

    @property
    def state(self) -> "DeviceState":
        """What the device says about itself, built on the way through."""
        from yewo.agent.device.state import DeviceState

        return DeviceState(self)

    def is_still_in_sut(self) -> bool:
        return self.state.is_still_in_sut()

    def showing_system_prompt(self) -> bool:
        return self.state.showing_system_prompt()

    def is_keyboard_open(self) -> Optional[bool]:
        return self.state.is_keyboard_open()

    def recover_to_sut(self) -> None:
        self.state.recover_to_sut()

    @property
    def focus(self) -> Focus:
        """Focus checks bound to this session's driver and screen size."""
        area = getattr(self, "screen_width", 0) * getattr(self, "screen_height", 0)
        return Focus(self.driver, area)

    def get_screenshot(self):
        """The screen as PNG bytes, or ObservationUnavailable if unreadable.

        The driver can raise almost anything for a lost session.
        """
        try:
            return self.driver.get_screenshot_as_png()
        except Exception as exc:
            raise ObservationUnavailable(
                f"Could not read the screen of '{self.device_name}': {exc}"
            ) from exc

    def __tap_screen(self, x, y):
        touch = PointerInput("touch", "finger")
        builder = ActionBuilder(self.driver, mouse=touch)
        builder.pointer_action.move_to_location(x, y)
        builder.pointer_action.pointer_down()
        builder.pointer_action.pause(0.1)
        builder.pointer_action.release()
        builder.perform()

    def tap(self, x, y, with_feedback: bool = True):
        if with_feedback:
            self.__tap_screen(x, y)
        else:
            self.driver.tap([(x, y)])





    def __focus_field(self, x, y, action: str) -> None:
        """Puts the cursor in the field at (x, y), or refuses the action.

        A field that is already focused at that point is left alone. Tapping
        it again would open the keyboard, which scrolls the form away from the
        very point that named the field.

        When it does tap, it reads the keyboard on either side. A keyboard
        that was closed and is now open is the form moving, so the point stops
        being evidence of where the field is.
        """
        focus = self.focus
        before = focus.rect()
        if focus.contains(x, y):
            focus.assert_at(x, y, action, before=before)
            return
        keyboard = self.is_keyboard_open()
        self.tap(x, y)
        focus.wait_for(x, y, before=before)
        focus.assert_at(
            x,
            y,
            action,
            before=before,
            layout_moved=keyboard is False and self.is_keyboard_open() is True,
        )

    def type(self, text, x, y):
        if self.platform == Platform.IOS:
            self.tap(x, y)
            time.sleep(1.0)
            self.execute_script("mobile: type", {"text": text})
        else:
            self.__focus_field(x, y, "TYPE")
            self.driver.switch_to.active_element.send_keys(text)
        return self.focus.value()

    def read_focused_field(self):
        """Read the active field after the app has settled."""
        return self.focus.reading()

    def clear(self, x, y):
        try:
            if self.platform == Platform.IOS:
                self.tap(x, y)
                self.focus.wait_for(x, y)
                self.driver.switch_to.active_element.clear()
                return
            self.__focus_field(x, y, "CLEAR")
            try:
                # Select All (Ctrl+A) and then Delete (Backspace) to ensure the field is completely cleared
                self.driver.press_keycode(29, metastate=28672)
                time.sleep(0.2)
                self.driver.press_keycode(67)
                time.sleep(0.2)
            except Exception as e:
                logger.warning("Failed Ctrl+A clear on Android: %s. Trying fallback.", e)
                self.driver.switch_to.active_element.clear()
        except Exception as e:
            logger.error("Failed to clear text field at (%s, %s): %s", x, y, e)
            raise

    def double_tap(self, x, y, with_feedback: bool = True):
        if self.platform == Platform.IOS:
            self.execute_script("mobile: doubleTap", {"x": x, "y": y})
        else:
            self.execute_script("mobile: doubleClickGesture", {"x": x, "y": y})
        time.sleep(0.5)

    def long_press(self, x, y, duration):
        if self.platform == Platform.IOS:
            self.execute_script(
                "mobile: touchAndHold", {"x": x, "y": y, "duration": duration}
            )
        else:
            self.execute_script(
                "mobile: longClickGesture",
                {"x": x, "y": y, "duration": int(duration * 1000)},
            )
        time.sleep(duration + 0.5)

    def drag_and_drop(self, source_x, source_y, destination_x, destination_y):
        if self.platform == Platform.IOS:
            self.execute_script(
                "mobile: dragFromToForDuration",
                {
                    "fromX": source_x,
                    "fromY": source_y,
                    "toX": destination_x,
                    "toY": destination_y,
                    "duration": 1.0,
                },
            )
        else:
            self.execute_script(
                "mobile: dragGesture",
                {
                    "startX": source_x,
                    "startY": source_y,
                    "endX": destination_x,
                    "endY": destination_y,
                },
            )
        time.sleep(0.5)

    def key_event(self, key):
        if self.platform == Platform.ANDROID:
            self.driver.press_keycode(key)
        else:
            self.execute_script("mobile: pressButton", {"name": key})
        time.sleep(0.5)

    def dismiss_keyboard(self) -> bool:
        """Closes the soft keyboard, which covers the lower half of the screen.

        Back is the Android way and is what the planner is told to use, so the
        two agree. iOS has no back key and Appium hides the keyboard directly.

        Returns whether it closed, read back rather than assumed. A key press
        that raised nothing is not a keyboard that went away, and the caller
        spends the screen's one attempt on the answer.
        """
        try:
            if self.platform == Platform.ANDROID:
                self.key_event(ANDROID_KEYCODE_BACK)
            else:
                self.driver.hide_keyboard()
        except Exception as exc:  # noqa: BLE001 - driver boundary
            logger.warning("Could not close the keyboard: %s", exc)
            return False

        # None means no check could answer. Treated as closed, because the
        # press was made and refusing to believe it would retry forever.
        return self.is_keyboard_open() is not True

    def swipe(self, start_x, start_y, end_x, end_y, duration_ms: int = 400):
        """Performs a swipe gesture using W3C pointer actions.

        Args:
            start_x: Starting x coordinate.
            start_y: Starting y coordinate.
            end_x: Ending x coordinate.
            end_y: Ending y coordinate.
            duration_ms: Duration of the swipe gesture in milliseconds.
        """
        touch = PointerInput("touch", "finger")
        builder = ActionBuilder(self.driver, mouse=touch)
        builder.pointer_action.move_to_location(start_x, start_y)
        builder.pointer_action.pointer_down()
        builder.pointer_action.pause(duration_ms / 1000)
        builder.pointer_action.move_to_location(end_x, end_y)
        builder.pointer_action.release()
        builder.perform()
        time.sleep(0.5)

    def remove_app(self, package_id):
        self.driver.remove_app(package_id)

    def remove_current_app(self):
        self.remove_app(self.package_id)

    def quit(self, interrupted: bool = False):
        if not interrupted:
            try:
                self.driver.quit()
            except Exception as e:
                logger.warning("Failed to quit driver cleanly: %s", e)
        if self.platform == Platform.ANDROID:
            serial = AdbControl.resolve_emulator_serial(
                self.serial or self.device_name or ""
            )
            if not serial:
                if interrupted:
                    logger.info("The Android target already stopped after the interruption.")
                return
            try:
                AdbControl.disable_feedback(serial)
            except Exception as e:
                log = logger.info if interrupted else logger.warning
                log("Failed to disable feedback: %s", e)
            if self.keep_device or getattr(self, "_device_was_running", False):
                logger.info("Leaving '%s' running for the next run.", self.device_name)
                return
            try:
                AdbControl.kill_emulator(self.device_name, serial=serial)
            except Exception as e:
                log = logger.info if interrupted else logger.warning
                log("The Android target stopped during cleanup: %s", e)

    def execute_script(self, script, args=None):
        """Runs one driver extension command and returns what it answered.

        The params go through as one object. Appium reads them as `args[0]`,
        so wrapping them in a list arrives as a list where an object belongs,
        and the driver refuses the call rather than ignoring the extra layer.
        """
        return self.driver.execute_script(script, args or {})
