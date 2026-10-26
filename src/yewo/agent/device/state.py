"""Whether the app is still the one on screen, and getting back to it.

What the drift check returns is prompt text, for the planner to read.
"""

import logging
import time
from typing import Optional

from yewo.agent.device.adb_control import AdbControl
from yewo.agent.device.platform import Platform
from yewo.agent.device.scope import DeviceScope
from yewo.utils.subprocess_runner import run_command

logger = logging.getLogger(__name__)

ANDROID_KEYCODE_BACK = 4

# Android draws runtime permission prompts from its own package, so the app
# under test is briefly not the foreground one.
PERMISSION_DIALOG_PACKAGES = frozenset(
    {
        "com.google.android.permissioncontroller",
        "com.android.permissioncontroller",
        "com.android.packageinstaller",
    }
)
IOS_APP_STATE_FOREGROUND = 4
RECOVERY_SLEEP_AFTER_BACK = 1.5
RECOVERY_SLEEP_AFTER_ACTIVATE = 2.0


def context_drift(
    device_control, last_external_nav: bool, settle_time: float
) -> Optional[str]:
    # Asked before anything else. On Android the prompt reads as lost focus and
    # recovery would press back, which answers it as "deny". On iOS the app
    # stays in the foreground and nothing below would see the prompt at all.
    # The planner reads it off the screenshot like any other screen.
    if device_control.showing_system_prompt():
        logger.info("A system prompt is up; leaving it for the planner to answer.")
        return (
            "Info: the operating system is asking you something. Its prompt is on "
            "screen now. The prompt belongs to the system, not to the app. Answer "
            "it as the scenario requires. Do not leave the prompt. A dismissed "
            "permission prompt denies the permission."
        )

    if not device_control.is_still_in_sut():
        if not last_external_nav:
            logger.warning("Unexpected context drift. Attempting recovery...")
            device_control.recover_to_sut()
            time.sleep(settle_time)
            if not device_control.is_still_in_sut():
                logger.error("Recovery failed.")
                return (
                    "Warning: The device has drifted outside the target app (SUT) and recovery failed. "
                    "Navigate back to the app, or stop the scenario."
                )
            logger.info("Recovery successful.")
            return None
        return (
            "Info: The device is currently outside the target app (SUT) as expected. "
            "Make sure to return to the target app once the external tasks are completed."
        )
    return None


class DeviceState:
    """What the device says about itself, and how to get the app back.

    The answers come from Appium, adb and idb. A caller does not need to know
    which.
    """

    def __init__(self, control):
        self.control = control

    def is_still_in_sut(self) -> bool:
        """Checks if the device is currently focused on the target app under test (SUT)."""
        if self.control.scope is DeviceScope.DEVICE:
            return True
        try:
            if self.control.platform == Platform.ANDROID:
                return self.control.driver.current_package == self.control.app_id
            if self.control.app_id:
                return (
                    self.control.driver.query_app_state(self.control.app_id)
                    == IOS_APP_STATE_FOREGROUND
                )
        except Exception as e:  # noqa: BLE001 - driver boundary
            logger.warning("Failed to check SUT focus: %s", e)
        return True

    def showing_system_prompt(self) -> bool:
        """Whether the operating system is asking the user something over the app.

        Android names the prompt: it is drawn from a package of its own, so the
        foreground package stops matching and the check is exact. Saying so
        matters there because drift recovery presses back, and back on a
        permission prompt denies it, after which the journey fails several
        steps later for a reason that looks nothing like a denial.

        iOS answers False, and the alternative is worse. Appium reports an
        alert for the app's own modals too, with nothing to tell the two apart,
        so claiming a prompt there would tell the planner an app dialog belongs
        to the system and must not be dismissed. Being wrong on iOS costs
        little by comparison: recovery brings the app forward rather than
        pressing back, so no permission is answered by accident.
        """
        if self.control.platform != Platform.ANDROID:
            return False
        try:
            return self.control.driver.current_package in PERMISSION_DIALOG_PACKAGES
        except Exception as exc:  # noqa: BLE001 - driver boundary
            logger.debug("Could not read the foreground package: %s", exc)
            return False

    def is_keyboard_open(self) -> Optional[bool]:
        """Detect whether the soft keyboard is currently visible on screen.

        Tries Appium first (cross-platform), then falls back to a native check:
        - Android: adb dumpsys input_method
        - iOS: idb accessibility tree query
        Returns None if all checks fail.
        """
        try:
            return bool(self.control.driver.is_keyboard_shown())
        except Exception:  # noqa: BLE001 - driver boundary
            logger.debug("Appium is_keyboard_shown() unavailable, using native fallback")

        if self.control.platform == Platform.ANDROID:
            # `serial` is read at session start, so a control built but not
            # yet connected has none. This probe must degrade, never raise.
            serial = getattr(self.control, "serial", None) or getattr(
                self.control, "device_name", None
            )
            shown = (
                AdbControl.ask(serial, "dumpsys", "input_method", timeout=3.0)
                if serial
                else None
            )
            if shown is not None:
                return "mInputShown=true" in shown

        elif self.control.platform == Platform.IOS:
            try:
                udid = self.control.serial or self.control.device_name
                result = run_command(
                    ["idb", "ui", "describe-all", "--udid", udid],
                    capture_output=True, text=True, timeout=3.0,
                )
                return "XCUIElementTypeKeyboard" in (result.stdout or "")
            except Exception as e:  # noqa: BLE001 - driver boundary
                logger.debug("idb keyboard check failed: %s", e)

        return None

    def recover_to_sut(self) -> None:
        """Attempts to recover the focus back to the target app under test (SUT) after drift."""
        if self.control.scope is DeviceScope.DEVICE:
            return
        logger.info(
            "Context drift detected. Attempting to recover focus back to the target app..."
        )
        if self.control.platform == Platform.ANDROID:
            try:
                self.control.key_event(ANDROID_KEYCODE_BACK)
                time.sleep(RECOVERY_SLEEP_AFTER_BACK)
            except Exception as e:  # noqa: BLE001 - driver boundary
                logger.warning("Failed to press back key during recovery: %s", e)

        if self.control.is_still_in_sut():
            return

        app_id = self.control.app_id

        if not app_id:
            logger.error("Cannot recover: no app ID available.")
            return

        try:
            logger.info("Activating SUT app: %s", app_id)
            self.control.driver.activate_app(app_id)
            time.sleep(RECOVERY_SLEEP_AFTER_ACTIVATE)
        except Exception as e:  # noqa: BLE001 - driver boundary
            logger.error("Failed to activate SUT app: %s", e)
