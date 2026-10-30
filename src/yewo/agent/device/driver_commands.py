"""The `mobile:` vocabulary, in one place.

The session already holds a driver, and the driver wraps most device state on
both platforms. It also reads back what it sets, which adb and simctl mostly
do not: `getConnectivity` answers the question `setConnectivity` asked, and
`queryAppState` answers `terminateApp`.

Reaching past the driver to adb or simctl means hand-building argv twice, once
per platform, and inventing a read for each. So this is the first place a
capability looks, and adb or simctl is the fallback for what the driver has no
method for.

Two kinds of call, named apart on purpose. `change` must work, and raises when
it does not. `ask` is a read, and answers None when the driver will not say. A state that
cannot be read is not the same as a state that is wrong.
"""

import base64
import logging
from typing import Any, Optional

from yewo.errors import DeviceUnavailable

logger = logging.getLogger(__name__)


class DriverCommands:
    """Named `mobile:` methods over one driver session."""

    def __init__(self, driver):
        self.driver = driver

    def change(self, method: str, params: Optional[dict] = None) -> Any:
        """One state-changing method, or the reason the driver refused it.

        Raises:
            DeviceUnavailable: The driver could not run the method. A driver
                that lacks it raises the same way as one whose device refused,
                because neither leaves the state the caller asked for.
        """
        try:
            return self.driver.execute_script(f"mobile: {method}", params or {})
        except Exception as exc:  # noqa: BLE001 - driver boundary
            raise DeviceUnavailable(f"The driver could not run '{method}': {exc}") from exc

    def ask(self, method: str, params: Optional[dict] = None) -> Any:
        """One read, or None when the driver will not answer it."""
        try:
            return self.driver.execute_script(f"mobile: {method}", params or {})
        except Exception as exc:  # noqa: BLE001 - driver boundary
            logger.warning("The driver could not read '%s': %s", method, exc)
            return None

    # --- State both platforms can set ---

    def set_clipboard(self, text: str) -> None:
        """Sets the clipboard, which carries its content base64 encoded."""
        encoded = base64.b64encode(text.encode()).decode()
        self.change("setClipboard", {"content": encoded, "contentType": "plaintext"})

    def clipboard(self) -> Optional[str]:
        """The clipboard text, decoded from the base64 the driver answers."""
        answer = self.ask("getClipboard", {"contentType": "plaintext"})
        if not isinstance(answer, str) or not answer:
            return None
        try:
            return base64.b64decode(answer).decode()
        except (ValueError, UnicodeDecodeError):
            logger.warning("The clipboard answer was not base64 text")
            return None

    def open_deeplink(self, url: str) -> None:
        self.change("deepLink", {"url": url})

    def terminate_app(self, app_id: str) -> None:
        self.change("terminateApp", {"appId": app_id, "bundleId": app_id})

    def clear_app(self, app_id: str) -> None:
        self.change("clearApp", {"appId": app_id, "bundleId": app_id})

    def app_state(self, app_id: str) -> Optional[int]:
        """The app's lifecycle state as the platform's own enum.

        0 not installed, 1 not running, 2 running in background suspended,
        3 running in background, 4 running in foreground. The same scale on
        both platforms.
        """
        answer = self.ask("queryAppState", {"appId": app_id, "bundleId": app_id})
        return answer if isinstance(answer, int) else None

    def activate_app(self, app_id: str) -> None:
        self.change("activateApp", {"appId": app_id, "bundleId": app_id})

    def background_app(self) -> None:
        """Leave the app backgrounded until an explicit activation.

        A negative duration has the same meaning in both Appium mobile
        drivers: do not automatically return the app to the foreground.
        """
        self.change("backgroundApp", {"seconds": -1})

    def orientation(self) -> Optional[str]:
        """The cross-platform WebDriver orientation, normalized to lower case."""
        try:
            answer = self.driver.orientation
        except Exception as exc:  # noqa: BLE001 - driver boundary
            logger.warning("The driver could not read orientation: %s", exc)
            return None
        if not isinstance(answer, str):
            return None
        normalized = answer.lower()
        return normalized if normalized in ("portrait", "landscape") else None

    def android_system_bars(self) -> Optional[dict]:
        """Android system-bar rectangles in the driver's screen coordinates."""
        try:
            answer = self.driver.get_system_bars()
        except Exception as exc:  # noqa: BLE001 - driver boundary
            logger.warning("The Android driver could not read system bars: %s", exc)
            return None
        return answer if isinstance(answer, dict) else None

    def ios_viewport_rect(self) -> Optional[dict]:
        """The iOS application viewport inside the complete screenshot."""
        answer = self.ask("viewportRect")
        return answer if isinstance(answer, dict) else None

    def set_orientation(self, orientation: str) -> None:
        try:
            self.driver.orientation = orientation.upper()
        except Exception as exc:  # noqa: BLE001 - driver boundary
            raise DeviceUnavailable(
                f"The driver could not set orientation: {exc}"
            ) from exc

    def is_locked(self) -> Optional[bool]:
        answer = self.ask("isLocked")
        if isinstance(answer, dict):
            answer = answer.get("isLocked", answer.get("locked"))
        return answer if isinstance(answer, bool) else None

    def lock(self) -> None:
        self.change("lock")

    def unlock(self) -> None:
        self.change("unlock")

    # --- State one platform sets its own way ---

    def set_connectivity(self, wifi: bool, data: bool, airplane: bool) -> None:
        """Android only. Takes the three radios apart rather than as a bitmask."""
        self.change(
            "setConnectivity",
            {"wifi": wifi, "data": data, "airplaneMode": airplane},
        )

    def connectivity(self) -> Optional[dict]:
        """Android only. What the three radios report."""
        answer = self.ask("getConnectivity")
        return answer if isinstance(answer, dict) else None

    def emulator_console(self, command: str) -> Optional[str]:
        """Android emulator console through Appium's authenticated channel."""
        answer = self.ask("execEmuConsoleCommand", {"command": command})
        return answer if isinstance(answer, str) and answer else None

    def condition_inducers(self) -> Optional[list[dict]]:
        """iOS physical-device conditions, including their active profiles."""
        answer = self.ask("listConditionInducers")
        if not isinstance(answer, list) or not all(isinstance(item, dict) for item in answer):
            return None
        return answer

    def enable_condition_inducer(self, condition_id: str, profile_id: str) -> None:
        answer = self.change(
            "enableConditionInducer",
            {"conditionID": condition_id, "profileID": profile_id},
        )
        if answer is not True:
            raise DeviceUnavailable(
                f"The driver did not confirm condition profile '{profile_id}'"
            )

    def disable_condition_inducer(self) -> None:
        answer = self.change("disableConditionInducer")
        if answer is not True:
            raise DeviceUnavailable("The driver did not confirm disabling the condition inducer")

    def set_android_location(self, latitude: float, longitude: float) -> None:
        """Sets the mocked location, which takes the pair at the top level."""
        self.change(
            "setGeolocation", {"latitude": latitude, "longitude": longitude}
        )

    def android_location(self) -> Optional[dict]:
        """Where the device thinks it is, after dropping the cached answer.

        `getGeolocation` reads a cache that `setGeolocation` does not clear.
        Without the refresh it reports the place before the change, and
        a receipt then refuses every location that was set correctly.
        """
        self.ask("refreshGpsCache")
        answer = self.ask("getGeolocation")
        return answer if isinstance(answer, dict) else None

    def set_ios_location(self, latitude: float, longitude: float) -> None:
        self.change(
            "setSimulatedLocation", {"latitude": latitude, "longitude": longitude}
        )

    def ios_location(self) -> Optional[dict]:
        answer = self.ask("getSimulatedLocation")
        return answer if isinstance(answer, dict) else None

    def reset_ios_location(self) -> None:
        """Gives the device its own location back.

        A simulated location outlives the session that set it, and the next
        scenario would run somewhere it never declared.
        """
        self.change("resetSimulatedLocation")

    def change_android_permissions(
        self, app_id: str, permissions: list[str], grant: bool
    ) -> None:
        self.change(
            "changePermissions",
            {
                "permissions": permissions,
                "appPackage": app_id,
                "action": "grant" if grant else "revoke",
            },
        )

    def android_permissions(self, app_id: str) -> Optional[list]:
        """The permissions the package currently holds."""
        answer = self.ask(
            "getPermissions", {"type": "granted", "appPackage": app_id}
        )
        return answer if isinstance(answer, list) else None

    def set_ios_permission(self, app_id: str, service: str, grant: bool) -> None:
        self.change(
            "setPermission",
            {"access": {service: "yes" if grant else "no"}, "bundleId": app_id},
        )

    def ios_permission(self, app_id: str, service: str) -> Optional[str]:
        """"yes", "no" or "unset" for one service, as the driver reports it."""
        answer = self.ask(
            "getPermission", {"service": service, "bundleId": app_id}
        )
        return answer if isinstance(answer, str) else None

    def send_android_trim_memory(self, app_id: str) -> None:
        self.change(
            "sendTrimMemory", {"pkg": app_id, "level": "RUNNING_CRITICAL"}
        )

    def send_ios_memory_warning(self, app_id: str) -> None:
        """Physical device only, from iOS 17 and Xcode 15.

        The bundle id is required. The driver finds the process from it, and
        without it the command is refused before it reaches the device.
        """
        self.change("sendMemoryWarning", {"bundleId": app_id})

    def gsm_call(self, phone: str) -> None:
        """Android emulator only."""
        self.change("gsmCall", {"phoneNumber": phone, "action": "call"})

    def fingerprint(self, finger_id: int) -> None:
        """Android emulator only."""
        self.change("fingerprint", {"fingerprintId": finger_id})

    def set_ios_voice_over(self, enabled: bool) -> None:
        self.change("enableVoiceOver" if enabled else "disableVoiceOver")

    def ios_voice_over(self) -> Optional[bool]:
        """Whether the screen reader is on, or None where the driver would not say.

        The driver answers with an object carrying `enabled`. A bare boolean
        is read too. The value comes from the test runner on the device, and
        the wrapper around it is not ours to depend on.
        """
        answer = self.ask("isVoiceOverEnabled")
        if isinstance(answer, dict):
            answer = answer.get("enabled")
        return answer if isinstance(answer, bool) else None
