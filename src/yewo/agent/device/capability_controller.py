"""Setting the device states a scenario asks for.

Every capability arrives as its own model, and every one is dispatched by an
explicit case. A capability this platform cannot apply raises, because the
alternative is a scenario that reads as having run offline while the device
stayed online.

The driver comes first and adb or simctl second. The session already holds a
driver, the driver speaks to both platforms in one vocabulary, and it reads
back what it sets. A command line is the fallback for what it has no method
for, and for a device the driver answers about but cannot change.
"""

import logging
import os
from typing import Callable, Optional

from yewo.agent.actuation.actions import (
    AccessibilityCapability,
    AppStateCapability,
    BatteryCapability,
    BiometricCapability,
    Capability,
    ClipboardCapability,
    DeeplinkCapability,
    InterruptCapability,
    LocaleCapability,
    LocationCapability,
    MediaInjectCapability,
    NetworkCapability,
    PermissionCapability,
    TimezoneCapability,
)
from yewo.agent.device import capability_receipts
from yewo.agent.device.adb_control import AdbControl
from yewo.agent.device.capability_maps import (
    ANDROID_PERMISSIONS,
    APPIUM_NETWORK_TYPES,
    CONNECTIVITY,
    IOS_PRIVACY_SERVICES,
    PERMISSION_LIMITED,
    TALKBACK,
)
from yewo.agent.device.driver_commands import DriverCommands
from yewo.agent.device.platform import Platform
from yewo.agent.device.receipts import Receipt
from yewo.agent.device.simctl_control import SimctlControl
from yewo.errors import ActionRejected, DeviceUnavailable

logger = logging.getLogger(__name__)

def _format_capability_label(capability: Capability) -> str:
    """A capability as a reader needs it, with the variant it chose.

    The kind alone leaves a log and a failure saying `interrupt` where the
    scenario asked for an incoming call. The two variants have different
    support, and they fail for different reasons.
    """
    variant = capability.support_variant
    return f"{capability.kind.value} '{variant}'" if variant else capability.kind.value


class CapabilityController:
    """Dispatches system-level state commands to the active device."""

    def __init__(self, device_control):
        """Initializes CapabilityController with active device controls.

        Args:
            device_control: Active DeviceControl instance.
        """
        self.device_control = device_control
        self.driver = device_control.driver
        self.platform = device_control.platform
        self.adb_control = getattr(device_control, "adb_control", None)
        self.commands = DriverCommands(device_control.driver)

    def resolve_device_identifier(self) -> str:
        """Resolves the current device serial number (Android) or UDID (iOS).

        Returns:
            String representing the device serial/UDID.
        """
        name = self.device_control.device_name
        if self.platform == Platform.IOS:
            return self.driver.capabilities.get("udid") or name or ""
        adb_runner = self.adb_control if self.adb_control is not None else AdbControl
        return adb_runner.resolve_emulator_serial(name) or name or ""

    def execute(self, capability: Capability) -> Optional[Receipt]:
        """Applies one capability, reads the state back, or refuses it.

        Args:
            capability: The device state to set.

        Returns:
            What the device reports about the state, or None where nothing can
            read it back.

        Raises:
            ActionRejected: The capability could not be applied here, or the
                device reports a state other than the one asked for. Nothing
                may read the device as being in the requested state.
        """
        capability_label = _format_capability_label(capability)
        logger.info("Executing capability control: %s", capability_label)
        target = self.resolve_device_identifier()
        try:
            self._apply_capability(target, capability)
        except ActionRejected:
            raise
        except Exception as exc:
            logger.error("Failed to execute capability %s: %s", capability_label, exc)
            raise ActionRejected(
                f"Capability {capability_label} failed: {exc}"
            ) from exc

        receipt = capability_receipts.read_capability_receipt(
            self.platform, target, capability, self.commands, self._resolve_app_id()
        )
        if receipt is None:
            return None
        if receipt.unreadable:
            logger.warning("Capability applied but not confirmed. %s", receipt)
        elif not receipt.confirmed:
            raise ActionRejected(f"The device did not take the state. {receipt}")
        else:
            logger.info("Capability confirmed. %s", receipt)
        return receipt

    def _apply_capability(self, target: str, capability: Capability) -> None:
        """Routes one capability to the method that applies it.

        Matched on the model rather than looked up by name. The wildcard below
        refuses a capability with no case rather than ignoring it, and
        `test_every_capability_has_a_case_in_the_dispatch` is what catches one
        before a device does.
        """
        match capability:
            case LocationCapability():
                self._apply_location(target, capability)
            case NetworkCapability():
                self._apply_network(target, capability)
            case BatteryCapability():
                self._apply_battery(target, capability)
            case TimezoneCapability():
                self._apply_timezone(capability)
            case LocaleCapability():
                self._apply_locale(target, capability)
            case PermissionCapability():
                self._apply_permission(target, capability)
            case InterruptCapability():
                self._apply_interrupt(target, capability)
            case BiometricCapability():
                self._apply_biometric(capability)
            case ClipboardCapability():
                self._apply_clipboard(capability)
            case DeeplinkCapability():
                self._apply_deeplink(target, capability)
            case AppStateCapability():
                self._apply_app_state(target, capability)
            case MediaInjectCapability():
                self._apply_media_injection(target, capability)
            case AccessibilityCapability():
                self._apply_accessibility(target, capability)
            case _:
                raise ActionRejected(
                    f"No handler for capability {capability.kind.value}"
                )

    def _unsupported_capability_error(self, capability: Capability, detail: str) -> ActionRejected:
        """The refusal to raise, naming the capability and this platform."""
        return ActionRejected(
            f"{capability.kind.value} cannot be applied on "
            f"{self.platform.value}: {detail}"
        )

    def _require_app_id(self, capability: Capability) -> str:
        """The app under test, which several capabilities act on.

        An Android package id or an iOS bundle id, whichever this session
        resolved. Asking for one of the two by name left every iOS capability
        that acts on the app without an answer.

        Raises:
            ActionRejected: No app is known, so the capability would be
                applied to nothing.
        """
        app_id = self._resolve_app_id()
        if not app_id:
            raise self._unsupported_capability_error(capability, "the app id is not known")
        return app_id

    def _use_driver_or_command(self, doing: str, driver: Callable, command: Callable) -> None:
        """The driver's way, and the command line when the driver has no way.

        A driver missing a method and a device refusing one raise alike, so
        both land here. Falling back on either is right: the command line is
        the older path and works where the driver is not built for the device.

        When both fail, the driver's reason leads. It names the constraint, where the command line reports the symptom.
        Refused a call on a physical phone, the driver says "gsmCall is
        only available for emulators". `adb emu` only exits non-zero,
        with nothing to say.

        Raises:
            DeviceUnavailable: Neither way worked, carrying both reasons.
        """
        refusal = None
        try:
            driver()
            return
        except DeviceUnavailable as exc:
            refusal = exc
            logger.warning(
                "The driver could not %s (%s). Using the command line.", doing, exc
            )
        try:
            command()
        except DeviceUnavailable as failure:
            raise DeviceUnavailable(
                f"Could not {doing}. The driver refused: {refusal}. "
                f"The command line then failed: {failure}"
            ) from failure

    # --- Platform Handler Implementations ---

    def _apply_location(self, target: str, capability: LocationCapability) -> None:
        if self.platform == Platform.IOS:
            self._use_driver_or_command(
                "set the location",
                lambda: self.commands.set_ios_location(
                    capability.latitude, capability.longitude
                ),
                lambda: SimctlControl.set_location(
                    target, capability.latitude, capability.longitude
                ),
            )
        else:
            self._use_driver_or_command(
                "set the location",
                lambda: self.commands.set_android_location(
                    capability.latitude, capability.longitude
                ),
                lambda: AdbControl.mock_location(
                    target, capability.latitude, capability.longitude
                ),
            )

    def _apply_network(self, target: str, capability: NetworkCapability) -> None:
        if self.platform != Platform.ANDROID:
            raise self._unsupported_capability_error(
                capability,
                "the simulator shares the host network connection, and the "
                "condition inducers that would change it need a real device",
            )

        radios = CONNECTIVITY[capability.mode]
        self._use_driver_or_command(
            f"set connectivity to {capability.mode}",
            lambda: self.commands.set_connectivity(**radios),
            lambda: self._apply_network_by_command(target, capability.mode),
        )

    def _apply_network_by_command(self, target: str, mode: str) -> None:
        """Airplane mode through adb, and the rest through Appium's bitmask."""
        if mode in ("offline", "airplane", "online"):
            AdbControl.set_airplane_mode(target, mode != "online")
            return
        self.driver.set_network_connection(APPIUM_NETWORK_TYPES[mode])

    def _apply_battery(self, target: str, capability: BatteryCapability) -> None:
        """Set on Android, and refused on iOS.

        `simctl status_bar` overrides what the status bar draws. The battery
        an app reads is a different value, and the override does not change
        it. So nothing confirms the state and nothing puts it back.
        """
        if self.platform == Platform.IOS:
            raise self._unsupported_capability_error(
                capability,
                "the status-bar override does not change the level an app "
                "reads, so no read confirms it",
            )
        else:
            AdbControl.set_battery_status(
                target, capability.level, capability.charging
            )

    def _apply_timezone(self, capability: TimezoneCapability) -> None:
        """Refused on both platforms, because neither offers a path that works.

        On Android the system zone is `persist.sys.timezone`, which the shell
        user cannot set. A write to `settings global timezone` changes
        nothing the device reads. Reading that setting back confirms
        the write while the clock stays where it was. simctl sets no zone at all.
        """
        raise self._unsupported_capability_error(
            capability,
            "no supported path sets the system zone, and the one that looks "
            "like it does is not read by the device",
        )

    def _apply_locale(self, target: str, capability: LocaleCapability) -> None:
        """Per-app on Android, and refused on iOS.

        simctl and the driver both set the locale of the whole simulator, and
        neither reaches an app that is already running. A scenario would go on
        reading the locale it launched with while the receipt confirmed the
        device had moved.
        """
        if self.platform == Platform.IOS:
            raise self._unsupported_capability_error(
                capability,
                "the localization command reaches no app that is already "
                "running, and it sets the simulator rather than the app",
            )
        package = self._require_app_id(capability)
        locale = (
            f"{capability.language}-{capability.country}"
            if capability.country
            else capability.language
        )
        AdbControl.set_app_locale(target, package, locale)

    def _apply_permission(self, target: str, capability: PermissionCapability) -> None:
        package = self._require_app_id(capability)

        if self.platform == Platform.IOS:
            self._apply_ios_permission(target, package, capability)
            return

        permission = ANDROID_PERMISSIONS.get(capability.permission)
        if not permission:
            raise self._unsupported_capability_error(
                capability, f"'{capability.permission}' names no Android permission"
            )
        self._use_driver_or_command(
            f"change {permission}",
            lambda: self.commands.change_android_permissions(
                package, [permission], capability.grant
            ),
            lambda: AdbControl.set_app_permission(
                target, package, permission, capability.grant
            ),
        )

    def _apply_ios_permission(
        self, target: str, app_id: str, capability: PermissionCapability
    ) -> None:
        """Grants or revokes one service, where the decision can be put back.

        A grant limited to part of the library is refused rather than
        overwritten. Nothing sets a limited grant, so a scenario that changed
        one would hand the next scenario a decision nobody declared.
        """
        service = IOS_PRIVACY_SERVICES.get(capability.permission)
        if not service:
            raise self._unsupported_capability_error(
                capability,
                f"'{capability.permission}' names no simctl privacy service",
            )
        if self.commands.ios_permission(app_id, service) == PERMISSION_LIMITED:
            raise self._unsupported_capability_error(
                capability,
                f"the {service} permission is limited to part of the library, "
                "and no command sets that state again",
            )
        action = "grant" if capability.grant else "revoke"
        self._use_driver_or_command(
            f"{action} {service}",
            lambda: self.commands.set_ios_permission(
                app_id, service, capability.grant
            ),
            lambda: SimctlControl.set_privacy(target, action, service, app_id),
        )

    def _apply_interrupt(self, target: str, capability: InterruptCapability) -> None:
        if capability.type == "low_memory":
            if self.platform == Platform.IOS:
                self._apply_ios_memory_warning(target)
                return
            package = self._require_app_id(capability)
            self._use_driver_or_command(
                "send a trim-memory signal",
                lambda: self.commands.send_android_trim_memory(package),
                lambda: self._apply_trim_memory_by_pid(target, package, capability),
            )
            return

        if self.platform != Platform.ANDROID:
            raise self._unsupported_capability_error(
                capability, "simctl cannot place a call into the simulator"
            )
        self._use_driver_or_command(
            "place a call",
            lambda: self.commands.gsm_call(capability.phone),
            lambda: AdbControl.simulate_incoming_call(target, capability.phone),
        )

    def _apply_ios_memory_warning(self, target: str) -> None:
        """The driver path on a device, and simctl on a simulator.

        The driver needs the bundle id to find the process. simctl needs
        neither it nor a running app, so a session that cannot name the app
        can still post a warning to a simulator.
        """
        package = self._resolve_app_id()
        if not package:
            SimctlControl.notify_low_memory(target)
            return
        self._use_driver_or_command(
            "post a memory warning",
            lambda: self.commands.send_ios_memory_warning(package),
            lambda: SimctlControl.notify_low_memory(target),
        )

    def _apply_trim_memory_by_pid(
        self, target: str, package: str, capability: Capability
    ) -> None:
        """The adb path, which needs the pid the driver did not."""
        pid = AdbControl.get_app_pid(target, package)
        if not pid:
            raise self._unsupported_capability_error(
                capability, f"{package} is not running, so it has no pid"
            )
        AdbControl.simulate_low_memory(target, pid)

    def _apply_biometric(self, capability: BiometricCapability) -> None:
        if self.platform == Platform.IOS:
            enroll = capability.action in ("enroll", "match")
            self.commands.change("enrollBiometric", {"isEnabled": enroll})
            if capability.action in ("match", "unmatch"):
                self.commands.change(
                    "sendBiometricMatch",
                    {
                        "type": capability.type,
                        "match": capability.action == "match",
                    },
                )
        else:
            self._use_driver_or_command(
                "touch the fingerprint sensor",
                lambda: self.commands.fingerprint(capability.finger_id),
                lambda: self.driver.finger_print(capability.finger_id),
            )

    def _apply_clipboard(self, capability: ClipboardCapability) -> None:
        logger.info("Setting device clipboard")
        self.commands.set_clipboard(capability.text)

    def _apply_deeplink(self, target: str, capability: DeeplinkCapability) -> None:
        if self.platform == Platform.IOS:
            self._use_driver_or_command(
                "open a URL",
                lambda: self.commands.open_deeplink(capability.url),
                lambda: SimctlControl.open_url(target, capability.url),
            )
            return
        package = self._require_app_id(capability)
        self._use_driver_or_command(
            "open a deep link",
            lambda: self.commands.open_deeplink(capability.url),
            lambda: AdbControl.open_deeplink(target, capability.url, package),
        )

    def _apply_app_state(self, target: str, capability: AppStateCapability) -> None:
        package = self._require_app_id(capability)

        if capability.action == "force_stop":
            self._use_driver_or_command(
                "terminate the app",
                lambda: self.commands.terminate_app(package),
                lambda: self._terminate_app_by_command(target, package),
            )
            return

        if self.platform != Platform.ANDROID:
            def no_command_line() -> None:
                raise self._unsupported_capability_error(
                    capability, "simctl cannot clear an installed app"
                )

            self._use_driver_or_command(
                "clear the app container",
                lambda: self.commands.clear_app(package),
                no_command_line,
            )
            return
        AdbControl.clear_app_data(target, package)

    def _terminate_app_by_command(self, target: str, package: str) -> None:
        if self.platform == Platform.IOS:
            SimctlControl.terminate(target, package)
        else:
            AdbControl.force_stop_app(target, package)

    def _apply_media_injection(self, target: str, capability: MediaInjectCapability) -> None:
        if not os.path.exists(capability.local_path):
            raise self._unsupported_capability_error(capability, f"no file at '{capability.local_path}'")

        if self.platform == Platform.IOS:
            SimctlControl.add_media(target, capability.local_path)
        else:
            destination = f"/sdcard/Pictures/{os.path.basename(capability.local_path)}"
            AdbControl.push_file(target, capability.local_path, destination)
            AdbControl.scan_media_file(target, destination)

    def _apply_accessibility(self, target: str, capability: AccessibilityCapability) -> None:
        if self.platform == Platform.IOS:
            self.commands.set_ios_voice_over(capability.enabled)
            return
        AdbControl.set_accessibility_service(target, TALKBACK, capability.enabled)

    def _resolve_app_id(self) -> Optional[str]:
        """The app under test, or None. Unlike `_require_app_id`, this never raises.

        A receipt is read after the capability was applied, so a missing app
        there is not a reason to fail the capability a second time.
        """
        return self.device_control.app_id
