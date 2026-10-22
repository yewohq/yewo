"""What the device says about the state a capability just asked for.

Kept apart from the code that applies a capability. Reading a state is a
different question from setting one, and the answer has different rules. A state nothing can read reports None. The command is then the only evidence
there is, and a receipt invented for it would say more than is known.

The driver answers first here too. It reads most states on both platforms,
where adb reads some and simctl reads almost none.
"""

import logging
from typing import Callable, Optional

from yewo.agent.actuation.actions import (
    AccessibilityCapability,
    AppStateCapability,
    BatteryCapability,
    Capability,
    ClipboardCapability,
    LocaleCapability,
    LocationCapability,
    NetworkCapability,
    PermissionCapability,
)
from yewo.agent.device import receipts
from yewo.agent.device.capability_maps import (
    ANDROID_PERMISSIONS,
    APP_NOT_RUNNING,
    CONNECTIVITY,
    IOS_PRIVACY_SERVICES,
    TALKBACK,
)
from yewo.agent.device.driver_commands import DriverCommands
from yewo.agent.device.platform import Platform
from yewo.agent.device.receipts import Receipt

logger = logging.getLogger(__name__)

ReceiptFactory = Callable[..., Optional[Receipt]]


def read_capability_receipt(
    platform: Platform,
    target: str,
    capability: Capability,
    commands: DriverCommands,
    app_id: Optional[str],
) -> Optional[Receipt]:
    """The receipt for one capability, or None when nothing can read it."""

    def create_receipt(requested, observed, source) -> Receipt:
        return Receipt(
            capability=capability.kind.value,
            requested=str(requested),
            observed=None if observed is None else str(observed),
            target=target,
            source=source,
            private=isinstance(capability, ClipboardCapability),
        )

    if platform == Platform.IOS:
        return _read_ios_receipt(capability, commands, app_id, create_receipt)
    return _read_android_receipt(target, capability, commands, app_id, create_receipt)


def _read_android_receipt(
    target, capability, commands, package, create_receipt: ReceiptFactory
) -> Optional[Receipt]:
    match capability:
        case NetworkCapability(mode=mode):
            radios = commands.connectivity()
            if radios is not None:
                return create_receipt(
                    _format_radio_state(CONNECTIVITY[mode]),
                    _format_radio_state({
                        "wifi": radios.get("wifi"),
                        "data": radios.get("data"),
                        "airplane": radios.get("airplaneMode"),
                    }),
                    "mobile: getConnectivity",
                )
            # adb reads airplane mode and nothing else, so it can only answer
            # the modes that are about airplane mode.
            if mode not in ("offline", "airplane", "online"):
                return None
            return create_receipt(
                "off" if mode == "online" else "on",
                receipts.airplane_mode(target),
                "settings get global airplane_mode_on",
            )
        case LocationCapability():
            return _read_location_receipt(
                capability,
                commands.android_location(),
                create_receipt,
                "mobile: getGeolocation",
            )
        case LocaleCapability():
            if not package:
                return None
            requested_locale = (
                f"{capability.language}-{capability.country}"
                if capability.country
                else capability.language
            )
            return create_receipt(
                requested_locale,
                receipts.app_locale(target, package),
                "cmd locale get-app-locales",
            )
        case BatteryCapability(level=level, charging=charging):
            return create_receipt(
                f"level={level},charging={str(charging).lower()}",
                receipts.battery_state(target),
                "dumpsys battery",
            )
        case PermissionCapability(grant=grant):
            return _read_android_permission_receipt(
                target, capability, grant, commands, package, create_receipt
            )
        case AccessibilityCapability(enabled=enabled):
            return create_receipt(
                "on" if enabled else "off",
                receipts.accessibility_enabled(target, TALKBACK),
                "settings get secure enabled_accessibility_services",
            )
        case AppStateCapability(action="force_stop"):
            return _read_stopped_app_receipt(
                target, commands, package, create_receipt
            )
        case ClipboardCapability(text=text):
            return create_receipt(text, commands.clipboard(), "mobile: getClipboard")
        case _:
            return None


def _read_ios_receipt(
    capability, commands, app_id, create_receipt: ReceiptFactory
) -> Optional[Receipt]:
    match capability:
        case LocationCapability():
            return _read_location_receipt(
                capability,
                commands.ios_location(),
                create_receipt,
                "mobile: getSimulatedLocation",
            )
        case PermissionCapability(grant=grant):
            service = IOS_PRIVACY_SERVICES.get(capability.permission)
            if not service or not app_id:
                return None
            return create_receipt(
                "yes" if grant else "no",
                commands.ios_permission(app_id, service),
                "mobile: getPermission",
            )
        case AccessibilityCapability(enabled=enabled):
            return create_receipt(
                enabled, commands.ios_voice_over(), "mobile: isVoiceOverEnabled"
            )
        case AppStateCapability(action="force_stop"):
            return _read_stopped_app_receipt(
                None, commands, app_id, create_receipt
            )
        case ClipboardCapability(text=text):
            return create_receipt(text, commands.clipboard(), "mobile: getClipboard")
        case _:
            return None


def _format_coordinates(latitude, longitude) -> str:
    """One coordinate pair, rounded to about eleven metres.

    The device answers with the float it stored, which is not always the float
    it was given: 35.6812 comes back as 35.6811983. Four places keeps that a
    match while a different place stays a mismatch.
    """
    return f"{round(latitude, 4)},{round(longitude, 4)}"


def _read_location_receipt(
    capability,
    observed_location,
    create_receipt: ReceiptFactory,
    source: str,
) -> Optional[Receipt]:
    """Both coordinates. A latitude alone confirms a parallel, not a place."""
    if observed_location is None:
        return None
    observed = observed_location.get("latitude"), observed_location.get("longitude")
    if any(value is None for value in observed):
        return None
    return create_receipt(
        _format_coordinates(capability.latitude, capability.longitude),
        _format_coordinates(float(observed[0]), float(observed[1])),
        source,
    )


def _format_radio_state(state: dict) -> str:
    """The three radios as one state, so none of them is left uncompared."""
    def flag(value):
        return "unknown" if value is None else ("on" if value else "off")

    return ",".join(f"{name}={flag(state.get(name))}"
                    for name in ("wifi", "data", "airplane"))


def _read_android_permission_receipt(
    target, capability, grant, commands, package, create_receipt: ReceiptFactory
):
    """Whether the package holds the permission, from whichever can say."""
    if not package:
        return None
    permission = ANDROID_PERMISSIONS[capability.permission]
    granted_permissions = commands.android_permissions(package)
    if granted_permissions is not None:
        return create_receipt(
            "granted" if grant else "denied",
            "granted" if permission in granted_permissions else "denied",
            "mobile: getPermissions",
        )
    return create_receipt(
        "granted" if grant else "denied",
        receipts.permission_granted(target, package, permission),
        "dumpsys package",
    )


def _read_stopped_app_receipt(
    target, commands, package, create_receipt: ReceiptFactory
) -> Optional[Receipt]:
    """Whether the app really stopped, from the driver or from its pid."""
    if not package:
        return None
    state = commands.app_state(package)
    if state is not None:
        return create_receipt(
            "stopped",
            "stopped" if state <= APP_NOT_RUNNING else "running",
            "mobile: queryAppState",
        )
    if target is None:
        return None
    return create_receipt("stopped", receipts.app_running(target, package), "pidof")
