"""Giving the device back the way a scenario found it.

Scenarios share one device, and `SetupMode` restores the app rather than the
system around it. A scenario that goes offline, or moves the clock, or switches
the screen reader on, leaves it that way for every scenario after it. The next
one then tests a device nobody configured, and its verdict is about a state it
never declared.

So each scenario reads the states a capability could change, and writes back
the ones that moved. Reading first is what makes this honest. A state that
could not be read is not restored, because restoring it would write a value
that was never observed.
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Sequence

from yewo.agent.actuation.actions import (
    Capability,
    CapabilityKind,
    PermissionCapability,
)
from yewo.agent.device import receipts
from yewo.agent.device.adb_control import AdbControl
from yewo.agent.device.capability_maps import IOS_PRIVACY_SERVICES, PERMISSION_UNSET
from yewo.agent.device.driver_commands import DriverCommands
from yewo.agent.device.platform import Platform
from yewo.agent.device.simctl_control import SimctlControl
from yewo.errors import DeviceUnavailable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RestorationEvent:
    """The result of putting one changed device state back."""

    state: str
    restored: bool
    reason: Optional[str] = None


class StateSnapshot:
    """What was read before a scenario, whichever platform answered.

    Every field of every snapshot is optional. None means the state could not
    be read, and an unread state is left alone rather than guessed at.
    """

    def describe(self) -> str:
        known = {
            name: value
            for name, value in vars(self).items()
            if value is not None
        }
        return ", ".join(f"{name}={value}" for name, value in known.items()) or "nothing"


@dataclass(frozen=True)
class AndroidStateSnapshot(StateSnapshot):
    """The Android states a capability can change, as they were found."""

    airplane: Optional[str] = None
    wifi: Optional[str] = None
    data: Optional[str] = None
    battery: Optional[str] = None
    timezone: Optional[str] = None
    locale: Optional[str] = None
    accessibility: Optional[str] = None
    """Every enabled accessibility service, not one service's state.

    The setting is one list, and a scenario that switches a screen reader
    on or off rewrites it. Keeping one boolean would restore the list to
    whatever that boolean implied and drop every other service in it.
    """


@dataclass(frozen=True)
class IosStateSnapshot(StateSnapshot):
    """The iOS states a capability can change, as they were found.

    Read only where the scenario declared the capability that changes one. A
    read for a state no scenario touches costs a driver call on every
    scenario. The screen reader answers nothing at all before iOS 27.
    """

    voice_over: Optional[bool] = None

    permissions: Optional[dict[str, str]] = None
    """What the app was allowed, service by service, before the scenario ran.

    A decision outlives the install, so each service is read and written back
    on its own. The values are the reader's own words, including `unset`,
    which `simctl privacy reset` is the way back to.
    """

    simulated_location: Optional[str] = None
    """Where the device was told it is, or `NOWHERE` where it was told nothing.

    Three states and not two. The driver answers with a pair of nulls for a
    location nobody simulated, which is a reading rather than a failed read.
    An unread state stays None, and nothing is written back for it.
    """


def _switch(state: Optional[bool]) -> Optional[str]:
    """A read-back boolean as "on" or "off", or None where nothing answered.

    One writer puts every state back, and it is handed text. A None read as
    False would restore a state the device was never read for.
    """
    return None if state is None else ("on" if state else "off")


def _radio(radios: Optional[dict], name: str) -> Optional[str]:
    """One radio as "on" or "off", or None when the driver did not say."""
    if radios is None or name not in radios:
        return None
    return "on" if radios[name] else "off"


def snapshot_android(
    serial: str, package: Optional[str], commands: Optional[DriverCommands] = None
) -> AndroidStateSnapshot:
    """The states to put back, read before the scenario runs.

    The three radios are read together where the driver can read them.
    Airplane mode alone is not enough. A scenario that asks for wifi_only
    turns mobile data off and leaves airplane mode where it was. A restore
    watching only airplane mode then sees nothing move, and leaves data off
    for every scenario after it.
    """
    radios = commands.connectivity() if commands is not None else None
    return AndroidStateSnapshot(
        airplane=receipts.airplane_mode(serial),
        wifi=_radio(radios, "wifi"),
        data=_radio(radios, "data"),
        battery=receipts.battery_level(serial),
        timezone=receipts.timezone(serial),
        locale=receipts.app_locale(serial, package) if package else None,
        accessibility=AdbControl.enabled_accessibility_services(serial),
    )


PERMISSION_ACTIONS = {
    "yes": "grant",
    "no": "revoke",
    PERMISSION_UNSET: "reset",
}
"""How to write back each decision the reader can report.

`limited` is absent. Nothing sets a partial grant, so the capability refuses
to change one rather than leaving it unrestorable here.
"""

NOWHERE = "none"
"""No location was simulated. Distinct from a location nothing could read."""


def _simulated_location(commands: DriverCommands) -> Optional[str]:
    """Where the device was told it is, as text, or None where it would not say.

    The driver answers with nulls for a location nobody set, so the absence
    of one is readable and `NOWHERE` records it. A reset then puts that back,
    which a pair of coordinates could not express.
    """
    answer = commands.ios_location()
    if answer is None:
        return None
    latitude, longitude = answer.get("latitude"), answer.get("longitude")
    if latitude is None or longitude is None:
        return NOWHERE
    return f"{latitude},{longitude}"


def _services_named(capabilities: Sequence[Capability]) -> set[str]:
    """The privacy services these steps ask for, as simctl names them.

    Read off the capabilities rather than off the kind. One permission step
    changes one service, and reading the other three costs a driver call each
    and can ask about a service this platform refuses.
    """
    return {
        service
        for capability in capabilities
        if isinstance(capability, PermissionCapability)
        and (service := IOS_PRIVACY_SERVICES.get(capability.permission))
    }


def snapshot_ios(
    commands: DriverCommands,
    capabilities: Sequence[Capability] = (),
    app_id: Optional[str] = None,
) -> IosStateSnapshot:
    """The iOS states to put back, read before the scenario runs.

    Only the states these capabilities could change are read, service by
    service rather than kind by kind. A scenario that asks for nothing is not
    worth four driver calls, and a screen-reader read answers nothing before
    iOS 27.

    The driver takes every read. No command line answers any of them, so this
    asks for a driver rather than one that might be absent.
    """
    wanted = {capability.kind for capability in capabilities}
    permissions = None
    services = _services_named(capabilities)
    if services and app_id:
        permissions = {
            service: decision
            for service in services
            if (decision := commands.ios_permission(app_id, service)) is not None
        }
    return IosStateSnapshot(
        permissions=permissions,
        voice_over=(
            commands.ios_voice_over()
            if CapabilityKind.ACCESSIBILITY in wanted
            else None
        ),
        simulated_location=(
            _simulated_location(commands)
            if CapabilityKind.LOCATION in wanted
            else None
        ),
    )


def _put_back(
    what: str,
    write: Callable[[str], None],
    wanted: Optional[str],
    observed: Optional[str],
    events: Optional[list[RestorationEvent]] = None,
    read_back: Optional[Callable[[], Optional[str]]] = None,
) -> None:
    """Write one state back, when it was read and has since moved.

    The writer is handed the value, rather than closing over it, so it is
    called only with a state that was actually observed.

    A failure here is reported and does not propagate. Restoration runs while
    a scenario is ending, often while an exception is on its way out. Losing
    that exception would hide why the scenario ended.
    """
    if wanted is None or observed is None or observed == wanted:
        return
    logger.info("Restoring %s: %s -> %s", what, observed, wanted)
    try:
        write(wanted)
    except (DeviceUnavailable, OSError) as exc:
        logger.warning("Could not restore %s to %s: %s", what, wanted, exc)
        if events is not None:
            events.append(RestorationEvent(what, False, str(exc)))
    else:
        if events is None:
            return
        restored = read_back() if read_back is not None else wanted
        if restored == wanted:
            events.append(RestorationEvent(what, True))
        else:
            events.append(
                RestorationEvent(
                    what,
                    False,
                    f"read-back returned {restored or 'nothing'} instead of {wanted}",
                )
            )


def _restore_radios(
    serial: str,
    found: AndroidStateSnapshot,
    commands: Optional[DriverCommands],
    events: Optional[list[RestorationEvent]] = None,
) -> None:
    """Put the three radios back, in one call where the driver can read them.

    Falls back to airplane mode alone, which is all adb can set. That cannot
    return wifi or data on its own, so a run without a driver keeps the narrower
    guarantee rather than a pretended one.
    """
    if commands is None or found.wifi is None or found.data is None:
        _put_back(
            "airplane mode",
            lambda value: AdbControl.set_airplane_mode(serial, value == "on"),
            found.airplane,
            receipts.airplane_mode(serial),
            events,
            lambda: receipts.airplane_mode(serial),
        )
        return

    now = commands.connectivity()
    if now is None:
        return
    wanted = {
        "wifi": found.wifi == "on",
        "data": found.data == "on",
        "airplane": found.airplane == "on",
    }
    current = {
        "wifi": bool(now.get("wifi")),
        "data": bool(now.get("data")),
        "airplane": bool(now.get("airplaneMode")),
    }
    if current == wanted:
        return
    logger.info("Restoring the radios: %s -> %s", current, wanted)
    try:
        commands.set_connectivity(**wanted)
    except DeviceUnavailable as exc:
        logger.warning("Could not restore the radios to %s: %s", wanted, exc)
        if events is not None:
            events.append(RestorationEvent("the radios", False, str(exc)))
    else:
        if events is not None:
            restored = commands.connectivity()
            restored_state = {
                "wifi": bool((restored or {}).get("wifi")),
                "data": bool((restored or {}).get("data")),
                "airplane": bool((restored or {}).get("airplaneMode")),
            }
            if restored is not None and restored_state == wanted:
                events.append(RestorationEvent("the radios", True))
            else:
                events.append(
                    RestorationEvent("the radios", False, "radio read-back did not match")
                )


def restore_android(
    serial: str,
    package: Optional[str],
    found: AndroidStateSnapshot,
    commands: Optional[DriverCommands] = None,
    events: Optional[list[RestorationEvent]] = None,
) -> None:
    """Put back every state that moved during the scenario."""
    _restore_radios(serial, found, commands, events)
    # Reset rather than written back: an override is either on or off, so
    # there is no level to restore it to, only the real battery.
    _put_back(
        "battery",
        lambda _: AdbControl.reset_battery(serial),
        found.battery,
        receipts.battery_level(serial),
        events,
        lambda: receipts.battery_level(serial),
    )
    _put_back(
        "timezone",
        lambda value: AdbControl.set_timezone(serial, value),
        found.timezone,
        receipts.timezone(serial),
        events,
        lambda: receipts.timezone(serial),
    )
    if package:
        _put_back(
            "app locale",
            lambda value: AdbControl.set_app_locale(serial, package, value),
            found.locale,
            receipts.app_locale(serial, package),
            events,
            lambda: receipts.app_locale(serial, package),
        )
    _put_back(
        "the accessibility services",
        lambda value: AdbControl.set_accessibility_services(serial, value),
        found.accessibility,
        AdbControl.enabled_accessibility_services(serial),
        events,
        lambda: AdbControl.enabled_accessibility_services(serial),
    )


def _restore_location(
    found: IosStateSnapshot,
    commands: DriverCommands,
    events: Optional[list[RestorationEvent]] = None,
) -> None:
    """Put the simulated location back, or take it away again.

    A reset is the only way back to no location, and a pair of coordinates
    is the way back to one. Which of the two is needed is why the snapshot
    records `NOWHERE` rather than None for a device nobody had placed.
    """

    def write(value: str) -> None:
        if value == NOWHERE:
            commands.reset_ios_location()
            return
        latitude, _, longitude = value.partition(",")
        commands.set_ios_location(float(latitude), float(longitude))

    _put_back(
        "the simulated location",
        write,
        found.simulated_location,
        _simulated_location(commands),
        events,
        lambda: _simulated_location(commands),
    )


def _restore_permissions(
    udid: str,
    app_id: Optional[str],
    found: IosStateSnapshot,
    commands: DriverCommands,
    events: Optional[list[RestorationEvent]] = None,
) -> None:
    """Put back each decision the scenario could have changed.

    Written through simctl rather than the driver. The driver takes a
    boolean, and `unset` is a third state that only `simctl privacy reset`
    reaches.
    """
    if not found.permissions or not app_id:
        return
    for service, decision in found.permissions.items():
        _put_back(
            f"the {service} permission",
            lambda value, service=service: SimctlControl.set_privacy(
                udid, PERMISSION_ACTIONS[value], service, app_id
            ),
            decision if decision in PERMISSION_ACTIONS else None,
            commands.ios_permission(app_id, service),
            events,
            lambda service=service: commands.ios_permission(app_id, service),
        )


def restore_ios(
    udid: str,
    app_id: Optional[str],
    found: IosStateSnapshot,
    commands: DriverCommands,
    events: Optional[list[RestorationEvent]] = None,
) -> None:
    """Put back every iOS state this snapshot read.

    A state the snapshot did not read is not read a second time. Nothing
    could be written back to it, and the read costs a driver call.
    """
    if found.voice_over is not None:
        _put_back(
            "VoiceOver",
            lambda value: commands.set_ios_voice_over(value == "on"),
            _switch(found.voice_over),
            _switch(commands.ios_voice_over()),
            events,
            lambda: _switch(commands.ios_voice_over()),
        )
    if found.simulated_location is not None:
        _restore_location(found, commands, events)
    _restore_permissions(udid, app_id, found, commands, events)


class RestorableDevice(Protocol):
    """What this module reads off a session to put its states back.

    Stated rather than reached for, so a caller holding something else fails
    where it is written and not inside a `finally` on the way out.
    """

    platform: Platform
    serial: Optional[str]
    package_id: Optional[str]
    app_id: Optional[str]
    driver: object


@contextmanager
def preserved(
    device_control: RestorableDevice,
    package: Optional[str] = None,
    capabilities: Sequence[Capability] = (),
    events: Optional[list[RestorationEvent]] = None,
):
    """Hands the device back as the scenario found it, whatever ended it.

    Restoration is in a `finally`, so it runs after a pass, after a failure,
    and after an interruption.

    `capabilities` are the typed states the steps ask for. Android reads its
    states through adb either way, which is cheap and needs no driver. iOS
    reads each state through the driver, so it reads only what these
    capabilities could change, down to the privacy service. Asking for a
    screen reader on iOS 18 answers nothing and warns about it, once per
    scenario.

    Not every state is covered. Android puts back the radios, the battery,
    the clock, the app locale and the accessibility services. iOS puts back
    the screen reader, the simulated location, and each privacy decision a
    step asked for. Those are the states a supported iOS capability changes
    and something reads.
    """
    platform = device_control.platform
    serial = device_control.serial or ""
    driver = device_control.driver
    commands = DriverCommands(driver) if driver is not None else None

    if platform == Platform.ANDROID and serial:
        package = package or device_control.package_id
        android_found = snapshot_android(serial, package, commands)
        logger.info("Device state before the scenario: %s", android_found.describe())
        try:
            yield android_found
        finally:
            restore_android(serial, package, android_found, commands, events)
        return

    if platform == Platform.IOS and commands is not None:
        app_id = device_control.app_id
        ios_found = snapshot_ios(commands, capabilities, app_id)
        logger.info("Device state before the scenario: %s", ios_found.describe())
        try:
            yield ios_found
        finally:
            restore_ios(serial, app_id, ios_found, commands, events)
        return

    yield None
