"""Snapshot and restore target-specific network shaping."""

from typing import Optional, Sequence

from yewo.agent.actuation.actions import Capability, NetworkCapability
from yewo.agent.device.capability_maps import SLOW_NETWORK_MODES, active_ios_condition
from yewo.agent.device.driver_commands import DriverCommands
from yewo.agent.device.emulator_network import (
    EmulatorNetworkShape,
    network_shape,
    set_network_shape,
)
from yewo.errors import DeviceUnavailable

NO_CONDITION = ("", "")
"""A readable iOS condition list with no active condition."""


def _asks_for_slow_network(capabilities: Sequence[Capability]) -> bool:
    return any(
        isinstance(item, NetworkCapability) and item.mode in SLOW_NETWORK_MODES
        for item in capabilities
    )


def snapshot_android_network(
    serial: str,
    capabilities: Sequence[Capability],
    commands: Optional[DriverCommands] = None,
) -> Optional[EmulatorNetworkShape]:
    return network_shape(serial, commands) if _asks_for_slow_network(capabilities) else None


def snapshot_ios_condition(
    commands: DriverCommands, capabilities: Sequence[Capability]
) -> Optional[tuple[str, str]]:
    if not _asks_for_slow_network(capabilities):
        return None
    conditions = commands.condition_inducers()
    if conditions is None:
        return None
    return active_ios_condition(conditions) or NO_CONDITION


def restore_android_network(
    serial: str,
    found: Optional[EmulatorNetworkShape],
    commands: Optional[DriverCommands] = None,
) -> Optional[tuple[bool, Optional[str]]]:
    if found is None or network_shape(serial, commands) == found:
        return None
    try:
        set_network_shape(serial, found.speed_argument(), found.delay_argument())
    except DeviceUnavailable as exc:
        return False, str(exc)
    return network_shape(serial, commands) == found, None


def restore_ios_condition(
    commands: DriverCommands, found: Optional[tuple[str, str]]
) -> Optional[tuple[bool, Optional[str]]]:
    if found is None:
        return None
    conditions = commands.condition_inducers()
    if conditions is None:
        return None
    current = active_ios_condition(conditions)
    wanted = None if found == NO_CONDITION else found
    if current == wanted:
        return None
    try:
        if current is not None:
            commands.disable_condition_inducer()
        if wanted is not None:
            commands.enable_condition_inducer(*wanted)
    except DeviceUnavailable as exc:
        return False, str(exc)
    after = commands.condition_inducers()
    restored = active_ios_condition(after) if after is not None else None
    return restored == wanted, None
