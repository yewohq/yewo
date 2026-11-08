"""Discover iOS simulators and attached physical devices."""

from __future__ import annotations

import json
import re

from yewo.devices._commands import CommandRunner, capture_optional, run_command
from yewo.devices.errors import DeviceCommandError, DeviceInventoryError
from yewo.devices.inventory import (
    DeviceDescriptor,
    DeviceKind,
    DevicePlatform,
    DeviceReadiness,
)
from yewo.devices.ios.simctl import SimctlClient


_SIMULATOR_READINESS = {
    "Booted": DeviceReadiness.READY,
    "Shutdown": DeviceReadiness.STARTABLE,
    "Creating": DeviceReadiness.TRANSITIONING,
    "Booting": DeviceReadiness.TRANSITIONING,
    "Shutting Down": DeviceReadiness.TRANSITIONING,
}
_IOS_RUNTIME_MARKER = ".iOS-"
_ATTACHED_DEVICE = re.compile(
    r"^(?P<name>.+?)\s+\((?P<release>\d[\d.]*)\)\s+"
    r"\((?P<target_id>[0-9A-Fa-f-]{8,})\)$"
)


def _ios_version(runtime_identifier: str) -> str | None:
    if _IOS_RUNTIME_MARKER not in runtime_identifier:
        return None
    return runtime_identifier.partition(_IOS_RUNTIME_MARKER)[2].replace("-", ".")


def _required_string(entry: dict[object, object], field: str) -> str:
    value = entry.get(field)
    if not isinstance(value, str) or not value.strip():
        raise DeviceInventoryError(f"simctl omitted the required {field} field.")
    return value.strip()


def _parse_simulators(output: str) -> tuple[DeviceDescriptor, ...]:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as error:
        raise DeviceInventoryError("simctl returned invalid JSON.") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("devices"), dict):
        raise DeviceInventoryError("simctl returned an invalid device inventory.")

    devices: list[DeviceDescriptor] = []
    for runtime_identifier, entries in payload["devices"].items():
        if not isinstance(runtime_identifier, str) or not isinstance(entries, list):
            raise DeviceInventoryError("simctl returned an invalid runtime entry.")
        operating_system_version = _ios_version(runtime_identifier)
        if operating_system_version is None:
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                raise DeviceInventoryError("simctl returned an invalid device entry.")
            if entry.get("isAvailable") is False:
                continue
            target_id = _required_string(entry, "udid")
            display_name = _required_string(entry, "name")
            native_state = _required_string(entry, "state")
            devices.append(
                DeviceDescriptor(
                    platform=DevicePlatform.IOS,
                    target_id=target_id,
                    connection_id=target_id,
                    display_name=display_name,
                    kind=DeviceKind.VIRTUAL,
                    readiness=_SIMULATOR_READINESS.get(native_state, DeviceReadiness.UNKNOWN),
                    native_state=native_state,
                    operating_system_version=operating_system_version,
                    device_model=display_name,
                )
            )
    return tuple(sorted(devices, key=lambda device: (device.display_name, device.target_id)))


def _parse_attached_devices(output: str) -> tuple[DeviceDescriptor, ...]:
    devices: list[DeviceDescriptor] = []
    in_device_section = False
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("=="):
            if in_device_section:
                break
            in_device_section = stripped == "== Devices =="
            continue
        if not in_device_section:
            continue
        device_match = _ATTACHED_DEVICE.match(stripped)
        if device_match:
            target_id = device_match["target_id"]
            devices.append(
                DeviceDescriptor(
                    platform=DevicePlatform.IOS,
                    target_id=target_id,
                    connection_id=target_id,
                    display_name=device_match["name"],
                    kind=DeviceKind.PHYSICAL,
                    readiness=DeviceReadiness.READY,
                    native_state="connected",
                    operating_system_version=device_match["release"],
                )
            )
    return tuple(devices)


class SimctlDeviceInventory:
    """List iOS simulators through one shared simctl client."""

    def __init__(
        self,
        runner: CommandRunner = run_command,
        *,
        xcrun_executable: str = "xcrun",
    ) -> None:
        self._simctl = SimctlClient(runner, xcrun_executable=xcrun_executable)

    def list_devices(self) -> tuple[DeviceDescriptor, ...]:
        try:
            listing = self._simctl.device_listing()
        except DeviceCommandError as error:
            raise DeviceInventoryError(str(error)) from error
        return _parse_simulators(listing)


class IosPhysicalDeviceInventory:
    """List attached iPhones and iPads without importing simulator control."""

    def __init__(
        self,
        runner: CommandRunner = run_command,
        *,
        xcrun_executable: str = "xcrun",
        idevice_id_executable: str = "idevice_id",
    ) -> None:
        self._runner = runner
        self._xcrun_executable = xcrun_executable
        self._idevice_id_executable = idevice_id_executable

    def list_devices(self) -> tuple[DeviceDescriptor, ...]:
        output = capture_optional(
            self._runner,
            [self._xcrun_executable, "xctrace", "list", "devices"],
            timeout_seconds=30,
        )
        if output:
            devices = _parse_attached_devices(output)
            if devices:
                return devices
        fallback = capture_optional(
            self._runner,
            [self._idevice_id_executable, "-l"],
            timeout_seconds=10,
        )
        return tuple(
            DeviceDescriptor(
                platform=DevicePlatform.IOS,
                target_id=target_id,
                connection_id=target_id,
                display_name=target_id,
                kind=DeviceKind.PHYSICAL,
                readiness=DeviceReadiness.READY,
                native_state="connected",
            )
            for target_id in (line.strip() for line in (fallback or "").splitlines())
            if target_id
        )


class IosDeviceInventory:
    """Combine attached iOS devices with CoreSimulator targets."""

    def __init__(
        self,
        simulators: SimctlDeviceInventory,
        physical_devices: IosPhysicalDeviceInventory,
    ) -> None:
        self._simulators = simulators
        self._physical_devices = physical_devices

    def list_devices(self) -> tuple[DeviceDescriptor, ...]:
        devices = (*self._physical_devices.list_devices(), *self._simulators.list_devices())
        return tuple(
            sorted(
                devices, key=lambda device: (device.kind is DeviceKind.VIRTUAL, device.display_name)
            )
        )
