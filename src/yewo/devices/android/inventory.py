"""Discover connected Android devices and configured AVDs."""

from __future__ import annotations

from yewo.devices._commands import CommandRunner, run_command
from yewo.devices.android.adb import AdbClient
from yewo.devices.errors import DeviceCommandError, DeviceInventoryError
from yewo.devices.inventory import (
    DeviceDescriptor,
    DeviceKind,
    DevicePlatform,
    DeviceReadiness,
)


_ADB_READINESS = {
    "device": DeviceReadiness.READY,
    "connecting": DeviceReadiness.TRANSITIONING,
    "authorizing": DeviceReadiness.TRANSITIONING,
    "offline": DeviceReadiness.NOT_READY,
    "unauthorized": DeviceReadiness.NOT_READY,
    "no permissions": DeviceReadiness.NOT_READY,
    "bootloader": DeviceReadiness.NOT_READY,
    "recovery": DeviceReadiness.NOT_READY,
    "rescue": DeviceReadiness.NOT_READY,
    "sideload": DeviceReadiness.NOT_READY,
    "host": DeviceReadiness.NOT_READY,
}


def _adb_row(line: str) -> tuple[str, str, dict[str, str]]:
    parts = line.split()
    if len(parts) < 2:
        raise DeviceInventoryError("ADB returned an invalid device row.")
    connection_id = parts[0]
    if parts[1:3] == ["no", "permissions"]:
        native_state = "no permissions"
        attribute_tokens = parts[3:]
    else:
        native_state = parts[1]
        attribute_tokens = parts[2:]
    attributes = {
        key: value
        for token in attribute_tokens
        if ":" in token
        for key, value in (token.split(":", 1),)
    }
    return connection_id, native_state, attributes


class AdbDeviceInventory:
    """List ADB targets and AVDs through one shared ADB client."""

    def __init__(
        self,
        runner: CommandRunner = run_command,
        *,
        adb_executable: str = "adb",
        emulator_executable: str | None = None,
        include_avds: bool = True,
    ) -> None:
        self._adb = AdbClient(
            runner,
            adb_executable=adb_executable,
            emulator_executable=emulator_executable,
        )
        self._include_avds = include_avds

    def _connected_devices(self) -> dict[str, DeviceDescriptor]:
        devices: dict[str, DeviceDescriptor] = {}
        try:
            listing = self._adb.device_listing()
        except DeviceCommandError as error:
            raise DeviceInventoryError(str(error)) from error
        for line in listing.splitlines():
            stripped = line.strip()
            if not stripped or stripped == "List of devices attached":
                continue
            connection_id, native_state, attributes = _adb_row(stripped)
            is_emulator = connection_id.startswith("emulator-")
            avd_name = self._adb.avd_name(connection_id) if is_emulator else None
            model = attributes.get("model")
            model_name = model.replace("_", " ") if model else None
            target_id = avd_name or connection_id
            display_name = avd_name or model_name or connection_id
            operating_system_version = None
            if native_state == "device":
                version = self._adb.ask(
                    connection_id,
                    "getprop",
                    "ro.build.version.release",
                )
                operating_system_version = version.strip() if version and version.strip() else None
            devices[target_id] = DeviceDescriptor(
                platform=DevicePlatform.ANDROID,
                target_id=target_id,
                connection_id=connection_id,
                display_name=display_name,
                kind=DeviceKind.VIRTUAL if is_emulator else DeviceKind.PHYSICAL,
                readiness=_ADB_READINESS.get(native_state, DeviceReadiness.UNKNOWN),
                native_state=native_state,
                operating_system_version=operating_system_version,
                device_model=model_name,
                architecture=attributes.get("abi"),
            )
        return devices

    def list_devices(self) -> tuple[DeviceDescriptor, ...]:
        """Return connected targets and configured AVDs without duplicates."""
        devices = self._connected_devices()
        avd_names = self._adb.list_avds() if self._include_avds else None
        for avd_name in avd_names or ():
            devices.setdefault(
                avd_name,
                DeviceDescriptor(
                    platform=DevicePlatform.ANDROID,
                    target_id=avd_name,
                    display_name=avd_name,
                    kind=DeviceKind.VIRTUAL,
                    readiness=DeviceReadiness.STARTABLE,
                    native_state="stopped",
                ),
            )
        return tuple(
            sorted(devices.values(), key=lambda device: (device.display_name, device.target_id))
        )
