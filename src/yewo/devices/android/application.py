"""Observe Android application installation state."""

from __future__ import annotations

from yewo.devices.android.adb import AdbClient
from yewo.devices.inventory import DeviceDescriptor, DevicePlatform


class AndroidApplicationPresence:
    """Read installed Android package IDs through ADB."""

    def __init__(self, adb: AdbClient) -> None:
        self._adb = adb

    def is_installed(
        self,
        device: DeviceDescriptor,
        application_id: str,
    ) -> bool | None:
        if device.platform is not DevicePlatform.ANDROID or device.connection_id is None:
            return None
        packages = self._adb.installed_packages(device.connection_id)
        return None if packages is None else application_id in packages
