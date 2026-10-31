"""Device discovery and lifecycle boundaries."""

from yewo.devices.focus import FocusedFieldReading
from yewo.devices.inventory import (
    DeviceDescriptor,
    DeviceInventory,
    DeviceKind,
    DevicePlatform,
    DeviceQuery,
    DeviceReadiness,
)
from yewo.devices.lifecycle import DeviceLease, DeviceLifecycle
from yewo.devices.screen import (
    ScreenCalibration,
    ScreenCoordinateSpace,
    ScreenRegionReading,
    SystemRegionKind,
)
from yewo.devices.selection import select_device
from yewo.resources import ResourceOwnership

__all__ = [
    "DeviceDescriptor",
    "DeviceInventory",
    "DeviceKind",
    "DeviceLease",
    "DeviceLifecycle",
    "DevicePlatform",
    "DeviceQuery",
    "DeviceReadiness",
    "FocusedFieldReading",
    "ResourceOwnership",
    "ScreenCalibration",
    "ScreenCoordinateSpace",
    "ScreenRegionReading",
    "SystemRegionKind",
    "select_device",
]
