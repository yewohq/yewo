"""iOS device discovery and simulator control."""

from yewo.devices.ios.application import IosApplicationInspector, IosApplicationPresence
from yewo.devices.ios.devicectl import DevicectlClient
from yewo.devices.ios.inventory import (
    IosDeviceInventory,
    IosPhysicalDeviceInventory,
    SimctlDeviceInventory,
)
from yewo.devices.ios.lifecycle import IosDeviceLifecycle
from yewo.devices.ios.options import IosDriverOptionsBuilder
from yewo.devices.ios.simctl import SimctlClient

__all__ = [
    "IosApplicationInspector",
    "IosApplicationPresence",
    "DevicectlClient",
    "IosDeviceInventory",
    "IosDeviceLifecycle",
    "IosDriverOptionsBuilder",
    "IosPhysicalDeviceInventory",
    "SimctlClient",
    "SimctlDeviceInventory",
]
