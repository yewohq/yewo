"""iOS device discovery and simulator control."""

from yewo.devices.ios.application import IosApplicationInspector, IosApplicationPresence
from yewo.devices.ios.crashlogs import DevicectlCrashLogReader
from yewo.devices.ios.devicectl import DevicectlClient
from yewo.devices.ios.inventory import (
    IosDeviceInventory,
    IosPhysicalDeviceInventory,
    SimctlDeviceInventory,
)
from yewo.devices.ios.lifecycle import IosDeviceLifecycle
from yewo.devices.ios.logs import IOS_LOG_SOURCES
from yewo.devices.ios.options import IosDriverOptionsBuilder, IosWebDriverAgentSigning
from yewo.devices.ios.simctl import SimctlClient

__all__ = [
    "IOS_LOG_SOURCES",
    "IosApplicationInspector",
    "IosApplicationPresence",
    "DevicectlClient",
    "DevicectlCrashLogReader",
    "IosDeviceInventory",
    "IosDeviceLifecycle",
    "IosDriverOptionsBuilder",
    "IosWebDriverAgentSigning",
    "IosPhysicalDeviceInventory",
    "SimctlClient",
    "SimctlDeviceInventory",
]
