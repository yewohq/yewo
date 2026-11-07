"""Android device discovery and control."""

from yewo.devices.android.application import AndroidApplicationPresence
from yewo.devices.android.adb import AdbClient, VisualFeedbackSettings
from yewo.devices.android.diagnostics import AdbApplicationExitReader
from yewo.devices.android.apk import ApkInfo, ApkInspector
from yewo.devices.android.inventory import AdbDeviceInventory
from yewo.devices.android.lifecycle import AndroidDeviceLifecycle
from yewo.devices.android.host_environment import MacOsAndroidEmulatorEnvironment
from yewo.devices.android.logs import ANDROID_LOG_SOURCES
from yewo.devices.android.options import AndroidDriverOptionsBuilder

__all__ = [
    "AdbClient",
    "AdbApplicationExitReader",
    "AdbDeviceInventory",
    "ANDROID_LOG_SOURCES",
    "AndroidApplicationPresence",
    "AndroidDeviceLifecycle",
    "MacOsAndroidEmulatorEnvironment",
    "AndroidDriverOptionsBuilder",
    "ApkInfo",
    "ApkInspector",
    "VisualFeedbackSettings",
]
