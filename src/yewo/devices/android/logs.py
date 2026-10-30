"""Android device-log sources exposed by UiAutomator2."""

from yewo.devices.logs.records import DeviceLogSource


ANDROID_LOG_SOURCES = (DeviceLogSource(log_type="logcat", artifact_stem="logcat"),)
