"""Compose platform-specific readers behind one device-log boundary."""

from yewo.applications.records import ApplicationDescriptor
from yewo.devices.inventory import DeviceDescriptor, DeviceKind, DevicePlatform
from yewo.devices.logs.appium import AppiumDeviceLogReader
from yewo.devices.logs.reader import DeviceLogReader
from yewo.devices.logs.router import RoutedDeviceLogReader


def build_platform_device_log_reader(
    driver: object,
    device: DeviceDescriptor,
    application: ApplicationDescriptor,
) -> DeviceLogReader:
    """Return the source-specific reader composition for one device session."""
    appium_reader = AppiumDeviceLogReader(driver)
    if device.platform is DevicePlatform.ANDROID:
        from yewo.devices.android.adb import AdbClient
        from yewo.devices.android.diagnostics import AdbApplicationExitReader

        connection_id = device.connection_id or device.target_id
        return RoutedDeviceLogReader(
            {
                "logcat": appium_reader,
                "exitinfo": AdbApplicationExitReader(
                    AdbClient(),
                    connection_id,
                    application.application_id,
                ),
            }
        )
    if device.kind is not DeviceKind.PHYSICAL:
        from yewo.devices.ios.appium_crashlogs import IosAppiumCrashLogReader

        return RoutedDeviceLogReader(
            {
                "syslog": appium_reader,
                "crashlog": IosAppiumCrashLogReader(
                    driver,
                    application.application_id,
                ),
            }
        )
    from yewo.devices.ios.crashlogs import DevicectlCrashLogReader

    device_address = device.connection_id or device.target_id
    return RoutedDeviceLogReader(
        {
            "syslog": appium_reader,
            "crashlog": DevicectlCrashLogReader(
                device_address,
                application.application_id,
            ),
        }
    )
