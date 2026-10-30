"""iOS device-log sources exposed by XCUITest."""

from yewo.devices.logs.records import DeviceLogSource


IOS_LOG_SOURCES = (
    DeviceLogSource(log_type="syslog", artifact_stem="syslog"),
    DeviceLogSource(
        log_type="crashlog",
        artifact_stem="crashlog",
        is_diagnostic=True,
    ),
)
