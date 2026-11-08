"""Tests for Android application installation presence."""

from devices._runner import CommandRunnerStub
from yewo.devices import (
    DeviceDescriptor,
    DeviceKind,
    DevicePlatform,
    DeviceReadiness,
)
from yewo.devices.android import AdbClient, AndroidApplicationPresence


def _device() -> DeviceDescriptor:
    return DeviceDescriptor(
        platform=DevicePlatform.ANDROID,
        target_id="Pixel_9",
        connection_id="emulator-5554",
        display_name="Pixel 9",
        kind=DeviceKind.VIRTUAL,
        readiness=DeviceReadiness.READY,
        native_state="device",
    )


def test_android_presence_reads_installed_packages() -> None:
    runner = CommandRunnerStub(stdout="package:com.example.app\npackage:com.android.settings\n")

    is_installed = AndroidApplicationPresence(AdbClient(runner)).is_installed(
        _device(),
        "com.example.app",
    )

    assert is_installed is True
    assert runner.calls == [
        (
            (
                "adb",
                "-s",
                "emulator-5554",
                "shell",
                "pm",
                "list",
                "packages",
            ),
            20,
        )
    ]


def test_android_presence_preserves_unknown_command_failure() -> None:
    runner = CommandRunnerStub(returncode=1)

    assert (
        AndroidApplicationPresence(AdbClient(runner)).is_installed(
            _device(),
            "com.example.app",
        )
        is None
    )
