"""Tests for Android discovery through ADB."""

import pytest

from devices._runner import CommandRunnerStub
from yewo.devices.android import AdbDeviceInventory
from yewo.devices.errors import DeviceInventoryError
from yewo.devices.inventory import (
    DeviceInventory,
    DeviceKind,
    DevicePlatform,
    DeviceReadiness,
)


ADB_OUTPUT = """List of devices attached
emulator-5554 device product:sdk model:Pixel_9_API_35 transport_id:1
R58M123 unauthorized usb:1-1 product:e3q model:Galaxy_S24 transport_id:2
"""


def test_adb_inventory_normalizes_connected_devices() -> None:
    runner = CommandRunnerStub(
        responses=(
            (ADB_OUTPUT, "", 0),
            ("Pixel_9_API_35\n", "", 0),
            ("15\n", "", 0),
            ("Pixel_9_API_35\nPixel_8\n", "", 0),
        )
    )
    inventory: DeviceInventory = AdbDeviceInventory(runner, emulator_executable="emulator")

    devices = inventory.list_devices()

    assert [device.target_id for device in devices] == ["R58M123", "Pixel_8", "Pixel_9_API_35"]
    phone, stopped_emulator, running_emulator = devices
    assert phone.platform is DevicePlatform.ANDROID
    assert phone.display_name == "Galaxy S24"
    assert phone.kind is DeviceKind.PHYSICAL
    assert phone.readiness is DeviceReadiness.NOT_READY
    assert phone.native_state == "unauthorized"
    assert stopped_emulator.readiness is DeviceReadiness.STARTABLE
    assert running_emulator.kind is DeviceKind.VIRTUAL
    assert running_emulator.connection_id == "emulator-5554"
    assert running_emulator.readiness is DeviceReadiness.READY
    assert running_emulator.native_state == "device"
    assert running_emulator.operating_system_version == "15"
    assert runner.calls == [
        (("adb", "devices", "-l"), 30),
        (("adb", "-s", "emulator-5554", "shell", "getprop", "ro.boot.qemu.avd_name"), 10),
        (("adb", "-s", "emulator-5554", "shell", "getprop", "ro.build.version.release"), 20),
        (("emulator", "-list-avds"), 30),
    ]


@pytest.mark.parametrize(
    ("native_state", "readiness"),
    [
        ("connecting", DeviceReadiness.TRANSITIONING),
        ("authorizing", DeviceReadiness.TRANSITIONING),
        ("offline", DeviceReadiness.NOT_READY),
        ("no permissions", DeviceReadiness.NOT_READY),
        ("bootloader", DeviceReadiness.NOT_READY),
        ("recovery", DeviceReadiness.NOT_READY),
        ("rescue", DeviceReadiness.NOT_READY),
        ("sideload", DeviceReadiness.NOT_READY),
        ("future-state", DeviceReadiness.UNKNOWN),
    ],
)
def test_adb_inventory_preserves_native_state(
    native_state: str, readiness: DeviceReadiness
) -> None:
    runner = CommandRunnerStub(stdout=f"List of devices attached\nR58M123 {native_state}\n")

    devices = AdbDeviceInventory(runner, include_avds=False).list_devices()

    assert devices[0].readiness is readiness
    assert devices[0].native_state == native_state


def test_adb_inventory_rejects_a_malformed_row() -> None:
    runner = CommandRunnerStub(stdout="List of devices attached\nmissing-state\n")

    with pytest.raises(DeviceInventoryError, match="invalid device row"):
        AdbDeviceInventory(runner, include_avds=False).list_devices()


def test_adb_inventory_reports_command_failure() -> None:
    runner = CommandRunnerStub(stderr="ADB server did not answer.", returncode=1)

    with pytest.raises(DeviceInventoryError, match="exit code was 1"):
        AdbDeviceInventory(runner, include_avds=False).list_devices()
