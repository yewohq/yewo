"""Tests for iOS simulator discovery through simctl."""

import json

import pytest

from devices._runner import CommandRunnerStub
from yewo.devices.errors import DeviceInventoryError
from yewo.devices.inventory import (
    DeviceInventory,
    DeviceKind,
    DevicePlatform,
    DeviceReadiness,
)
from yewo.devices.ios import SimctlDeviceInventory


def _simctl_output() -> str:
    return json.dumps(
        {
            "devices": {
                "com.apple.CoreSimulator.SimRuntime.iOS-18-2": [
                    {
                        "name": "iPhone 16 Pro",
                        "udid": "BOOTED-UDID",
                        "state": "Booted",
                        "isAvailable": True,
                    },
                    {
                        "name": "iPad Air",
                        "udid": "STOPPED-UDID",
                        "state": "Shutdown",
                        "isAvailable": True,
                    },
                    {
                        "name": "Unavailable simulator",
                        "udid": "UNAVAILABLE-UDID",
                        "state": "Shutdown",
                        "isAvailable": False,
                    },
                ],
                "com.apple.CoreSimulator.SimRuntime.tvOS-18-2": [
                    {
                        "name": "Apple TV",
                        "udid": "TV-UDID",
                        "state": "Shutdown",
                        "isAvailable": True,
                    }
                ],
            }
        }
    )


def test_simctl_inventory_normalizes_ios_simulators() -> None:
    runner = CommandRunnerStub(stdout=_simctl_output())
    inventory: DeviceInventory = SimctlDeviceInventory(runner)

    devices = inventory.list_devices()

    assert [device.target_id for device in devices] == ["STOPPED-UDID", "BOOTED-UDID"]
    ipad, iphone = devices
    assert ipad.platform is DevicePlatform.IOS
    assert ipad.kind is DeviceKind.VIRTUAL
    assert ipad.readiness is DeviceReadiness.STARTABLE
    assert ipad.native_state == "Shutdown"
    assert ipad.operating_system_version == "18.2"
    assert iphone.readiness is DeviceReadiness.READY
    assert runner.calls == [(("xcrun", "simctl", "list", "devices", "available", "--json"), 30)]


def test_simctl_inventory_rejects_invalid_json() -> None:
    runner = CommandRunnerStub(stdout="not json")

    with pytest.raises(DeviceInventoryError, match="invalid JSON"):
        SimctlDeviceInventory(runner).list_devices()


@pytest.mark.parametrize(
    ("native_state", "readiness"),
    [
        ("Creating", DeviceReadiness.TRANSITIONING),
        ("Booting", DeviceReadiness.TRANSITIONING),
        ("Shutting Down", DeviceReadiness.TRANSITIONING),
        ("Future State", DeviceReadiness.UNKNOWN),
    ],
)
def test_simctl_inventory_preserves_native_state(
    native_state: str, readiness: DeviceReadiness
) -> None:
    output = json.dumps(
        {
            "devices": {
                "com.apple.CoreSimulator.SimRuntime.iOS-18-2": [
                    {
                        "name": "iPhone 16",
                        "udid": "STATE-UDID",
                        "state": native_state,
                    }
                ]
            }
        }
    )

    device = SimctlDeviceInventory(CommandRunnerStub(stdout=output)).list_devices()[0]

    assert device.readiness is readiness
    assert device.native_state == native_state


def test_simctl_inventory_requires_a_udid() -> None:
    output = json.dumps(
        {
            "devices": {
                "com.apple.CoreSimulator.SimRuntime.iOS-18-2": [
                    {"name": "iPhone 16", "state": "Shutdown"}
                ]
            }
        }
    )

    with pytest.raises(DeviceInventoryError, match="required udid"):
        SimctlDeviceInventory(CommandRunnerStub(stdout=output)).list_devices()
