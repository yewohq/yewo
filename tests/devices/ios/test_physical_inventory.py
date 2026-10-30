"""Tests for attached iOS hardware discovery."""

from devices._runner import CommandRunnerStub
from yewo.devices.inventory import DeviceKind, DeviceReadiness
from yewo.devices.ios import (
    IosDeviceInventory,
    IosPhysicalDeviceInventory,
    SimctlDeviceInventory,
)


XCTRACE_OUTPUT = """== Devices ==
Studio Mac (7BB293E1-5FE9-564C-8A69-F2EAFFB24CD4)
QA iPhone (18.2) (00008140-00096DC43A52801C)

== Simulators ==
iPhone 16 Pro (18.2) (2506D933-6214-4860-BFF2-6D3193F0B801)
"""


def test_xctrace_inventory_keeps_only_attached_ios_hardware() -> None:
    inventory = IosPhysicalDeviceInventory(CommandRunnerStub(stdout=XCTRACE_OUTPUT))

    devices = inventory.list_devices()

    assert len(devices) == 1
    assert devices[0].display_name == "QA iPhone"
    assert devices[0].target_id == "00008140-00096DC43A52801C"
    assert devices[0].operating_system_version == "18.2"
    assert devices[0].kind is DeviceKind.PHYSICAL
    assert devices[0].readiness is DeviceReadiness.READY


def test_idevice_id_is_used_when_xctrace_has_no_phone() -> None:
    runner = CommandRunnerStub(
        responses=(("== Devices ==\nStudio Mac (HOST-ID)\n", "", 0), ("PHONE-UDID\n", "", 0))
    )

    devices = IosPhysicalDeviceInventory(runner).list_devices()

    assert [device.target_id for device in devices] == ["PHONE-UDID"]
    assert [call[0][0] for call in runner.calls] == ["xcrun", "idevice_id"]


def test_combined_ios_inventory_lists_physical_devices_first() -> None:
    physical_runner = CommandRunnerStub(stdout=XCTRACE_OUTPUT)
    simulator_runner = CommandRunnerStub(
        stdout='{"devices":{"com.apple.CoreSimulator.SimRuntime.iOS-18-2":['
        '{"name":"iPhone 16 Pro","udid":"SIM-UDID","state":"Shutdown"}]}}'
    )
    inventory = IosDeviceInventory(
        SimctlDeviceInventory(simulator_runner),
        IosPhysicalDeviceInventory(physical_runner),
    )

    devices = inventory.list_devices()

    assert [device.kind for device in devices] == [DeviceKind.PHYSICAL, DeviceKind.VIRTUAL]
