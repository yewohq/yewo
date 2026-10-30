"""Tests for iOS device startup and ownership."""

import pytest

from devices._runner import CommandRunnerStub
from yewo.devices.errors import DeviceCommandError, DeviceLifecycleError
from yewo.devices.inventory import (
    DeviceDescriptor,
    DeviceKind,
    DevicePlatform,
    DeviceReadiness,
)
from yewo.devices.ios import IosDeviceLifecycle, SimctlClient
from yewo.resources import ResourceOwnership


def _ios_device(
    *,
    kind: DeviceKind = DeviceKind.VIRTUAL,
    readiness: DeviceReadiness = DeviceReadiness.STARTABLE,
    native_state: str = "Shutdown",
) -> DeviceDescriptor:
    return DeviceDescriptor(
        platform=DevicePlatform.IOS,
        target_id="SIMULATOR-UDID",
        connection_id="SIMULATOR-UDID",
        display_name="iPhone 17",
        kind=kind,
        readiness=readiness,
        native_state=native_state,
    )


def test_stopped_simulator_is_started_waited_for_and_stopped_once() -> None:
    runner = CommandRunnerStub(responses=(("", "", 0), ("", "", 0), ("", "", 0)))

    lease = IosDeviceLifecycle(SimctlClient(runner), timeout_seconds=45).prepare(_ios_device())
    lease.close()
    lease.close()

    assert lease.ownership is ResourceOwnership.RUN
    assert lease.descriptor.readiness is DeviceReadiness.READY
    assert lease.descriptor.native_state == "Booted"
    assert runner.calls == [
        (("xcrun", "simctl", "boot", "SIMULATOR-UDID"), 30),
        (("xcrun", "simctl", "bootstatus", "SIMULATOR-UDID", "-b"), 45),
        (("xcrun", "simctl", "shutdown", "SIMULATOR-UDID"), 30),
    ]


@pytest.mark.parametrize(
    "device",
    [
        _ios_device(readiness=DeviceReadiness.READY, native_state="Booted"),
        _ios_device(
            kind=DeviceKind.PHYSICAL,
            readiness=DeviceReadiness.READY,
            native_state="connected",
        ),
    ],
)
def test_ready_target_is_external_and_is_not_stopped(device: DeviceDescriptor) -> None:
    runner = CommandRunnerStub()

    lease = IosDeviceLifecycle(SimctlClient(runner)).prepare(device)
    lease.close()

    assert lease.ownership is ResourceOwnership.EXTERNAL
    assert runner.calls == []


def test_transitioning_simulator_is_waited_for_but_remains_external() -> None:
    runner = CommandRunnerStub()

    lease = IosDeviceLifecycle(SimctlClient(runner)).prepare(
        _ios_device(
            readiness=DeviceReadiness.TRANSITIONING,
            native_state="Booting",
        )
    )
    lease.close()

    assert lease.ownership is ResourceOwnership.EXTERNAL
    assert runner.calls == [(("xcrun", "simctl", "bootstatus", "SIMULATOR-UDID", "-b"), 120)]


def test_failed_simulator_start_does_not_claim_ownership() -> None:
    runner = CommandRunnerStub(stderr="CoreSimulator failed", returncode=1)

    with pytest.raises(DeviceCommandError, match="start the simulator"):
        IosDeviceLifecycle(SimctlClient(runner)).prepare(_ios_device())

    assert runner.calls == [
        (("xcrun", "simctl", "boot", "SIMULATOR-UDID"), 30),
    ]


def test_failed_simulator_wait_stops_the_started_simulator() -> None:
    runner = CommandRunnerStub(
        responses=(("", "", 0), ("", "CoreSimulator failed", 1), ("", "", 0))
    )

    with pytest.raises(DeviceLifecycleError, match="did not finish startup"):
        IosDeviceLifecycle(SimctlClient(runner)).prepare(_ios_device())

    assert runner.calls == [
        (("xcrun", "simctl", "boot", "SIMULATOR-UDID"), 30),
        (("xcrun", "simctl", "bootstatus", "SIMULATOR-UDID", "-b"), 120),
        (("xcrun", "simctl", "shutdown", "SIMULATOR-UDID"), 30),
    ]
