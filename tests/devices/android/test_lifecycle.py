"""Tests for Android device startup and ownership."""

from collections.abc import Sequence
import subprocess

import pytest

from yewo.devices.android import AndroidDeviceLifecycle
from yewo.devices.errors import DeviceLifecycleError, DeviceStartupTimeoutError
from yewo.devices.inventory import (
    DeviceDescriptor,
    DeviceKind,
    DevicePlatform,
    DeviceReadiness,
)
from yewo.resources import ResourceOwnership


def _android_device(
    *,
    readiness: DeviceReadiness,
    connection_id: str | None = None,
    native_state: str = "stopped",
) -> DeviceDescriptor:
    return DeviceDescriptor(
        platform=DevicePlatform.ANDROID,
        target_id="Pixel_9",
        connection_id=connection_id,
        display_name="Pixel 9",
        kind=DeviceKind.VIRTUAL,
        readiness=readiness,
        native_state=native_state,
    )


class _InventoryStub:
    def __init__(self, devices: Sequence[DeviceDescriptor]) -> None:
        self._devices = tuple(devices)
        self.calls = 0

    def list_devices(self) -> tuple[DeviceDescriptor, ...]:
        self.calls += 1
        return self._devices


class _AdbStub:
    def __init__(
        self,
        *,
        booted: bool = True,
        services_ready: bool = True,
        executable: str | None = "emulator",
        keyboard_setting: bool | None = False,
        handwriting_setting: bool | None = True,
    ) -> None:
        self.booted = booted
        self.services_ready = services_ready
        self.executable = executable
        self.keyboard_setting = keyboard_setting
        self.handwriting_setting = handwriting_setting
        self.boot_checks: list[str] = []
        self.service_checks: list[str] = []
        self.killed: list[str] = []
        self.keyboard_changes: list[tuple[str, bool | None]] = []
        self.handwriting_changes: list[tuple[str, bool | None]] = []

    def emulator_executable(self) -> str | None:
        return self.executable

    def boot_completed(self, connection_id: str) -> bool:
        self.boot_checks.append(connection_id)
        return self.booted

    def system_services_ready(self, connection_id: str) -> bool:
        self.service_checks.append(connection_id)
        return self.services_ready

    def read_show_ime_with_hard_keyboard(self, connection_id: str) -> bool | None:
        del connection_id
        return self.keyboard_setting

    def set_show_ime_with_hard_keyboard(self, connection_id: str, enabled: bool | None) -> None:
        self.keyboard_changes.append((connection_id, enabled))

    def read_stylus_handwriting_enabled(self, connection_id: str) -> bool | None:
        del connection_id
        return self.handwriting_setting

    def set_stylus_handwriting_enabled(self, connection_id: str, enabled: bool | None) -> None:
        self.handwriting_changes.append((connection_id, enabled))

    def kill_emulator(self, connection_id: str) -> None:
        self.killed.append(connection_id)


class _ProcessStub:
    def __init__(self, *, returncode: int | None = None) -> None:
        self.returncode = returncode
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return self.returncode or 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


class _ProcessStarterStub:
    def __init__(self, process: _ProcessStub) -> None:
        self.process = process
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command: Sequence[str]) -> _ProcessStub:
        self.commands.append(tuple(command))
        return self.process


class _ClockStub:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def test_running_avd_is_external_and_is_not_stopped() -> None:
    device = _android_device(
        readiness=DeviceReadiness.READY,
        connection_id="emulator-5554",
        native_state="device",
    )
    adb = _AdbStub()

    lease = AndroidDeviceLifecycle(_InventoryStub(()), adb).prepare(device)
    lease.close()

    assert lease.ownership is ResourceOwnership.EXTERNAL
    assert lease.descriptor is device
    assert adb.killed == []
    assert adb.keyboard_changes == []
    assert adb.handwriting_changes == []


def test_stopped_avd_is_started_waited_for_and_stopped_once() -> None:
    stopped = _android_device(readiness=DeviceReadiness.STARTABLE)
    ready = _android_device(
        readiness=DeviceReadiness.READY,
        connection_id="emulator-5554",
        native_state="device",
    )
    adb = _AdbStub()
    process = _ProcessStub()
    starter = _ProcessStarterStub(process)
    lifecycle = AndroidDeviceLifecycle(
        _InventoryStub((ready,)),
        adb,
        headless=True,
        process_starter=starter,
        clock=_ClockStub(),
    )

    lease = lifecycle.prepare(stopped)
    lease.close()
    lease.close()

    assert starter.commands == [("emulator", "-avd", "Pixel_9", "-no-window")]
    assert lease.ownership is ResourceOwnership.RUN
    assert lease.descriptor == ready
    assert adb.keyboard_changes == [
        ("emulator-5554", True),
        ("emulator-5554", False),
    ]
    assert adb.handwriting_changes == [
        ("emulator-5554", False),
        ("emulator-5554", True),
    ]
    assert adb.killed == ["emulator-5554"]


def test_boot_completed_waits_for_required_android_services() -> None:
    stopped = _android_device(readiness=DeviceReadiness.STARTABLE)
    ready = _android_device(
        readiness=DeviceReadiness.READY,
        connection_id="emulator-5554",
        native_state="device",
    )
    adb = _AdbStub(services_ready=False)
    process = _ProcessStub()

    with pytest.raises(DeviceStartupTimeoutError):
        AndroidDeviceLifecycle(
            _InventoryStub((ready,)),
            adb,
            timeout_seconds=1,
            poll_interval_seconds=1,
            process_starter=_ProcessStarterStub(process),
            clock=_ClockStub(),
        ).prepare(stopped)

    assert adb.boot_checks == ["emulator-5554"]
    assert adb.service_checks == ["emulator-5554"]
    assert process.terminated


def test_owned_avd_restores_an_absent_soft_keyboard_setting() -> None:
    stopped = _android_device(readiness=DeviceReadiness.STARTABLE)
    ready = _android_device(
        readiness=DeviceReadiness.READY,
        connection_id="emulator-5554",
        native_state="device",
    )
    adb = _AdbStub(keyboard_setting=None, handwriting_setting=None)
    lease = AndroidDeviceLifecycle(
        _InventoryStub((ready,)),
        adb,
        process_starter=_ProcessStarterStub(_ProcessStub()),
        clock=_ClockStub(),
    ).prepare(stopped)

    lease.close()

    assert adb.keyboard_changes == [
        ("emulator-5554", True),
        ("emulator-5554", None),
    ]
    assert adb.handwriting_changes == [
        ("emulator-5554", False),
        ("emulator-5554", None),
    ]


def test_avd_startup_timeout_stops_the_owned_process() -> None:
    stopped = _android_device(readiness=DeviceReadiness.STARTABLE)
    process = _ProcessStub()

    with pytest.raises(DeviceStartupTimeoutError, match="did not start within 2 seconds"):
        AndroidDeviceLifecycle(
            _InventoryStub((stopped,)),
            _AdbStub(),
            timeout_seconds=2,
            poll_interval_seconds=1,
            process_starter=_ProcessStarterStub(process),
            clock=_ClockStub(),
        ).prepare(stopped)

    assert process.terminated


def test_avd_process_exit_is_reported() -> None:
    stopped = _android_device(readiness=DeviceReadiness.STARTABLE)
    process = _ProcessStub(returncode=1)

    with pytest.raises(DeviceLifecycleError, match="stopped during startup"):
        AndroidDeviceLifecycle(
            _InventoryStub(()),
            _AdbStub(),
            process_starter=_ProcessStarterStub(process),
            clock=_ClockStub(),
        ).prepare(stopped)

    assert not process.terminated


def test_process_stop_escalates_after_terminate_timeout() -> None:
    class StubbornProcess(_ProcessStub):
        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout: float | None = None) -> int:
            if not self.killed:
                raise subprocess.TimeoutExpired("emulator", timeout or 0)
            return -9

    stopped = _android_device(readiness=DeviceReadiness.STARTABLE)
    process = StubbornProcess()

    with pytest.raises(DeviceStartupTimeoutError):
        AndroidDeviceLifecycle(
            _InventoryStub((stopped,)),
            _AdbStub(),
            timeout_seconds=1,
            poll_interval_seconds=1,
            process_starter=_ProcessStarterStub(process),
            clock=_ClockStub(),
        ).prepare(stopped)

    assert process.terminated
    assert process.killed
