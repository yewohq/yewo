"""Prepare Android targets and preserve device ownership."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import suppress
import subprocess
import time
from typing import Protocol

from yewo.devices.errors import DeviceLifecycleError, DeviceStartupTimeoutError
from yewo.devices.inventory import (
    DeviceDescriptor,
    DeviceInventory,
    DeviceKind,
    DevicePlatform,
    DeviceReadiness,
)
from yewo.devices.lifecycle import DeviceLease
from yewo.resources import ResourceOwnership


class _Clock(Protocol):
    def monotonic(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


class _Process(Protocol):
    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def kill(self) -> None: ...


class _ProcessStarter(Protocol):
    def __call__(self, command: Sequence[str]) -> _Process: ...


class _AndroidControl(Protocol):
    def emulator_executable(self) -> str | None: ...

    def boot_completed(self, connection_id: str) -> bool: ...

    def kill_emulator(self, connection_id: str) -> None: ...


class _SystemClock:
    monotonic = staticmethod(time.monotonic)
    sleep = staticmethod(time.sleep)


def _start_process(command: Sequence[str]) -> _Process:
    return subprocess.Popen(  # noqa: S603 - the command is an argument list
        list(command),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _stop_process(process: _Process) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


class AndroidDeviceLifecycle:
    """Prepare Android hardware or an AVD for one run."""

    def __init__(
        self,
        inventory: DeviceInventory,
        adb: _AndroidControl,
        *,
        timeout_seconds: float = 120,
        poll_interval_seconds: float = 1,
        headless: bool = False,
        process_starter: _ProcessStarter = _start_process,
        clock: _Clock | None = None,
    ) -> None:
        self._inventory = inventory
        self._adb = adb
        self._timeout_seconds = timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._headless = headless
        self._process_starter = process_starter
        self._clock = clock or _SystemClock()

    def prepare(self, device: DeviceDescriptor) -> DeviceLease:
        """Return a ready Android target and stop only an AVD started here."""
        self._validate(device)
        if device.readiness is DeviceReadiness.STARTABLE:
            return self._start_avd(device)
        if device.readiness not in {
            DeviceReadiness.READY,
            DeviceReadiness.TRANSITIONING,
        }:
            raise DeviceLifecycleError(
                f"Android target '{device.target_id}' is {device.native_state}."
            )
        ready, _ = self._wait_until_ready(device)
        return DeviceLease(ready, ResourceOwnership.EXTERNAL)

    @staticmethod
    def _validate(device: DeviceDescriptor) -> None:
        if device.platform is not DevicePlatform.ANDROID:
            raise DeviceLifecycleError("The Android lifecycle requires an Android target.")
        if device.readiness is DeviceReadiness.STARTABLE and device.kind is not DeviceKind.VIRTUAL:
            raise DeviceLifecycleError("Only a virtual Android target can be started.")

    def _start_avd(self, device: DeviceDescriptor) -> DeviceLease:
        executable = self._adb.emulator_executable()
        if executable is None:
            raise DeviceLifecycleError("The Android emulator executable is not available.")
        command = [executable, "-avd", device.target_id]
        if self._headless:
            command.append("-no-window")
        try:
            process = self._process_starter(command)
        except OSError as error:
            raise DeviceLifecycleError(
                f"Could not start Android target '{device.target_id}'."
            ) from error
        try:
            ready, connection_id = self._wait_until_ready(device, process=process)
        except BaseException:
            with suppress(Exception):
                _stop_process(process)
            raise

        def release() -> None:
            try:
                self._adb.kill_emulator(connection_id)
            except Exception:
                _stop_process(process)
                raise

        return DeviceLease(ready, ResourceOwnership.RUN, release)

    def _wait_until_ready(
        self, initial: DeviceDescriptor, *, process: _Process | None = None
    ) -> tuple[DeviceDescriptor, str]:
        deadline = self._clock.monotonic() + self._timeout_seconds
        observed = initial
        while True:
            connection_id = observed.connection_id
            if (
                observed.readiness is DeviceReadiness.READY
                and connection_id is not None
                and self._adb.boot_completed(connection_id)
            ):
                return observed, connection_id
            if process is not None and process.poll() is not None:
                raise DeviceLifecycleError(
                    f"Android target '{initial.target_id}' stopped during startup."
                )
            if self._clock.monotonic() >= deadline:
                raise DeviceStartupTimeoutError(
                    f"Android target '{initial.target_id}' did not start within "
                    f"{self._timeout_seconds:g} seconds."
                )
            self._clock.sleep(self._poll_interval_seconds)
            observed = self._find_target(initial.target_id) or observed

    def _find_target(self, target_id: str) -> DeviceDescriptor | None:
        return next(
            (
                candidate
                for candidate in self._inventory.list_devices()
                if candidate.target_id == target_id
            ),
            None,
        )
