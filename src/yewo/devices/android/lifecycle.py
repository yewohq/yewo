"""Prepare Android targets and preserve device ownership."""

from __future__ import annotations

from collections.abc import Callable, Sequence
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

    def system_services_ready(self, connection_id: str) -> bool: ...

    def read_show_ime_with_hard_keyboard(self, connection_id: str) -> bool | None: ...

    def set_show_ime_with_hard_keyboard(self, connection_id: str, enabled: bool | None) -> None: ...

    def read_stylus_handwriting_enabled(self, connection_id: str) -> bool | None: ...

    def set_stylus_handwriting_enabled(self, connection_id: str, enabled: bool | None) -> None: ...

    def kill_emulator(self, connection_id: str) -> None: ...


class _SystemClock:
    monotonic = staticmethod(time.monotonic)
    sleep = staticmethod(time.sleep)


class _HostEnvironmentControl(Protocol):
    def isolate_clipboard(self) -> Callable[[], None]: ...


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
        host_environment: _HostEnvironmentControl | None = None,
    ) -> None:
        self._inventory = inventory
        self._adb = adb
        self._timeout_seconds = timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._headless = headless
        self._process_starter = process_starter
        self._clock = clock or _SystemClock()
        self._host_environment = host_environment

    def prepare(self, device: DeviceDescriptor) -> DeviceLease:
        """Return a ready Android target and stop only an AVD started here."""
        self._validate(device)
        restore_host = self._isolate_host_clipboard(device)
        try:
            if device.readiness is DeviceReadiness.STARTABLE:
                return self._start_avd(device, restore_host)
            if device.readiness not in {
                DeviceReadiness.READY,
                DeviceReadiness.TRANSITIONING,
            }:
                raise DeviceLifecycleError(
                    f"Android target '{device.target_id}' is {device.native_state}."
                )
            ready, connection_id = self._wait_until_ready(device)
            restore_device = self._prepare_virtual_environment(ready, connection_id)
            return DeviceLease(
                ready,
                ResourceOwnership.EXTERNAL,
                restore_environment=_combine_restorations(restore_device, restore_host),
            )
        except BaseException:
            if restore_host is not None:
                with suppress(Exception):
                    restore_host()
            raise

    @staticmethod
    def _validate(device: DeviceDescriptor) -> None:
        if device.platform is not DevicePlatform.ANDROID:
            raise DeviceLifecycleError("The Android lifecycle requires an Android target.")
        if device.readiness is DeviceReadiness.STARTABLE and device.kind is not DeviceKind.VIRTUAL:
            raise DeviceLifecycleError("Only a virtual Android target can be started.")

    def _start_avd(
        self,
        device: DeviceDescriptor,
        restore_host: Callable[[], None] | None,
    ) -> DeviceLease:
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
            restore_environment = self._prepare_virtual_environment(ready, connection_id)
        except BaseException:
            with suppress(Exception):
                _stop_process(process)
            raise

        def release() -> None:
            try:
                self._adb.kill_emulator(connection_id)
            except Exception:  # noqa: BLE001 - process fallback must still run
                _stop_process(process)
                raise

        return DeviceLease(
            ready,
            ResourceOwnership.RUN,
            release,
            restore_environment=_combine_restorations(
                restore_environment,
                restore_host,
            ),
        )

    def _isolate_host_clipboard(
        self,
        device: DeviceDescriptor,
    ) -> Callable[[], None] | None:
        if device.kind is not DeviceKind.VIRTUAL or self._host_environment is None:
            return None
        return self._host_environment.isolate_clipboard()

    def _prepare_virtual_environment(
        self,
        device: DeviceDescriptor,
        connection_id: str,
    ) -> Callable[[], None] | None:
        if device.kind is not DeviceKind.VIRTUAL:
            return None
        prior_keyboard_setting = self._adb.read_show_ime_with_hard_keyboard(connection_id)
        prior_handwriting_setting = self._adb.read_stylus_handwriting_enabled(connection_id)
        try:
            self._adb.set_show_ime_with_hard_keyboard(connection_id, True)
            self._adb.set_stylus_handwriting_enabled(connection_id, False)
        except BaseException:
            with suppress(Exception):
                self._adb.set_show_ime_with_hard_keyboard(
                    connection_id,
                    prior_keyboard_setting,
                )
            with suppress(Exception):
                self._adb.set_stylus_handwriting_enabled(
                    connection_id,
                    prior_handwriting_setting,
                )
            raise

        def restore() -> None:
            errors: list[Exception] = []
            for operation in (
                lambda: self._adb.set_show_ime_with_hard_keyboard(
                    connection_id,
                    prior_keyboard_setting,
                ),
                lambda: self._adb.set_stylus_handwriting_enabled(
                    connection_id,
                    prior_handwriting_setting,
                ),
            ):
                try:
                    operation()
                except Exception as error:  # noqa: BLE001 - restoration must continue
                    errors.append(error)
            if errors:
                raise errors[0]

        return restore

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
                and self._adb.system_services_ready(connection_id)
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


def _combine_restorations(
    *operations: Callable[[], None] | None,
) -> Callable[[], None] | None:
    restorations = tuple(operation for operation in operations if operation is not None)
    if not restorations:
        return None

    def restore() -> None:
        errors: list[Exception] = []
        for operation in restorations:
            try:
                operation()
            except Exception as error:  # noqa: BLE001 - restoration must continue
                errors.append(error)
        if errors:
            raise errors[0]

    return restore
