"""Android emulator host-environment isolation."""

from collections.abc import Sequence
import subprocess

import pytest

from yewo.devices.android import MacOsAndroidEmulatorEnvironment
from yewo.devices.errors import DeviceLifecycleError


class _DefaultsRunner:
    def __init__(self, value: str | None, *, apply_writes: bool = True) -> None:
        self.value = value
        self.apply_writes = apply_writes
        self.commands: list[tuple[str, ...]] = []

    def __call__(
        self,
        command: Sequence[str],
        *,
        timeout_seconds: float,
    ) -> subprocess.CompletedProcess[str]:
        del timeout_seconds
        arguments = tuple(command)
        self.commands.append(arguments)
        operation = arguments[1]
        if operation == "read":
            if self.value is None:
                return subprocess.CompletedProcess(command, 1, "", "missing")
            return subprocess.CompletedProcess(command, 0, f"{self.value}\n", "")
        if operation == "write":
            if self.apply_writes:
                self.value = "1" if arguments[-1] == "true" else "0"
        elif operation == "delete":
            self.value = None
        return subprocess.CompletedProcess(command, 0, "", "")


def test_clipboard_sharing_is_disabled_verified_and_restored() -> None:
    runner = _DefaultsRunner("1")

    restore = MacOsAndroidEmulatorEnvironment(runner).isolate_clipboard()

    assert runner.value == "0"
    restore()
    assert runner.value == "1"


def test_absent_clipboard_preference_is_removed_after_restore() -> None:
    runner = _DefaultsRunner(None)

    restore = MacOsAndroidEmulatorEnvironment(runner).isolate_clipboard()
    restore()

    assert runner.value is None


def test_failed_isolation_verification_restores_the_original_preference() -> None:
    runner = _DefaultsRunner("1", apply_writes=False)

    with pytest.raises(DeviceLifecycleError, match="Could not verify"):
        MacOsAndroidEmulatorEnvironment(runner).isolate_clipboard()

    assert runner.value == "1"
