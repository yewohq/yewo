"""Load a host module with its Windows branch bound, for tests on any host."""

from __future__ import annotations

from collections.abc import Sequence
import importlib.util
from pathlib import Path
import sys
import types

import yewo.host


HOST_PACKAGE_ROOT = Path(yewo.host.__file__).parent


class FakeWindowsLocking:
    """Record msvcrt locking calls and fail acquisition on demand."""

    LK_NBLCK = 1
    LK_UNLCK = 0

    def __init__(self, failures_before_success: int = 0) -> None:
        self.calls: list[tuple[int, int]] = []
        self._remaining_failures = failures_before_success

    def locking(self, descriptor: int, mode: int, byte_count: int) -> None:
        del descriptor
        self.calls.append((mode, byte_count))
        if mode == self.LK_NBLCK and self._remaining_failures:
            self._remaining_failures -= 1
            raise PermissionError("the lock is held")

    @property
    def modes(self) -> Sequence[int]:
        return tuple(mode for mode, _ in self.calls)


def load_host_module_as(
    monkeypatch,
    module_name: str,
    platform_name: str,
) -> types.ModuleType:
    """Import one yewo.host module as it binds on the named host."""
    monkeypatch.setattr(sys, "platform", platform_name)
    specification = importlib.util.spec_from_file_location(
        f"{module_name}_as_{platform_name}",
        HOST_PACKAGE_ROOT / f"{module_name}.py",
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def load_host_module_as_windows(
    monkeypatch,
    module_name: str,
    locking: FakeWindowsLocking | None = None,
) -> types.ModuleType:
    """Import one yewo.host module as it binds on Windows."""
    monkeypatch.setitem(sys.modules, "msvcrt", locking or FakeWindowsLocking())
    return load_host_module_as(monkeypatch, module_name, "win32")


class ImmediateClock:
    """Advance time without sleeping so bounded retries stay deterministic."""

    def __init__(self, step_seconds: float = 1.0) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []
        self._step_seconds = step_seconds

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += self._step_seconds
