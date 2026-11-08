"""Tests for owning the lifetime of started child processes."""

from __future__ import annotations

import subprocess
import sys

import pytest

from host.windows_simulation import load_host_module_as
from yewo.host import start_detached_process, stop_process_tree


class _ProcessStub:
    """Record teardown requests without starting a real child."""

    def __init__(self, *, exit_codes: list[int | None] | None = None) -> None:
        self.pid = 4321
        self.terminated = False
        self.killed = False
        self.wait_timeouts: list[float | None] = []
        self._exit_codes = exit_codes or [None]
        self._timeout_once = False

    def poll(self) -> int | None:
        return self._exit_codes[0]

    def terminate(self) -> None:
        self.terminated = True
        self._exit_codes = [0]

    def kill(self) -> None:
        self.killed = True
        self._exit_codes = [0]

    def wait(self, timeout: float | None = None) -> int:
        self.wait_timeouts.append(timeout)
        if self._timeout_once:
            self._timeout_once = False
            raise subprocess.TimeoutExpired("child", timeout or 0)
        return 0

    def refuse_one_wait(self) -> None:
        self._timeout_once = True


def test_an_exited_child_needs_no_teardown() -> None:
    process = _ProcessStub(exit_codes=[0])

    stop_process_tree(process)

    assert not process.terminated
    assert not process.killed


def test_a_running_child_is_asked_to_stop() -> None:
    process = _ProcessStub()

    stop_process_tree(process)

    assert process.terminated
    assert not process.killed


def test_a_child_that_ignores_the_request_is_killed() -> None:
    process = _ProcessStub()
    process.refuse_one_wait()

    stop_process_tree(process)

    assert process.terminated
    assert process.killed


def test_a_started_child_runs_and_can_be_stopped() -> None:
    process = start_detached_process([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert process.poll() is None
    finally:
        stop_process_tree(process)

    assert process.poll() is not None


def test_a_started_child_inherits_only_the_given_environment() -> None:
    process = start_detached_process(
        [sys.executable, "-c", "import os,sys; sys.exit(0 if os.environ.get('YEWO_PROBE')=='1' else 1)"],
        environment={"YEWO_PROBE": "1"},
    )

    assert process.wait(timeout=30) == 0


def test_posix_detachment_starts_a_new_session(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_host_module_as(monkeypatch, "process_control", "linux")
    recorded: dict[str, object] = {}

    def record_popen(command: list[str], **keywords: object) -> _ProcessStub:
        recorded["command"] = command
        recorded.update(keywords)
        return _ProcessStub()

    monkeypatch.setattr(module.subprocess, "Popen", record_popen)

    module.start_detached_process(["adb", "shell", "screenrecord"])

    assert recorded["start_new_session"] is True
    assert "creationflags" not in recorded


def test_windows_detachment_uses_its_own_process_group(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200, raising=False
    )
    monkeypatch.setattr(subprocess, "CREATE_NO_WINDOW", 0x8000000, raising=False)
    module = load_host_module_as(monkeypatch, "process_control", "win32")
    recorded: dict[str, object] = {}

    def record_popen(command: list[str], **keywords: object) -> _ProcessStub:
        recorded["command"] = command
        recorded.update(keywords)
        return _ProcessStub()

    monkeypatch.setattr(module.subprocess, "Popen", record_popen)

    module.start_detached_process(["node", "appium"])

    assert recorded["creationflags"] == 0x200 | 0x8000000
    assert "start_new_session" not in recorded


def test_windows_teardown_ends_the_whole_process_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_host_module_as(monkeypatch, "process_control", "win32")
    commands: list[list[str]] = []

    def record_run(command: list[str], **keywords: object) -> subprocess.CompletedProcess[bytes]:
        del keywords
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", record_run)
    process = _ProcessStub()

    module.stop_process_tree(process)

    assert commands == [["taskkill", "/F", "/T", "/PID", "4321"]]
    assert not process.terminated


def test_windows_teardown_falls_back_when_the_host_tool_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_host_module_as(monkeypatch, "process_control", "win32")

    def refuse_run(command: list[str], **keywords: object) -> subprocess.CompletedProcess[bytes]:
        del keywords
        return subprocess.CompletedProcess(command, 128)

    monkeypatch.setattr(module.subprocess, "run", refuse_run)
    process = _ProcessStub()

    module.stop_process_tree(process)

    assert process.terminated
