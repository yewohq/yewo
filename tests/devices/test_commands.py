"""Tests for the shared subprocess boundary's narration behavior."""

from pathlib import Path

import pytest

from yewo.devices._commands import execute_command
from yewo.devices.errors import DeviceCommandError
from yewo.run_narration import RunNarrator, narration_scope

from ._runner import CommandRunnerStub


def test_a_call_outside_narration_scope_narrates_nothing(tmp_path: Path) -> None:
    runner = CommandRunnerStub(stdout="ok")

    execute_command(runner, ["adb", "devices"], purpose="list Android devices")

    assert not (tmp_path / "runtime/yewo-session.log").exists()
    assert not (tmp_path / "runtime/trace.log").exists()


def test_a_successful_call_is_traced_without_session_narration(tmp_path: Path) -> None:
    runner = CommandRunnerStub(stdout="ok")
    narrator = RunNarrator(tmp_path)

    with narration_scope(narrator):
        execute_command(
            runner,
            ["adb", "shell", "am", "start", "-d", "example://open?token=secret-value"],
            purpose="open a deep link",
        )

    trace_log = (tmp_path / "runtime/trace.log").read_text()
    assert "device-command: adb: 5 arguments omitted" in trace_log
    assert not (tmp_path / "runtime/yewo-session.log").exists()
    assert "secret-value" not in trace_log
    assert "example://" not in trace_log


def test_a_failing_call_is_traced_without_session_narration(tmp_path: Path) -> None:
    runner = CommandRunnerStub(stdout="", stderr="no device", returncode=1)
    narrator = RunNarrator(tmp_path)

    with narration_scope(narrator), pytest.raises(DeviceCommandError):
        execute_command(runner, ["adb", "devices"], purpose="list Android devices")

    trace_log = (tmp_path / "runtime/trace.log").read_text()
    assert "device-command: adb: 1 arguments omitted" in trace_log
    assert not (tmp_path / "runtime/yewo-session.log").exists()
