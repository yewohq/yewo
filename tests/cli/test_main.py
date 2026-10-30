"""Tests for safe activation of the deterministic run command."""

import json
from datetime import datetime, timezone
from pathlib import Path
import signal

import pytest

from yewo.cli.main import main
from yewo.devices import DevicePlatform
from yewo.runs import RunStage, RunTerminal, RunTerminalCategory
from yewo.runs.errors import RunTerminalPersistenceError


def _write_script(path: Path, *, platform: str = "android") -> None:
    path.write_text(
        json.dumps(
            {
                "script_id": "script_1",
                "platform": platform,
                "actions": [{"action_id": "wait_1", "kind": "wait", "duration_seconds": 0}],
            }
        )
    )


def _arguments(tmp_path: Path, script: Path) -> list[str]:
    return [
        "run",
        "--output-root",
        str(tmp_path / "runs"),
        "--script",
        str(script),
        "--platform",
        "android",
        "--device-id",
        "emulator-5554",
        "--application-id",
        "com.example.app",
    ]


def test_run_command_loads_the_script_and_reports_the_terminal(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script_path = tmp_path / "script.json"
    _write_script(script_path)
    captured: list[tuple[Path | None, DevicePlatform]] = []

    def execute(request, script, interruption) -> RunTerminal:
        assert not interruption.is_set()
        captured.append((request.workflow_input_path, script.platform))
        return _terminal()

    exit_code = main(_arguments(tmp_path, script_path), executor=execute)

    assert exit_code == 0
    assert captured == [(script_path, DevicePlatform.ANDROID)]
    assert capsys.readouterr().out == "run_1 completed\n"


def test_run_command_rejects_a_script_for_another_platform(tmp_path: Path) -> None:
    script_path = tmp_path / "script.json"
    _write_script(script_path, platform="ios")

    def execute(request, script, interruption) -> RunTerminal:
        del request, script, interruption
        raise AssertionError("The executor must not receive an invalid script.")

    with pytest.raises(SystemExit) as captured:
        main(_arguments(tmp_path, script_path), executor=execute)

    assert captured.value.code == 2


def test_sigterm_is_forwarded_as_cooperative_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = tmp_path / "script.json"
    _write_script(script_path)
    installed_handler = None

    def install_handler(signal_number, handler):
        nonlocal installed_handler
        assert signal_number == signal.SIGTERM
        if callable(handler):
            installed_handler = handler
        return signal.SIG_DFL

    monkeypatch.setattr(signal, "signal", install_handler)

    def execute(request, script, interruption) -> RunTerminal:
        del request, script
        assert installed_handler is not None
        installed_handler(signal.SIGTERM, None)
        assert interruption.is_set()
        return _terminal(category=RunTerminalCategory.INTERRUPTED)

    exit_code = main(_arguments(tmp_path, script_path), executor=execute)

    assert exit_code == 130


def test_terminal_persistence_failure_reports_the_computed_outcome(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script_path = tmp_path / "script.json"
    _write_script(script_path)

    def execute(request, script, interruption) -> RunTerminal:
        del request, script, interruption
        raise RunTerminalPersistenceError(_terminal())

    exit_code = main(_arguments(tmp_path, script_path), executor=execute)

    output = capsys.readouterr()
    assert exit_code == 1
    assert output.out == "run_1 completed\n"
    assert "could not persist" in output.err


def _terminal(
    *,
    category: RunTerminalCategory = RunTerminalCategory.COMPLETED,
) -> RunTerminal:
    return RunTerminal(
        run_id="run_1",
        category=category,
        stage=(
            RunStage.FINISHED if category is RunTerminalCategory.COMPLETED else RunStage.EXECUTION
        ),
        reason="The deterministic script reached its terminal.",
        completed_at=datetime(2026, 10, 30, tzinfo=timezone.utc),
        partial_evidence_available=True,
    )
