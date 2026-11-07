"""Tests for the product command-line entry point."""

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from yewo.agent import YewoRunConfiguration
from yewo.cli.main import main
from yewo.test_config import YewoTestConfiguration


def test_cli_requires_a_command(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as captured:
        main([])

    assert captured.value.code == 2
    output = capsys.readouterr()
    assert "{run,test,doctor,setup}" in output.err


def test_cli_dispatches_an_explicit_yewo_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = cast(
        YewoRunConfiguration,
        SimpleNamespace(model_profiles=object()),
    )
    selected: list[str] = []
    monkeypatch.setattr(
        "yewo.cli.main.run_command.configuration",
        lambda arguments: selected.append(arguments.goal) or configuration,
    )
    monkeypatch.setattr("yewo.cli.main.validate_model_credentials", lambda profiles: None)
    monkeypatch.setattr("yewo.cli.main._run_yewo", lambda value: 0)
    monkeypatch.setattr("yewo.cli.main.record_new_installation", lambda: None)

    assert main(["run", "Open the notifications screen."]) == 0
    assert selected == ["Open the notifications screen."]


def test_cli_dispatches_selected_yewo_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = cast(
        YewoTestConfiguration,
        SimpleNamespace(model_profiles=object()),
    )
    selected: list[tuple[str | None, str | None]] = []
    monkeypatch.setattr(
        "yewo.cli.main.test_command.configurations",
        lambda arguments: (
            selected.append((arguments.workflow, arguments.target)) or (configuration,)
        ),
    )
    monkeypatch.setattr("yewo.cli.main.validate_model_credentials", lambda profiles: None)
    monkeypatch.setattr("yewo.cli.main._run_yewo_tests", lambda values: 0)
    monkeypatch.setattr("yewo.cli.main.record_new_installation", lambda: None)

    assert main(["test", "--workflow", "suite", "--target", "pixel"]) == 0
    assert selected == [("suite", "pixel")]


def test_cli_returns_130_when_a_run_is_interrupted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = cast(
        YewoRunConfiguration,
        SimpleNamespace(model_profiles=object()),
    )
    monkeypatch.setattr("yewo.cli.main.run_command.configuration", lambda arguments: configuration)
    monkeypatch.setattr("yewo.cli.main.validate_model_credentials", lambda profiles: None)
    monkeypatch.setattr(
        "yewo.cli.main._run_yewo",
        lambda value: (_ for _ in ()).throw(KeyboardInterrupt),
    )
    monkeypatch.setattr("yewo.cli.main.record_new_installation", lambda: None)

    assert main(["run", "Open settings."]) == 130


def test_cli_reports_a_missing_yaml_project_as_a_usage_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("yewo.cli.main.record_new_installation", lambda: None)

    with pytest.raises(SystemExit) as captured:
        main(["run", "Open settings."])

    assert captured.value.code == 2
    assert "No yewo.yaml was found" in capsys.readouterr().err


def test_cli_dispatches_doctor_without_loading_a_run_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("yewo.cli.main.record_new_installation", lambda: None)
    monkeypatch.setattr("yewo.cli.main.doctor_command.run", lambda arguments: 1)
    monkeypatch.setattr(
        "yewo.cli.main.run_command.configuration",
        lambda arguments: pytest.fail("Doctor must not compile a run."),
    )

    assert main(["doctor", "--platform", "android"]) == 1


def test_cli_dispatches_setup_without_loading_a_run_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("yewo.cli.main.record_new_installation", lambda: None)
    monkeypatch.setattr("yewo.cli.main.setup_command.run", lambda arguments: 0)
    monkeypatch.setattr(
        "yewo.cli.main.run_command.configuration",
        lambda arguments: pytest.fail("Setup must not compile a run."),
    )

    assert main(["setup", "--dry-run"]) == 0
