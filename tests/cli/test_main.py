"""Tests for the product command-line entry point."""

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from yewo.agent import NormalRunConfiguration
from yewo.cli.main import main


def test_cli_requires_a_command(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as captured:
        main([])

    assert captured.value.code == 2
    output = capsys.readouterr()
    assert "{run}" in output.err


def test_cli_dispatches_an_explicit_normal_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = cast(
        NormalRunConfiguration,
        SimpleNamespace(model_profiles=object()),
    )
    selected: list[Path] = []
    monkeypatch.setattr(
        "yewo.cli.main.load_normal_run_configuration",
        lambda path: selected.append(path) or configuration,
    )
    monkeypatch.setattr("yewo.cli.main.validate_model_credentials", lambda profiles: None)
    monkeypatch.setattr("yewo.cli.main._run_normal", lambda value: 0)

    assert main(["run", "normal-run.json"]) == 0
    assert selected == [Path("normal-run.json")]
