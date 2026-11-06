"""Tests for Setup command orchestration."""

from argparse import Namespace

import pytest

from yewo.cli.setup import run
from yewo.setup import SetupReport, SetupStatus, SetupStep


def test_dry_run_does_not_run_doctor(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    report = SetupReport(
        platforms=("android",),
        steps=(
            SetupStep(
                name="Appium",
                status=SetupStatus.PLANNED,
                detail="would install",
                command=("npm", "install", "--global", "appium"),
            ),
        ),
    )
    monkeypatch.setattr("yewo.cli.setup.run_setup", lambda context: report)
    monkeypatch.setattr(
        "yewo.cli.setup.doctor_command.run",
        lambda arguments, start=None: pytest.fail("Dry-run must not invoke Doctor."),
    )

    exit_code = run(
        Namespace(platform="android", dry_run=True, assume_yes=False),
        start=tmp_path,
    )

    assert exit_code == 0
    assert "would run yewo doctor" in capsys.readouterr().out
