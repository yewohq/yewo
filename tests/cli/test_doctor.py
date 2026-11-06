"""Tests for Doctor command rendering and exit codes."""

from argparse import Namespace
import json

import pytest

from yewo.cli.doctor import run
from yewo.doctor import DoctorCheck, DoctorReport, DoctorSection, DoctorStatus


def _report(status: DoctorStatus) -> DoctorReport:
    return DoctorReport(
        sections=(
            DoctorSection(
                title="Project",
                checks=(
                    DoctorCheck(
                        name="yewo.yaml",
                        status=status,
                        detail="test detail",
                        hint="test hint" if status is not DoctorStatus.OK else None,
                    ),
                ),
            ),
        )
    )


def test_json_output_contains_stable_summary_counts(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "yewo.cli.doctor.run_doctor", lambda start, platform: _report(DoctorStatus.FAIL)
    )

    exit_code = run(Namespace(platform="android", as_json=True))
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["blocking"] == 1
    assert payload["warnings"] == 0
    assert payload["sections"][0]["checks"][0]["status"] == "fail"


def test_human_report_returns_success_for_warnings(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "yewo.cli.doctor.run_doctor", lambda start, platform: _report(DoctorStatus.WARN)
    )

    exit_code = run(Namespace(platform=None, as_json=False))
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "WARN" in output
    assert "Ready with 1 warning" in output
