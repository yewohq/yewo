"""Tests for durable run-coordination records."""

from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError
import pytest

from yewo.applications import ApplicationInstallMode, ApplicationRequest
from yewo.devices import DevicePlatform, DeviceQuery
from yewo.runs import (
    RunMode,
    RunRequest,
    RunStage,
    RunTerminal,
    RunTerminalCategory,
)


def _request(tmp_path: Path) -> RunRequest:
    return RunRequest(
        mode=RunMode.NORMAL,
        output_root=tmp_path / "runs",
        device=DeviceQuery(platform=DevicePlatform.ANDROID, target_id="emulator-5554"),
        application=ApplicationRequest(
            platform=DevicePlatform.ANDROID,
            install_mode=ApplicationInstallMode.REUSE,
            application_id="com.example.app",
        ),
    )


def test_request_rejects_mismatched_platforms(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="platforms must match"):
        RunRequest(
            mode=RunMode.NORMAL,
            output_root=tmp_path / "runs",
            device=DeviceQuery(platform=DevicePlatform.IOS),
            application=_request(tmp_path).application,
        )


def test_completed_terminal_requires_the_finished_stage(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="finish every coordinator stage"):
        RunTerminal(
            run_id="run_1",
            category=RunTerminalCategory.COMPLETED,
            stage=RunStage.EXECUTION,
            reason="The workflow completed.",
            completed_at=datetime(2026, 10, 30, tzinfo=timezone.utc),
            partial_evidence_available=True,
        )


def test_noncompleted_terminal_identifies_the_affected_stage(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="affected stage"):
        RunTerminal(
            run_id="run_1",
            category=RunTerminalCategory.INFRASTRUCTURE_FAILURE,
            stage=RunStage.FINISHED,
            reason="The device inventory failed.",
            completed_at=datetime(2026, 10, 30, tzinfo=timezone.utc),
            partial_evidence_available=False,
        )
