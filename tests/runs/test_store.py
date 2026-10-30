"""Tests for isolated run-directory ownership."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from yewo.applications import ApplicationInstallMode, ApplicationRequest
from yewo.devices import DevicePlatform, DeviceQuery
from yewo.runs import (
    RunRequest,
    RunStage,
    RunStore,
    RunTerminal,
    RunTerminalCategory,
    RunWorkspaceFactory,
)
from yewo.runs.errors import RunCoordinationError


STARTED_AT = datetime(2026, 10, 30, 12, tzinfo=timezone.utc)


def _request(tmp_path: Path) -> RunRequest:
    return RunRequest(
        output_root=tmp_path / "runs",
        device=DeviceQuery(platform=DevicePlatform.ANDROID, target_id="emulator-5554"),
        application=ApplicationRequest(
            platform=DevicePlatform.ANDROID,
            install_mode=ApplicationInstallMode.REUSE,
            application_id="com.example.app",
        ),
    )


def test_factory_creates_an_isolated_run_and_persists_its_start(tmp_path: Path) -> None:
    workspace = RunWorkspaceFactory(
        identifier_factory=lambda: "run_1",
        time_factory=lambda: STARTED_AT,
    ).create(_request(tmp_path))

    start_path = workspace.root / "run/start.json"
    payload = json.loads(start_path.read_text())
    assert workspace.root == tmp_path / "runs/run_1"
    assert payload["run_id"] == "run_1"
    assert payload["request"]["device"]["target_id"] == "emulator-5554"


def test_factory_never_reuses_an_existing_run_directory(tmp_path: Path) -> None:
    request = _request(tmp_path)
    factory = RunWorkspaceFactory(
        identifier_factory=lambda: "run_1",
        time_factory=lambda: STARTED_AT,
    )
    factory.create(request)

    with pytest.raises(RunCoordinationError, match="already exists"):
        factory.create(request)


def test_store_persists_exactly_one_matching_terminal(tmp_path: Path) -> None:
    workspace = RunWorkspaceFactory(
        identifier_factory=lambda: "run_1",
        time_factory=lambda: STARTED_AT,
    ).create(_request(tmp_path))
    terminal = RunTerminal(
        run_id="run_1",
        category=RunTerminalCategory.COMPLETED,
        stage=RunStage.FINISHED,
        reason="The device-foundation workflow completed.",
        completed_at=STARTED_AT + timedelta(seconds=2),
        partial_evidence_available=True,
    )
    store = RunStore(workspace.root)

    store.persist_terminal(workspace.start, terminal)

    assert json.loads((workspace.root / "run/terminal.json").read_text())["category"] == "completed"
    with pytest.raises(RunCoordinationError, match="Could not persist"):
        store.persist_terminal(workspace.start, terminal)


def test_store_rejects_a_terminal_for_another_run(tmp_path: Path) -> None:
    workspace = RunWorkspaceFactory(
        identifier_factory=lambda: "run_1",
        time_factory=lambda: STARTED_AT,
    ).create(_request(tmp_path))
    terminal = RunTerminal(
        run_id="run_2",
        category=RunTerminalCategory.INTERRUPTED,
        stage=RunStage.STARTUP,
        reason="The run was interrupted.",
        completed_at=STARTED_AT,
        partial_evidence_available=False,
    )

    with pytest.raises(RunCoordinationError, match="must match"):
        RunStore(workspace.root).persist_terminal(workspace.start, terminal)
