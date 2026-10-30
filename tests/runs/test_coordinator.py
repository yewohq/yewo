"""Tests for run ownership, classification, and cleanup."""

from pathlib import Path

import pytest

from yewo.devices import ResourceOwnership
from yewo.observations import RedactionRegion
from yewo.runs import (
    RunStage,
    RunStart,
    RunStore,
    RunTerminal,
    RunTerminalCategory,
)
from yewo.runs.errors import (
    RunCoordinationError,
    RunTerminalPersistenceError,
    RunWorkflowInterrupted,
)
from .coordinator_support import (
    SignalStub,
    WorkflowStub,
    build_coordinator,
    run_request,
)


def test_success_runs_only_the_selected_workflow_and_closes_owned_resources(
    tmp_path: Path,
) -> None:
    events: list[str] = []

    terminal = build_coordinator(events, WorkflowStub(events)).run(run_request(tmp_path))

    assert terminal.category is RunTerminalCategory.COMPLETED
    assert terminal.stage is RunStage.FINISHED
    assert events == [
        "inventory",
        "device.prepare",
        "application.resolve",
        "appium.acquire",
        "session.open",
        "workflow.execute",
        "session.close",
        "appium.close",
        "device.close",
    ]
    assert (tmp_path / "runs/run_1/run/terminal.json").is_file()


def test_session_policy_is_applied_before_workflow_execution(tmp_path: Path) -> None:
    events: list[str] = []
    region = RedactionRegion(left=1, top=2, right=3, bottom=4)
    workflow = WorkflowStub(events, recording_redaction_regions=(region,))

    build_coordinator(events, workflow).run(run_request(tmp_path))

    assert events.index("session.redaction:1") < events.index("session.open")
    assert events.index("session.open") < events.index("workflow.execute")


def test_selection_failure_is_a_setup_block_without_opening_resources(tmp_path: Path) -> None:
    events: list[str] = []

    terminal = build_coordinator(events, WorkflowStub(events)).run(
        run_request(tmp_path, target_id="missing")
    )

    assert terminal.category is RunTerminalCategory.BLOCKED_BY_SETUP
    assert terminal.stage is RunStage.DEVICE_SELECTION
    assert events == ["inventory"]


def test_component_exception_is_infrastructure_failure_and_preserves_evidence(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    workflow = WorkflowStub(
        events,
        failure=RuntimeError("provider failed"),
        write_evidence=True,
    )

    terminal = build_coordinator(events, workflow).run(run_request(tmp_path))

    assert terminal.category is RunTerminalCategory.INFRASTRUCTURE_FAILURE
    assert terminal.stage is RunStage.EXECUTION
    assert terminal.partial_evidence_available
    assert events[-3:] == ["session.close", "appium.close", "device.close"]


def test_interruption_before_selection_creates_a_terminal_without_resources(
    tmp_path: Path,
) -> None:
    events: list[str] = []

    terminal = build_coordinator(events, WorkflowStub(events)).run(
        run_request(tmp_path),
        interruption=SignalStub(events, initially_interrupted=True),
    )

    assert terminal.category is RunTerminalCategory.INTERRUPTED
    assert terminal.stage is RunStage.STARTUP
    assert events == []


def test_missing_workflow_is_a_startup_setup_block(tmp_path: Path) -> None:
    events: list[str] = []
    coordinator = build_coordinator(
        events,
        WorkflowStub(events),
        include_workflow=False,
    )

    terminal = coordinator.run(run_request(tmp_path))

    assert terminal.category is RunTerminalCategory.BLOCKED_BY_SETUP
    assert terminal.stage is RunStage.STARTUP
    assert events == []


def test_cleanup_failure_overrides_success_without_erasing_it(tmp_path: Path) -> None:
    events: list[str] = []

    terminal = build_coordinator(
        events,
        WorkflowStub(events),
        fail_close=True,
    ).run(run_request(tmp_path))

    assert terminal.category is RunTerminalCategory.INFRASTRUCTURE_FAILURE
    assert terminal.stage is RunStage.CLEANUP
    assert terminal.preceding_category is RunTerminalCategory.COMPLETED


def test_cleanup_interruption_preserves_the_completed_outcome(tmp_path: Path) -> None:
    events: list[str] = []

    terminal = build_coordinator(
        events,
        WorkflowStub(events),
        interrupt_close=True,
    ).run(run_request(tmp_path))

    assert terminal.category is RunTerminalCategory.INTERRUPTED
    assert terminal.stage is RunStage.CLEANUP
    assert terminal.preceding_category is RunTerminalCategory.COMPLETED


def test_workflow_interruption_is_not_misclassified_as_infrastructure(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    workflow = WorkflowStub(events, failure=RunWorkflowInterrupted())

    terminal = build_coordinator(events, workflow).run(run_request(tmp_path))

    assert terminal.category is RunTerminalCategory.INTERRUPTED
    assert terminal.stage is RunStage.EXECUTION


def test_terminal_write_failure_preserves_the_computed_terminal(tmp_path: Path) -> None:
    class FailingTerminalStore(RunStore):
        def persist_terminal(self, start: RunStart, terminal: RunTerminal) -> Path:
            del start, terminal
            raise RunCoordinationError("terminal write failed")

    events: list[str] = []
    coordinator = build_coordinator(
        events,
        WorkflowStub(events),
        store_factory=FailingTerminalStore,
    )

    with pytest.raises(RunTerminalPersistenceError) as captured:
        coordinator.run(run_request(tmp_path))

    assert captured.value.terminal.category is RunTerminalCategory.COMPLETED


def test_external_resources_are_closed_without_releasing_them(tmp_path: Path) -> None:
    events: list[str] = []
    coordinator = build_coordinator(
        events,
        WorkflowStub(events),
        device_ownership=ResourceOwnership.EXTERNAL,
        server_ownership=ResourceOwnership.EXTERNAL,
    )

    terminal = coordinator.run(run_request(tmp_path))

    assert terminal.category is RunTerminalCategory.COMPLETED
    assert "session.close" in events
    assert "appium.close" not in events
    assert "device.close" not in events
