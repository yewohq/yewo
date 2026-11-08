"""Tests for run ownership, classification, and cleanup."""

from pathlib import Path

import pytest

from yewo.artifacts import ARTIFACTS_DIRECTORY
from yewo.devices import ResourceOwnership
from yewo.observations import RedactionRegion
from yewo.runs import (
    RunStage,
    RunStart,
    RunStore,
    RunTerminal,
    RunTerminalCategory,
    RunWorkflowOutcome,
)
from yewo.runs.failure_diagnostics import has_partial_evidence
from yewo.runs.errors import (
    RunCoordinationError,
    RunTerminalPersistenceError,
    RunWorkflowEvidenceError,
    RunWorkflowInfrastructureError,
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
        "workflow.prepare",
        "appium.acquire",
        "session.open",
        "workflow.execute",
        "session.close",
        "appium.close",
        "device.close",
    ]
    assert (tmp_path / "runs/run_1/run/terminal.json").is_file()
    session_log_lines = (tmp_path / "runs/run_1/runtime/yewo-session.log").read_text().splitlines()
    assert all(" run-coordinator: " in line for line in session_log_lines)
    assert [line.split(" ", 2)[2] for line in session_log_lines] == [
        f"Run run_1 started. Evidence: {tmp_path / 'runs/run_1'}",
        "Selecting a device...",
        "Device selected: Pixel (android, virtual).",
        "Preparing device emulator-5554...",
        "Device emulator-5554 is ready.",
        "Resolving the application...",
        "Application resolved: com.example.app.",
        "Starting the Appium server...",
        "Appium server started.",
        "Opening the automation session...",
        "Automation session opened.",
        "Application installation change: none.",
        "Starting task execution...",
        "Task execution concluded: completed — The workflow completed.",
        "Closing automation resources...",
        "Automation resources closed.",
        "terminal: completed at finished — The workflow completed.",
    ]
    trace = (tmp_path / "runs/run_1/runtime/trace.log").read_text()
    assert "run=run_1 mode=normal stage=startup" in trace
    assert "device=emulator-5554 connection=emulator-5554 readiness=ready" in trace
    assert "appium_log=" in trace
    assert "cleanup=automation-session state=completed" in trace
    assert "cleanup=appium-server state=completed" in trace
    assert "cleanup=device state=completed" in trace
    assert "terminal_artifact=run/terminal.json" in trace


def test_long_workflow_reason_is_bounded_without_losing_its_terminal(tmp_path: Path) -> None:
    events: list[str] = []
    workflow = WorkflowStub(
        events,
        outcome=RunWorkflowOutcome(RunTerminalCategory.INCOMPLETE, "reason " * 300),
    )

    terminal = build_coordinator(events, workflow).run(run_request(tmp_path))

    assert terminal.category is RunTerminalCategory.INCOMPLETE
    assert terminal.stage is RunStage.EXECUTION
    assert len(terminal.reason) == 1024
    assert terminal.reason.endswith("...")
    assert (tmp_path / "runs/run_1/run/terminal.json").is_file()


def test_session_policy_is_applied_before_workflow_execution(tmp_path: Path) -> None:
    events: list[str] = []
    region = RedactionRegion(left=1, top=2, right=3, bottom=4)
    workflow = WorkflowStub(events, recording_redaction_regions=(region,))

    build_coordinator(events, workflow).run(run_request(tmp_path))

    assert events.index("session.redaction:1") < events.index("session.open")
    assert events.index("workflow.prepare") < events.index("session.open")
    assert events.index("session.open") < events.index("workflow.execute")


def test_selection_failure_is_a_setup_block_without_opening_resources(tmp_path: Path) -> None:
    events: list[str] = []

    terminal = build_coordinator(events, WorkflowStub(events)).run(
        run_request(tmp_path, target_id="missing")
    )

    assert terminal.category is RunTerminalCategory.BLOCKED_BY_SETUP
    assert terminal.stage is RunStage.DEVICE_SELECTION
    assert events == ["inventory"]
    trace_log = (tmp_path / "runs/run_1/runtime/trace.log").read_text()
    assert "DeviceNotFoundError during device_selection" in trace_log
    assert "target_id=missing" not in trace_log


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
    trace_log = (tmp_path / "runs/run_1/runtime/trace.log").read_text()
    assert "RuntimeError during execution" in trace_log
    assert "provider failed" not in trace_log


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
    trace_log = (tmp_path / "runs/run_1/runtime/trace.log").read_text()
    assert "RuntimeError during cleanup" in trace_log
    assert "cleanup failed" not in trace_log


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


def test_failed_partial_evidence_is_infrastructure_with_interruption_precedence(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    workflow = WorkflowStub(
        events,
        failure=RunWorkflowEvidenceError(
            interrupted=True,
            original_error=RunWorkflowInterrupted(),
        ),
    )

    terminal = build_coordinator(events, workflow).run(run_request(tmp_path))

    assert terminal.category is RunTerminalCategory.INFRASTRUCTURE_FAILURE
    assert terminal.stage is RunStage.EXECUTION
    assert terminal.preceding_category is RunTerminalCategory.INTERRUPTED
    assert terminal.reason == "Yewo could not persist partial workflow evidence."
    trace_log = (tmp_path / "runs/run_1/runtime/trace.log").read_text()
    assert "RunWorkflowInterrupted with evidence persistence failure" in trace_log


def test_workflow_infrastructure_reason_reaches_the_terminal(tmp_path: Path) -> None:
    events: list[str] = []
    workflow = WorkflowStub(
        events,
        failure=RunWorkflowInfrastructureError("Yewo could not restore the original device state."),
    )

    terminal = build_coordinator(events, workflow).run(run_request(tmp_path))

    assert terminal.category is RunTerminalCategory.INFRASTRUCTURE_FAILURE
    assert terminal.reason == "Yewo could not restore the original device state."


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


def test_a_ledger_lock_alone_is_not_partial_evidence(tmp_path: Path) -> None:
    artifacts = tmp_path / ARTIFACTS_DIRECTORY
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "observations.jsonl.lock").write_bytes(b"")

    assert not has_partial_evidence(tmp_path)

    (artifacts / "observations.jsonl").write_bytes(b'{"record_id":"1"}\n')

    assert has_partial_evidence(tmp_path)
