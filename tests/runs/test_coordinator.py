"""Tests for run ownership, classification, and cleanup."""

from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast

import pytest

from yewo.applications import (
    ApplicationDescriptor,
    ApplicationInstallMode,
    ApplicationRequest,
)
from yewo.automation import (
    AppiumServerConfig,
    AppiumServerError,
    AppiumServerLease,
    AutomationSession,
)
from yewo.devices import (
    DeviceDescriptor,
    DeviceKind,
    DeviceLease,
    DevicePlatform,
    DeviceQuery,
    DeviceReadiness,
    ResourceOwnership,
)
from yewo.devices.logs import DeviceLogRedactionResult
from yewo.devices.errors import DeviceLifecycleError
from yewo.observations import RedactionRegion
from yewo.runs import (
    RunCoordinator,
    RunExecutionContext,
    RunMode,
    RunRequest,
    RunSessionPolicy,
    RunStage,
    RunStart,
    RunTerminal,
    RunTerminalCategory,
    RunWorkflowOutcome,
    RunWorkspaceFactory,
    RunStore,
)
from yewo.runs.errors import (
    RunCoordinationError,
    RunTerminalPersistenceError,
    RunWorkflowInterrupted,
)


STARTED_AT = datetime(2026, 10, 30, 12, tzinfo=timezone.utc)
DEVICE = DeviceDescriptor(
    platform=DevicePlatform.ANDROID,
    target_id="emulator-5554",
    connection_id="emulator-5554",
    display_name="Pixel",
    kind=DeviceKind.VIRTUAL,
    readiness=DeviceReadiness.READY,
    native_state="device",
)
APPLICATION = ApplicationDescriptor(
    platform=DevicePlatform.ANDROID,
    application_id="com.example.app",
)


class _InventoryStub:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def list_devices(self) -> Sequence[DeviceDescriptor]:
        self.events.append("inventory")
        return (DEVICE,)


class _LifecycleStub:
    def __init__(self, events: list[str], failure: Exception | None = None) -> None:
        self.events = events
        self.failure = failure

    def prepare(self, device: DeviceDescriptor) -> DeviceLease:
        assert device is DEVICE
        self.events.append("device.prepare")
        if self.failure is not None:
            raise self.failure
        return DeviceLease(
            device,
            ResourceOwnership.RUN,
            lambda: self.events.append("device.close"),
        )


class _ApplicationResolverStub:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def resolve(self, request: ApplicationRequest) -> ApplicationDescriptor:
        del request
        self.events.append("application.resolve")
        return APPLICATION


class _ServerManagerStub:
    def __init__(self, events: list[str], failure: Exception | None = None) -> None:
        self.events = events
        self.failure = failure

    def acquire(self, config: AppiumServerConfig) -> AppiumServerLease:
        assert config.log_path is not None
        self.events.append("appium.acquire")
        if self.failure is not None:
            raise self.failure
        return AppiumServerLease(
            config.server_url,
            ResourceOwnership.RUN,
            release=lambda: self.events.append("appium.close"),
        )


class _SessionStub:
    def __init__(
        self,
        events: list[str],
        device: DeviceLease,
        server: AppiumServerLease,
        *,
        fail_close: bool = False,
        interrupt_close: bool = False,
    ) -> None:
        self.events = events
        self.device = device
        self.server = server
        self.fail_close = fail_close
        self.interrupt_close = interrupt_close

    def close(self) -> None:
        self.events.append("session.close")
        self.server.close()
        self.device.close()
        if self.interrupt_close:
            raise KeyboardInterrupt
        if self.fail_close:
            raise RuntimeError("cleanup failed")


class _SessionManagerStub:
    def __init__(
        self,
        events: list[str],
        *,
        fail_close: bool = False,
        interrupt_close: bool = False,
    ) -> None:
        self.events = events
        self.fail_close = fail_close
        self.interrupt_close = interrupt_close
        self.recording_redaction_regions: tuple[RedactionRegion, ...] = ()

    def open(
        self,
        device,
        server,
        request,
        application,
        *,
        artifact_root: Path,
        scrub_device_log_text,
        recording_redaction_regions,
    ) -> AutomationSession:
        del request, application, scrub_device_log_text
        self.recording_redaction_regions = tuple(recording_redaction_regions)
        if self.recording_redaction_regions:
            self.events.append(f"session.redaction:{len(self.recording_redaction_regions)}")
        assert artifact_root.name == "run_1"
        self.events.append("session.open")
        return cast(
            AutomationSession,
            _SessionStub(
                self.events,
                device,
                server,
                fail_close=self.fail_close,
                interrupt_close=self.interrupt_close,
            ),
        )


class _WorkflowStub:
    def __init__(
        self,
        events: list[str],
        *,
        outcome: RunWorkflowOutcome | None = None,
        failure: Exception | None = None,
        write_evidence: bool = False,
        recording_redaction_regions: tuple[RedactionRegion, ...] = (),
    ) -> None:
        self.events = events
        self.outcome = outcome or RunWorkflowOutcome(
            RunTerminalCategory.COMPLETED,
            "The device-foundation workflow completed.",
        )
        self.failure = failure
        self.write_evidence = write_evidence
        self.recording_redaction_regions = recording_redaction_regions

    def session_policy(self) -> RunSessionPolicy:
        return RunSessionPolicy(
            recording_redaction_regions=self.recording_redaction_regions,
        )

    def execute(self, context: RunExecutionContext, interruption) -> RunWorkflowOutcome:
        assert not interruption.is_set()
        self.events.append("workflow.execute")
        if self.write_evidence:
            evidence = context.workspace.root / "assets/evidence.txt"
            evidence.parent.mkdir(parents=True)
            evidence.write_text("partial")
        if self.failure is not None:
            raise self.failure
        return self.outcome


class _Signal:
    def __init__(self, interrupted: bool) -> None:
        self.interrupted = interrupted

    def is_set(self) -> bool:
        return self.interrupted


def _request(tmp_path: Path, *, target_id: str = "emulator-5554") -> RunRequest:
    return RunRequest(
        mode=RunMode.DEVICE_FOUNDATION,
        output_root=tmp_path / "runs",
        device=DeviceQuery(platform=DevicePlatform.ANDROID, target_id=target_id),
        application=ApplicationRequest(
            platform=DevicePlatform.ANDROID,
            install_mode=ApplicationInstallMode.REUSE,
            application_id="com.example.app",
        ),
    )


def _coordinator(
    tmp_path: Path,
    events: list[str],
    workflow: _WorkflowStub,
    *,
    fail_close: bool = False,
    interrupt_close: bool = False,
    lifecycle_failure: Exception | None = None,
    server_failure: Exception | None = None,
    store_factory: Callable[[Path], RunStore] = RunStore,
) -> RunCoordinator:
    del tmp_path
    return RunCoordinator(
        RunWorkspaceFactory(
            identifier_factory=lambda: "run_1",
            time_factory=lambda: STARTED_AT,
        ),
        {DevicePlatform.ANDROID: _InventoryStub(events)},
        {DevicePlatform.ANDROID: _LifecycleStub(events, lifecycle_failure)},
        _ApplicationResolverStub(events),
        _ServerManagerStub(events, server_failure),
        _SessionManagerStub(
            events,
            fail_close=fail_close,
            interrupt_close=interrupt_close,
        ),
        {RunMode.DEVICE_FOUNDATION: workflow},
        lambda workspace: AppiumServerConfig(log_path=workspace.root / "runtime/appium.log"),
        lambda text: DeviceLogRedactionResult(text, "test-policy", 0),
        store_factory=store_factory,
        time_factory=lambda: STARTED_AT + timedelta(seconds=1),
    )


def test_success_runs_only_the_selected_workflow_and_closes_owned_resources(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    workflow = _WorkflowStub(events)

    terminal = _coordinator(tmp_path, events, workflow).run(_request(tmp_path))

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
    workflow = _WorkflowStub(events, recording_redaction_regions=(region,))
    coordinator = _coordinator(tmp_path, events, workflow)

    coordinator.run(_request(tmp_path))

    assert "session.redaction:1" in events


def test_selection_failure_is_a_setup_block_without_opening_resources(tmp_path: Path) -> None:
    events: list[str] = []
    workflow = _WorkflowStub(events)

    terminal = _coordinator(tmp_path, events, workflow).run(_request(tmp_path, target_id="missing"))

    assert terminal.category is RunTerminalCategory.BLOCKED_BY_SETUP
    assert terminal.stage is RunStage.DEVICE_SELECTION
    assert events == ["inventory"]


def test_device_lifecycle_failure_is_a_setup_block(tmp_path: Path) -> None:
    events: list[str] = []
    coordinator = _coordinator(
        tmp_path,
        events,
        _WorkflowStub(events),
        lifecycle_failure=DeviceLifecycleError("device did not become ready"),
    )

    terminal = coordinator.run(_request(tmp_path))

    assert terminal.category is RunTerminalCategory.BLOCKED_BY_SETUP
    assert terminal.stage is RunStage.DEVICE_PREPARATION


def test_appium_server_failure_is_a_setup_block(tmp_path: Path) -> None:
    events: list[str] = []
    coordinator = _coordinator(
        tmp_path,
        events,
        _WorkflowStub(events),
        server_failure=AppiumServerError("server unavailable"),
    )

    terminal = coordinator.run(_request(tmp_path))

    assert terminal.category is RunTerminalCategory.BLOCKED_BY_SETUP
    assert terminal.stage is RunStage.APPIUM_STARTUP
    assert events[-1] == "device.close"


def test_component_exception_is_infrastructure_failure_and_preserves_evidence(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    workflow = _WorkflowStub(
        events,
        failure=RuntimeError("provider failed"),
        write_evidence=True,
    )

    terminal = _coordinator(tmp_path, events, workflow).run(_request(tmp_path))

    assert terminal.category is RunTerminalCategory.INFRASTRUCTURE_FAILURE
    assert terminal.stage is RunStage.EXECUTION
    assert terminal.partial_evidence_available
    assert events[-3:] == ["session.close", "appium.close", "device.close"]


def test_interruption_before_selection_creates_a_terminal_without_resources(
    tmp_path: Path,
) -> None:
    events: list[str] = []

    terminal = _coordinator(tmp_path, events, _WorkflowStub(events)).run(
        _request(tmp_path),
        interruption=_Signal(True),
    )

    assert terminal.category is RunTerminalCategory.INTERRUPTED
    assert terminal.stage is RunStage.DEVICE_SELECTION
    assert events == []


def test_cleanup_failure_overrides_success_without_erasing_it(tmp_path: Path) -> None:
    events: list[str] = []

    terminal = _coordinator(
        tmp_path,
        events,
        _WorkflowStub(events),
        fail_close=True,
    ).run(_request(tmp_path))

    assert terminal.category is RunTerminalCategory.INFRASTRUCTURE_FAILURE
    assert terminal.stage is RunStage.CLEANUP
    assert terminal.preceding_category is RunTerminalCategory.COMPLETED


def test_cleanup_interruption_preserves_the_completed_outcome(tmp_path: Path) -> None:
    events: list[str] = []

    terminal = _coordinator(
        tmp_path,
        events,
        _WorkflowStub(events),
        interrupt_close=True,
    ).run(_request(tmp_path))

    assert terminal.category is RunTerminalCategory.INTERRUPTED
    assert terminal.stage is RunStage.CLEANUP
    assert terminal.preceding_category is RunTerminalCategory.COMPLETED


def test_workflow_interruption_is_not_misclassified_as_infrastructure(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    workflow = _WorkflowStub(events, failure=RunWorkflowInterrupted())

    terminal = _coordinator(tmp_path, events, workflow).run(_request(tmp_path))

    assert terminal.category is RunTerminalCategory.INTERRUPTED
    assert terminal.stage is RunStage.EXECUTION


def test_terminal_write_failure_preserves_the_computed_terminal(tmp_path: Path) -> None:
    class FailingTerminalStore(RunStore):
        def persist_terminal(self, start: RunStart, terminal: RunTerminal) -> Path:
            del start, terminal
            raise RunCoordinationError("terminal write failed")

    events: list[str] = []
    coordinator = _coordinator(
        tmp_path,
        events,
        _WorkflowStub(events),
        store_factory=FailingTerminalStore,
    )

    with pytest.raises(RunTerminalPersistenceError) as captured:
        coordinator.run(_request(tmp_path))

    assert captured.value.terminal.category is RunTerminalCategory.COMPLETED
