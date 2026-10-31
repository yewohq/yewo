"""Own one invocation without absorbing component responsibilities."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from yewo.applications import ApplicationInspectionError, ApplicationResolver
from yewo.automation import (
    AppiumServerError,
    AppiumServerConfig,
    AppiumServerLease,
    AutomationSession,
)
from yewo.devices import (
    DeviceInventory,
    DeviceLease,
    DeviceLifecycle,
    DevicePlatform,
    DeviceQuery,
    select_device,
)
from yewo.devices.errors import DeviceLifecycleError, DeviceSelectionError
from yewo.devices.logs import DeviceLogTextScrubber
from yewo.run_narration import narration_scope
from yewo.observations import RedactionRegion
from yewo.runs.errors import (
    RunCoordinationError,
    RunTerminalPersistenceError,
    RunWorkflowEvidenceError,
    RunWorkflowInterrupted,
)
from yewo.runs.records import (
    RunMode,
    RunRequest,
    RunStage,
    RunTerminal,
    RunTerminalCategory,
)
from yewo.runs.store import RunStore, RunWorkspace, RunWorkspaceFactory


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class InterruptionSignal(Protocol):
    """Expose cooperative cancellation without owning the signal source."""

    def is_set(self) -> bool:
        """Return whether the caller requested interruption."""
        ...


class NeverInterrupted:
    """Default signal for callers that do not provide cancellation."""

    @staticmethod
    def is_set() -> bool:
        """Return false for the lifetime of the invocation."""
        return False


@dataclass(frozen=True)
class RunWorkflowOutcome:
    """A product workflow conclusion that still requires coordinator cleanup."""

    category: RunTerminalCategory
    reason: str

    def __post_init__(self) -> None:
        allowed = {
            RunTerminalCategory.COMPLETED,
            RunTerminalCategory.INCOMPLETE,
            RunTerminalCategory.BUDGET_EXHAUSTED,
            RunTerminalCategory.BLOCKED_BY_POLICY,
            RunTerminalCategory.APPLICATION_FAILURE,
        }
        if self.category not in allowed:
            raise ValueError("A workflow returned a coordinator-owned terminal category.")
        if not self.reason.strip():
            raise ValueError("A workflow outcome must explain its conclusion.")


@dataclass(frozen=True)
class RunExecutionContext:
    """Owned resources exposed to exactly one selected product workflow."""

    workspace: RunWorkspace
    session: AutomationSession


@dataclass(frozen=True)
class RunSessionPolicy:
    """Workflow requirements that must exist before an automation session starts."""

    recording_redaction_regions: tuple[RedactionRegion, ...] = ()


class RunWorkflow(Protocol):
    """Execute one selected product workflow through an owned session."""

    def session_policy(self) -> RunSessionPolicy:
        """Return safety requirements applied before the driver-owned work starts."""
        ...

    def execute(
        self,
        context: RunExecutionContext,
        interruption: InterruptionSignal,
    ) -> RunWorkflowOutcome:
        """Return a typed conclusion without closing coordinator resources."""
        ...


class _AppiumServerManager(Protocol):
    def acquire(self, config: AppiumServerConfig) -> AppiumServerLease: ...


class _AutomationSessionManager(Protocol):
    def open(
        self,
        device: DeviceLease,
        server: AppiumServerLease,
        request,
        application,
        *,
        artifact_root: Path,
        scrub_device_log_text: DeviceLogTextScrubber,
        recording_redaction_regions: Sequence[RedactionRegion],
    ) -> AutomationSession: ...


ServerConfigFactory = Callable[[RunWorkspace], AppiumServerConfig]


_NARRATION_SOURCE = "run-coordinator"


class RunCoordinator:
    """Select components, own their leases, and persist one truthful terminal."""

    def __init__(
        self,
        workspace_factory: RunWorkspaceFactory,
        inventories: Mapping[DevicePlatform, DeviceInventory],
        lifecycles: Mapping[DevicePlatform, DeviceLifecycle],
        application_resolver: ApplicationResolver,
        appium_servers: _AppiumServerManager,
        automation_sessions: _AutomationSessionManager,
        workflows: Mapping[RunMode, RunWorkflow],
        server_config_factory: ServerConfigFactory,
        scrub_device_log_text: DeviceLogTextScrubber,
        *,
        store_factory: Callable[[Path], RunStore] = RunStore,
        time_factory: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._workspace_factory = workspace_factory
        self._inventories = inventories
        self._lifecycles = lifecycles
        self._application_resolver = application_resolver
        self._appium_servers = appium_servers
        self._automation_sessions = automation_sessions
        self._workflows = workflows
        self._server_config_factory = server_config_factory
        self._scrub_device_log_text = scrub_device_log_text
        self._store_factory = store_factory
        self._time_factory = time_factory

    def run(
        self,
        request: RunRequest,
        *,
        interruption: InterruptionSignal | None = None,
    ) -> RunTerminal:
        """Run one invocation and return its durably persisted terminal."""
        signal = interruption or NeverInterrupted()
        workspace = self._workspace_factory.create(request)
        store = self._store_factory(workspace.root)
        narrator = workspace.narrator
        stage = RunStage.STARTUP
        category: RunTerminalCategory | None = None
        reason = ""
        device: DeviceLease | None = None
        server: AppiumServerLease | None = None
        session: AutomationSession | None = None
        workflow_preceding_category: RunTerminalCategory | None = None
        with narration_scope(narrator):
            try:
                self._require_continuation(signal)
                workflow = self._workflows.get(request.mode)
                if workflow is None:
                    raise _SetupBlock("No workflow is configured for the requested run mode.")
                stage = RunStage.DEVICE_SELECTION
                self._require_continuation(signal)
                inventory = self._inventories.get(request.application.platform)
                lifecycle = self._lifecycles.get(request.application.platform)
                if inventory is None or lifecycle is None:
                    raise _SetupBlock(
                        "No device adapters are configured for the requested platform."
                    )
                query = _platform_query(request.device, request.application.platform)
                narrator.record(_NARRATION_SOURCE, "Selecting a device...")
                selected_device = select_device(query, inventory.list_devices())
                narrator.record(
                    _NARRATION_SOURCE,
                    f"Device selected: {selected_device.target_id} ({selected_device.kind.value}).",
                )
                stage = RunStage.DEVICE_PREPARATION
                self._require_continuation(signal)
                narrator.record(
                    _NARRATION_SOURCE,
                    f"Preparing device {selected_device.target_id}...",
                )
                device = lifecycle.prepare(selected_device)
                narrator.record(
                    _NARRATION_SOURCE,
                    f"Device {device.descriptor.target_id} is ready.",
                )
                stage = RunStage.APPLICATION_INSPECTION
                self._require_continuation(signal)
                narrator.record(_NARRATION_SOURCE, "Resolving the application...")
                application = self._application_resolver.resolve(request.application)
                version = f" v{application.version}" if application.version else ""
                narrator.record(
                    _NARRATION_SOURCE,
                    f"Application resolved: {application.application_id}{version}.",
                )
                stage = RunStage.APPIUM_STARTUP
                self._require_continuation(signal)
                narrator.record(_NARRATION_SOURCE, "Starting the Appium server...")
                server = self._appium_servers.acquire(self._server_config_factory(workspace))
                narrator.record(_NARRATION_SOURCE, "Appium server started.")
                stage = RunStage.AUTOMATION_STARTUP
                self._require_continuation(signal)
                narrator.record(_NARRATION_SOURCE, "Opening the automation session...")
                session = self._automation_sessions.open(
                    device,
                    server,
                    request.application,
                    application,
                    artifact_root=workspace.root,
                    scrub_device_log_text=self._scrub_device_log_text,
                    recording_redaction_regions=(
                        workflow.session_policy().recording_redaction_regions
                    ),
                )
                narrator.record(_NARRATION_SOURCE, "Automation session opened.")
                stage = RunStage.EXECUTION
                self._require_continuation(signal)
                outcome = workflow.execute(
                    RunExecutionContext(workspace=workspace, session=session),
                    signal,
                )
                category = outcome.category
                reason = outcome.reason
            except (KeyboardInterrupt, _RunInterrupted, RunWorkflowInterrupted):
                category = RunTerminalCategory.INTERRUPTED
                reason = "The run was interrupted."
            except RunWorkflowEvidenceError as error:
                category = RunTerminalCategory.INFRASTRUCTURE_FAILURE
                reason = "Yewo could not persist partial workflow evidence."
                if error.interrupted:
                    workflow_preceding_category = RunTerminalCategory.INTERRUPTED
                narrator.trace(
                    _NARRATION_SOURCE,
                    f"{type(error.original_error).__name__} with evidence persistence failure "
                    f"during {stage.value}",
                )
            except (
                _SetupBlock,
                DeviceSelectionError,
                DeviceLifecycleError,
                ApplicationInspectionError,
                AppiumServerError,
            ) as error:
                category = RunTerminalCategory.BLOCKED_BY_SETUP
                reason = "A required device, application, or runtime prerequisite was unavailable."
                narrator.trace(
                    _NARRATION_SOURCE,
                    f"{type(error).__name__} during {stage.value}",
                )
            except Exception as error:  # noqa: BLE001 - coordinator must classify component failures
                category = RunTerminalCategory.INFRASTRUCTURE_FAILURE
                reason = f"Yewo infrastructure failed during {stage.value}."
                narrator.trace(
                    _NARRATION_SOURCE,
                    f"{type(error).__name__} during {stage.value}",
                )

            preceding_category = workflow_preceding_category
            cleanup_error, cleanup_interrupted = _close_resources(session, server, device)
            cleanup_interrupted = cleanup_interrupted or (
                category is not RunTerminalCategory.INTERRUPTED and signal.is_set()
            )
            if cleanup_error is not None:
                if category is not RunTerminalCategory.INFRASTRUCTURE_FAILURE:
                    preceding_category = category
                category = RunTerminalCategory.INFRASTRUCTURE_FAILURE
                stage = RunStage.CLEANUP
                reason = "Yewo infrastructure failed during cleanup."
                narrator.trace(
                    _NARRATION_SOURCE,
                    f"{type(cleanup_error).__name__} during cleanup",
                )
            elif cleanup_interrupted:
                preceding_category = (
                    category if category is not RunTerminalCategory.INTERRUPTED else None
                )
                category = RunTerminalCategory.INTERRUPTED
                stage = RunStage.CLEANUP
                reason = "The run was interrupted during cleanup."
            elif category is RunTerminalCategory.COMPLETED:
                stage = RunStage.FINISHED
            if category is None:
                category = RunTerminalCategory.INFRASTRUCTURE_FAILURE
                reason = "Yewo did not produce a run conclusion."
            terminal = self._build_terminal(
                workspace,
                category,
                stage,
                reason,
                preceding_category,
            )
            narrator.record(
                _NARRATION_SOURCE,
                f"terminal: {terminal.category.value} at {terminal.stage.value} — {terminal.reason}",
            )
            try:
                store.persist_terminal(workspace.start, terminal)
            except RunCoordinationError as error:
                raise RunTerminalPersistenceError(terminal) from error
            return terminal

    def _build_terminal(
        self,
        workspace: RunWorkspace,
        category: RunTerminalCategory,
        stage: RunStage,
        reason: str,
        preceding_category: RunTerminalCategory | None,
    ) -> RunTerminal:
        try:
            return RunTerminal(
                run_id=workspace.start.run_id,
                category=category,
                stage=stage,
                reason=reason,
                completed_at=self._time_factory(),
                partial_evidence_available=_has_partial_evidence(workspace.root),
                preceding_category=preceding_category,
            )
        except ValidationError as error:
            raise RunCoordinationError("Could not build a valid run terminal.") from error

    @staticmethod
    def _require_continuation(signal: InterruptionSignal) -> None:
        if signal.is_set():
            raise _RunInterrupted


class _RunInterrupted(Exception):
    pass


class _SetupBlock(Exception):
    pass


def _platform_query(query: DeviceQuery, platform: DevicePlatform) -> DeviceQuery:
    return query if query.platform is not None else query.model_copy(update={"platform": platform})


def _close_resources(
    session: AutomationSession | None,
    server: AppiumServerLease | None,
    device: DeviceLease | None,
) -> tuple[Exception | None, bool]:
    errors: list[Exception] = []
    interrupted = False
    resources = (session, server, device)
    for resource in resources:
        if resource is None:
            continue
        try:
            resource.close()
        except KeyboardInterrupt:
            interrupted = True
            try:
                resource.close()
            except KeyboardInterrupt:
                interrupted = True
            except Exception as error:  # noqa: BLE001 - cleanup must finish
                errors.append(error)
        except Exception as error:  # noqa: BLE001 - cleanup must finish
            errors.append(error)
    return (errors[0] if errors else None), interrupted


def _has_partial_evidence(root: Path) -> bool:
    assets = root / "assets"
    try:
        return assets.is_dir() and any(path.is_file() for path in assets.rglob("*"))
    except OSError:
        return False
