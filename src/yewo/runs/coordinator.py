"""Own one invocation without absorbing component responsibilities."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from yewo.runs.failure_diagnostics import (
    error_chain,
    has_partial_evidence,
    validation_detail,
)
from yewo.applications import (
    ApplicationDescriptor,
    ApplicationInspectionError,
    ApplicationResolver,
)
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
from yewo.run_narration import RunNarrator, narration_scope
from yewo.observations import RedactionRegion
from yewo.runs.errors import (
    RunCoordinationError,
    RunTerminalPersistenceError,
    RunWorkflowEvidenceError,
    RunWorkflowInfrastructureError,
    RunWorkflowInterrupted,
)
from yewo.runs.records import (
    RUN_TERMINAL_REASON_MAX_LENGTH,
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
        allowed_categories = {
            RunTerminalCategory.COMPLETED,
            RunTerminalCategory.INCOMPLETE,
            RunTerminalCategory.BUDGET_EXHAUSTED,
            RunTerminalCategory.BLOCKED_BY_POLICY,
            RunTerminalCategory.APPLICATION_FAILURE,
        }
        if self.category not in allowed_categories:
            raise ValueError("A workflow returned a coordinator-owned terminal category.")
        if not self.reason.strip():
            raise ValueError("A workflow outcome must explain its conclusion.")


@dataclass(frozen=True)
class RunExecutionContext:
    """Owned resources exposed to exactly one selected product workflow."""

    workspace: RunWorkspace
    session: AutomationSession


@dataclass(frozen=True)
class RunPreparationContext:
    """Resolved inputs available before Appium or application startup."""

    workspace: RunWorkspace
    application: ApplicationDescriptor


@dataclass(frozen=True)
class RunSessionPolicy:
    """Workflow requirements that must exist before an automation session starts."""

    recording_redaction_regions: tuple[RedactionRegion, ...] = ()


class RunWorkflow(Protocol):
    """Execute one selected product workflow through an owned session."""

    def session_policy(self) -> RunSessionPolicy:
        """Return safety requirements applied before the driver-owned work starts."""
        ...

    def prepare(
        self,
        context: RunPreparationContext,
        interruption: InterruptionSignal,
    ) -> object | None:
        """Prepare workflow evidence before the automation session starts."""
        ...

    def execute(
        self,
        context: RunExecutionContext,
        interruption: InterruptionSignal,
        preparation: object | None = None,
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
            narrator.record(
                _NARRATION_SOURCE,
                f"Run {workspace.start.run_id} started. Evidence: {workspace.root}",
            )
            narrator.trace(
                _NARRATION_SOURCE,
                f"run={workspace.start.run_id} mode={request.mode.value} stage={stage.value}",
            )
            try:
                self._require_continuation(signal)
                workflow = self._workflows.get(request.mode)
                if workflow is None:
                    raise _SetupBlock("The requested run mode has no workflow.")
                stage = RunStage.DEVICE_SELECTION
                self._require_continuation(signal)
                inventory = self._inventories.get(request.application.platform)
                lifecycle = self._lifecycles.get(request.application.platform)
                if inventory is None or lifecycle is None:
                    raise _SetupBlock(
                        f"Yewo has no {request.application.platform.value} device "
                        "adapters on this host."
                    )
                query = _platform_query(request.device, request.application.platform)
                narrator.record(_NARRATION_SOURCE, "Selecting a device...")
                selected_device = select_device(query, inventory.list_devices())
                narrator.record(
                    _NARRATION_SOURCE,
                    f"Device selected: {selected_device.display_name} "
                    f"({selected_device.platform.value}, {selected_device.kind.value}).",
                )
                narrator.trace(
                    _NARRATION_SOURCE,
                    f"device={selected_device.target_id} connection="
                    f"{selected_device.connection_id or 'none'} readiness="
                    f"{selected_device.readiness.value} os="
                    f"{selected_device.operating_system_version or 'unknown'} model="
                    f"{selected_device.device_model or 'unknown'} screen="
                    f"{selected_device.screen_width_px or 'unknown'}x"
                    f"{selected_device.screen_height_px or 'unknown'}",
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
                narrator.trace(
                    _NARRATION_SOURCE,
                    f"device={device.descriptor.target_id} state=ready os="
                    f"{device.descriptor.operating_system_version or 'unknown'} model="
                    f"{device.descriptor.device_model or 'unknown'} screen="
                    f"{device.descriptor.screen_width_px or 'unknown'}x"
                    f"{device.descriptor.screen_height_px or 'unknown'}",
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
                stage = RunStage.WORKFLOW_PREPARATION
                self._require_continuation(signal)
                preparation = workflow.prepare(
                    RunPreparationContext(workspace=workspace, application=application),
                    signal,
                )
                stage = RunStage.APPIUM_STARTUP
                self._require_continuation(signal)
                narrator.record(_NARRATION_SOURCE, "Starting the Appium server...")
                server_config = self._server_config_factory(workspace)
                narrator.trace(
                    _NARRATION_SOURCE,
                    f"appium_url={server_config.server_url} "
                    f"appium_log={server_config.log_path or 'none'}",
                )
                server = self._appium_servers.acquire(server_config)
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
                narrator.record(
                    _NARRATION_SOURCE,
                    f"Application installation change: {session.installation_change.value}.",
                )
                narrator.trace(
                    _NARRATION_SOURCE,
                    f"installation_ownership={session.installation_ownership.value} "
                    f"retain_installation={str(session.retain_installation).lower()}",
                )
                stage = RunStage.EXECUTION
                self._require_continuation(signal)
                narrator.record(_NARRATION_SOURCE, "Starting task execution...")
                outcome = workflow.execute(
                    RunExecutionContext(workspace=workspace, session=session),
                    signal,
                    preparation,
                )
                category = outcome.category
                reason = outcome.reason
                narrator.record(
                    _NARRATION_SOURCE,
                    f"Task execution concluded: {category.value} — {reason}",
                )
            except (KeyboardInterrupt, _RunInterrupted, RunWorkflowInterrupted):
                category = RunTerminalCategory.INTERRUPTED
                reason = "The caller interrupted the run."
                narrator.record(_NARRATION_SOURCE, "Run interruption requested.")
            except RunWorkflowEvidenceError as error:
                category = RunTerminalCategory.INFRASTRUCTURE_FAILURE
                reason = "Yewo could not persist partial workflow evidence."
                if error.interrupted:
                    workflow_preceding_category = RunTerminalCategory.INTERRUPTED
                narrator.trace(
                    _NARRATION_SOURCE,
                    f"{type(error.original_error).__name__} with evidence persistence failure "
                    f"during {stage.value} chain={error_chain(error.original_error)}",
                )
            except RunWorkflowInfrastructureError as error:
                category = RunTerminalCategory.INFRASTRUCTURE_FAILURE
                reason = error.reason
                narrator.trace(
                    _NARRATION_SOURCE,
                    f"{type(error).__name__} during {stage.value}",
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
                    f"{type(error).__name__} during {stage.value} chain={error_chain(error)}",
                )
            except Exception as error:  # noqa: BLE001 - coordinator must classify component failures
                category = RunTerminalCategory.INFRASTRUCTURE_FAILURE
                reason = f"Yewo infrastructure failed during {stage.value}."
                narrator.trace(
                    _NARRATION_SOURCE,
                    f"{type(error).__name__} during {stage.value} "
                    f"chain={error_chain(error)}{validation_detail(error)}",
                )

            preceding_category = workflow_preceding_category
            narrator.record(_NARRATION_SOURCE, "Closing automation resources...")
            cleanup_error, cleanup_interrupted = _close_resources(session, server, device, narrator)
            if cleanup_error is None and not cleanup_interrupted:
                narrator.record(_NARRATION_SOURCE, "Automation resources closed.")
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
                    f"{type(cleanup_error).__name__} during cleanup "
                    f"chain={error_chain(cleanup_error)}",
                )
            elif cleanup_interrupted:
                preceding_category = (
                    category if category is not RunTerminalCategory.INTERRUPTED else None
                )
                category = RunTerminalCategory.INTERRUPTED
                stage = RunStage.CLEANUP
                reason = "The caller interrupted resource cleanup."
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
                narrator.trace(
                    _NARRATION_SOURCE,
                    "terminal_artifact=run/terminal.json",
                )
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
                reason=_bounded_terminal_reason(reason),
                completed_at=self._time_factory(),
                partial_evidence_available=has_partial_evidence(workspace.root),
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


def _bounded_terminal_reason(reason: str) -> str:
    if len(reason) <= RUN_TERMINAL_REASON_MAX_LENGTH:
        return reason
    content_length = RUN_TERMINAL_REASON_MAX_LENGTH - 3
    return f"{reason[:content_length].rstrip()}..."


def _platform_query(query: DeviceQuery, platform: DevicePlatform) -> DeviceQuery:
    return query if query.platform is not None else query.model_copy(update={"platform": platform})


def _close_resources(
    session: AutomationSession | None,
    server: AppiumServerLease | None,
    device: DeviceLease | None,
    narrator: RunNarrator,
) -> tuple[Exception | None, bool]:
    errors: list[Exception] = []
    interrupted = False
    resources = (
        ("automation-session", session),
        ("appium-server", server),
        ("device", device),
    )
    for resource_name, resource in resources:
        if resource is None:
            continue
        narrator.trace(_NARRATION_SOURCE, f"cleanup={resource_name} state=starting")
        try:
            resource.close()
            narrator.trace(_NARRATION_SOURCE, f"cleanup={resource_name} state=completed")
        except KeyboardInterrupt:
            interrupted = True
            narrator.trace(_NARRATION_SOURCE, f"cleanup={resource_name} state=interrupted")
            try:
                resource.close()
            except KeyboardInterrupt:
                interrupted = True
            except Exception as error:  # noqa: BLE001 - cleanup must finish
                errors.append(error)
                narrator.trace(
                    _NARRATION_SOURCE,
                    f"cleanup={resource_name} state=failed chain={error_chain(error)}",
                )
        except Exception as error:  # noqa: BLE001 - cleanup must finish
            errors.append(error)
            narrator.trace(
                _NARRATION_SOURCE,
                f"cleanup={resource_name} state=failed chain={error_chain(error)}",
            )
    return (errors[0] if errors else None), interrupted
