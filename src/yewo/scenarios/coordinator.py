"""Coordinate one generated scenario around the shared Device Agent."""

from __future__ import annotations

from asyncio import CancelledError
from collections.abc import Callable
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, Self

from pydantic import BaseModel, ConfigDict, model_validator

from yewo.agent.replay.procedure.records import (
    ProcedureReplayResult,
    ProcedureReplayTerminal,
)
from yewo.agent.runtime.records import AgentResult, AgentTask, AgentTerminalReason
from yewo.artifacts import AtomicArtifactWriter, ArtifactWriteError, RecordIdentifier
from yewo.devices.capabilities import (
    CapabilityApplicationError,
    CapabilityPreparation,
    CapabilityRestorationError,
    CapabilitySupportResult,
    CapabilitySupportState,
    CapabilityTransactionReceipt,
)
from yewo.devices.application_preconditions import (
    ApplicationPrecondition,
    ApplicationPreconditionReceipt,
)
from yewo.project_config.capabilities import CapabilitySpec, DeviceCapabilitySpec
from yewo.reporting import ScenarioExecutionClassification, ScenarioExecutionRecord
from yewo.runs.errors import RunWorkflowInterrupted
from yewo.scenarios.records import ScenarioDefinition, ScenarioSuite
from yewo.scenarios.paths import scenario_execution_path
from yewo.scenarios.taxonomy import ScenarioExecutionChange
from yewo.scenarios.budget import ScenarioBudgetExhausted


class ScenarioRouteExecutor(Protocol):
    """Execute the discovery-approved route for one scenario."""

    def execute(self, scenario: ScenarioDefinition) -> ProcedureReplayResult: ...


class ScenarioTaskExecutor(Protocol):
    """Execute remaining scenario work through the shared Device Agent."""

    def execute(self, task: AgentTask) -> AgentResult: ...


class ScenarioCapabilityPreparer(Protocol):
    """Prepare a supported and restorable device capability request."""

    def prepare(self, request: DeviceCapabilitySpec) -> CapabilityPreparation[CapabilitySpec]: ...


class ScenarioApplicationPreparer(Protocol):
    """Establish and verify one scenario application start state."""

    def establish(
        self,
        requested_state: ApplicationPrecondition,
    ) -> ApplicationPreconditionReceipt: ...


class ScenarioCoordinationResult(BaseModel):
    """Persistable execution evidence retained before independent audit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: RecordIdentifier
    task: AgentTask | None = None
    execution: ScenarioExecutionRecord
    application_setup: ApplicationPreconditionReceipt | None = None
    route: ProcedureReplayResult | None = None
    agent: AgentResult | None = None
    capability_support: tuple[CapabilitySupportResult, ...] = ()
    capability_receipts: tuple[CapabilityTransactionReceipt, ...] = ()

    @model_validator(mode="after")
    def _validate_execution_evidence(self) -> Self:
        if self.agent is not None and self.task is None:
            raise ValueError("Scenario agent evidence requires a normalized task brief.")
        transaction_ids = tuple(item.transaction_id for item in self.capability_receipts)
        if len(transaction_ids) != len(set(transaction_ids)):
            raise ValueError("Scenario capability receipts must be unique.")
        return self


class ScenarioExecutionStoreError(RuntimeError):
    """A scenario execution result could not be persisted."""


class ScenarioExecutionStore:
    """Persist one immutable result under its isolated scenario directory."""

    def __init__(self, run_root: Path) -> None:
        self._writer = AtomicArtifactWriter(run_root)

    def persist(self, result: ScenarioCoordinationResult) -> Path:
        """Write one result without replacing prior scenario evidence."""
        relative_path = scenario_execution_path(result.scenario_id)
        try:
            self._writer.write(
                relative_path,
                f"{result.model_dump_json(indent=2)}\n".encode(),
            )
        except ArtifactWriteError as error:
            raise ScenarioExecutionStoreError(
                "Could not persist the scenario execution result."
            ) from error
        return relative_path


class ScenarioCoordinator:
    """Surround route replay and one shared-agent task with verified state."""

    def __init__(
        self,
        route_executor: ScenarioRouteExecutor,
        task_executor: ScenarioTaskExecutor,
        application_preparer: ScenarioApplicationPreparer,
        capability_preparer: ScenarioCapabilityPreparer,
        result_store: ScenarioExecutionStore,
        *,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._route_executor = route_executor
        self._task_executor = task_executor
        self._application_preparer = application_preparer
        self._capability_preparer = capability_preparer
        self._result_store = result_store
        self._now = now

    def execute(
        self,
        suite: ScenarioSuite,
        scenario: ScenarioDefinition,
    ) -> ScenarioCoordinationResult:
        """Execute and persist one scenario without making an audit verdict."""
        _validate_scenario(suite, scenario)
        started_at = self._now()
        task = _build_scenario_task(suite, scenario)
        route: ProcedureReplayResult | None = None
        agent: AgentResult | None = None
        application_setup: ApplicationPreconditionReceipt | None = None
        preparations: tuple[CapabilityPreparation[CapabilitySpec], ...] = ()
        restoration_failure_reason: str | None = None
        classification, reason = _execution_readiness_classification(scenario)
        if classification is None:
            try:
                preparations = self._prepare_capabilities(scenario)
            except Exception:  # noqa: BLE001 - device preflight is an injected boundary
                preparations = ()
                classification = ScenarioExecutionClassification.CAPABILITY_FAILURE
                reason = "Yewo could not inspect the required device capabilities."
            else:
                classification, reason = _capability_preflight_classification(
                    tuple(item.support for item in preparations),
                )
        if classification is None:
            try:
                application_setup = self._application_preparer.establish(scenario.setup.app_state)
            except Exception:  # noqa: BLE001 - application setup is an injected boundary
                classification = ScenarioExecutionClassification.INFRASTRUCTURE_FAILURE
                reason = "Yewo could not establish the requested application state."
            else:
                if not application_setup.verified:
                    classification = ScenarioExecutionClassification.INFRASTRUCTURE_FAILURE
                    reason = application_setup.reason
        supports = tuple(item.support for item in preparations)
        transactions = tuple(
            item.transaction for item in preparations if item.transaction is not None
        )
        try:
            if classification is None:
                with ExitStack() as setup_stack:
                    setup_count = len(scenario.setup.capabilities)
                    for transaction in transactions[:setup_count]:
                        setup_stack.enter_context(transaction)
                    route = self._route_executor.execute(scenario)
                    classification, reason = _route_classification(route)
                    if classification is ScenarioExecutionClassification.COMPLETED and (
                        scenario.execution_change is not ScenarioExecutionChange.ACCEPTED_ROUTE
                    ):
                        with ExitStack() as execution_stack:
                            for transaction in transactions[setup_count:]:
                                execution_stack.enter_context(transaction)
                            agent = self._task_executor.execute(task)
                            classification, reason = _agent_classification(agent)
        except (CancelledError, RunWorkflowInterrupted):
            classification = ScenarioExecutionClassification.INTERRUPTED
            reason = "Scenario execution was interrupted."
        except CapabilityApplicationError:
            classification = ScenarioExecutionClassification.CAPABILITY_FAILURE
            reason = "Yewo could not apply a required device capability."
        except CapabilityRestorationError:
            restoration_failure_reason = (
                "Yewo could not restore and verify the original device state."
            )
            if classification is None:
                classification = ScenarioExecutionClassification.CAPABILITY_FAILURE
                reason = restoration_failure_reason
        except ScenarioBudgetExhausted:
            classification = ScenarioExecutionClassification.BUDGET_EXHAUSTED
            reason = "The scenario Device Agent budget was exhausted."
        except Exception as error:  # noqa: BLE001 - coordinator classifies component failures
            classification = ScenarioExecutionClassification.INFRASTRUCTURE_FAILURE
            reason = f"A Yewo component failed during scenario execution: {type(error).__name__}."
        receipts = tuple(
            receipt for transaction in transactions if (receipt := transaction.receipt) is not None
        )
        if any(not item.restoration.verified for item in receipts):
            restoration_failure_reason = (
                "Yewo could not restore and verify the original device state."
            )
        if classification is None or reason is None:
            raise AssertionError("Scenario coordination requires a terminal classification.")
        result = ScenarioCoordinationResult(
            scenario_id=scenario.scenario_id,
            task=(
                task
                if scenario.execution_change is not ScenarioExecutionChange.ACCEPTED_ROUTE
                else None
            ),
            execution=ScenarioExecutionRecord(
                classification=classification,
                reason=reason,
                started_at=started_at,
                completed_at=self._now(),
                action_count=_action_count(route, agent),
                model_call_count=_model_call_count(route, agent),
                evidence_ids=_evidence_ids(application_setup, route, agent, receipts),
                restoration_failure_reason=restoration_failure_reason,
            ),
            application_setup=application_setup,
            route=route,
            agent=agent,
            capability_support=supports,
            capability_receipts=receipts,
        )
        self._result_store.persist(result)
        return result

    def _prepare_capabilities(
        self,
        scenario: ScenarioDefinition,
    ) -> tuple[CapabilityPreparation[CapabilitySpec], ...]:
        requests = (*scenario.setup.capabilities, *scenario.execution_capabilities)
        return tuple(self._capability_preparer.prepare(request) for request in requests)


def _validate_scenario(suite: ScenarioSuite, scenario: ScenarioDefinition) -> None:
    matching = next(
        (item for item in suite.scenarios if item.scenario_id == scenario.scenario_id),
        None,
    )
    if matching != scenario:
        raise ValueError("The scenario does not belong to the supplied generated suite.")


def _build_scenario_task(
    suite: ScenarioSuite,
    scenario: ScenarioDefinition,
) -> AgentTask:
    surface_ids = {
        surface_id
        for obligation in scenario.obligations
        for surface_id in obligation.required_surface_ids
    }
    surface_facts = tuple(
        f"Accepted application fact: {surface.summary}"
        for surface in suite.surfaces
        if surface.surface_id in surface_ids
    )
    setup_fact = f"Verified application setup: {scenario.setup.app_state}."
    capability_facts = tuple(
        f"Yewo already applied and verified the {request.kind.value} device capability."
        for request in scenario.execution_capabilities
    )
    if scenario.execution_change is ScenarioExecutionChange.LIFECYCLE_INTERRUPTION:
        objective = f"Execute the lifecycle scenario variation: {scenario.variation}"
        open_work = (
            scenario.variation,
            "Inspect the current application state and complete only work that remains.",
        )
    else:
        objective = f"Execute the scenario variation: {scenario.variation}"
        open_work = (scenario.variation,)
    return AgentTask(
        objective=objective,
        known_facts=(*surface_facts, setup_fact, *capability_facts),
        open_work=open_work,
        completion_conditions=(
            "The requested variation has executed and its resulting application state is visible.",
        ),
        declared_value_names=scenario.declared_value_names,
        targeted_type_available=(
            scenario.execution_change is ScenarioExecutionChange.INPUT_VARIATION
        ),
    )


def _execution_readiness_classification(
    scenario: ScenarioDefinition,
) -> tuple[ScenarioExecutionClassification | None, str | None]:
    if scenario.execution_change not in {
        ScenarioExecutionChange.ACCEPTED_ROUTE,
        ScenarioExecutionChange.LIFECYCLE_INTERRUPTION,
    }:
        return (
            ScenarioExecutionClassification.INSUFFICIENT_DISCOVERY,
            "Discovery did not identify a concrete replay step for this execution variation.",
        )
    if (
        scenario.execution_change is ScenarioExecutionChange.LIFECYCLE_INTERRUPTION
        and not scenario.execution_capabilities
    ):
        return (
            ScenarioExecutionClassification.INSUFFICIENT_DISCOVERY,
            "The lifecycle scenario does not define a concrete device capability.",
        )
    return None, None


def _capability_preflight_classification(
    supports: tuple[CapabilitySupportResult, ...],
) -> tuple[ScenarioExecutionClassification | None, str | None]:
    unsupported = next(
        (item for item in supports if item.state is CapabilitySupportState.UNSUPPORTED),
        None,
    )
    if unsupported is not None:
        return (
            ScenarioExecutionClassification.UNSUPPORTED_CAPABILITY,
            f"The target does not support the required {unsupported.capability.value} capability.",
        )
    unavailable = next(
        (item for item in supports if item.state is CapabilitySupportState.UNAVAILABLE),
        None,
    )
    if unavailable is not None:
        return (
            ScenarioExecutionClassification.CAPABILITY_FAILURE,
            f"The required {unavailable.capability.value} capability is unavailable.",
        )
    return None, None


def _route_classification(
    result: ProcedureReplayResult,
) -> tuple[ScenarioExecutionClassification, str]:
    classification = {
        ProcedureReplayTerminal.COMPLETED: ScenarioExecutionClassification.COMPLETED,
        ProcedureReplayTerminal.REPLAY_COMPLETE: (
            ScenarioExecutionClassification.APPLICATION_FAILURE
        ),
        ProcedureReplayTerminal.DIVERGED: (ScenarioExecutionClassification.INSUFFICIENT_DISCOVERY),
        ProcedureReplayTerminal.BUDGET_EXHAUSTED: (
            ScenarioExecutionClassification.BUDGET_EXHAUSTED
        ),
        ProcedureReplayTerminal.APPLICATION_FAILURE: (
            ScenarioExecutionClassification.APPLICATION_FAILURE
        ),
        ProcedureReplayTerminal.DEVICE_REFUSED: (ScenarioExecutionClassification.BLOCKED_BY_POLICY),
    }[result.terminal]
    return classification, result.reason


def _agent_classification(
    result: AgentResult,
) -> tuple[ScenarioExecutionClassification, str]:
    classification = {
        AgentTerminalReason.DONE_REQUESTED: ScenarioExecutionClassification.COMPLETED,
        AgentTerminalReason.CANNOT_PROGRESS: (
            ScenarioExecutionClassification.UNSUPPORTED_APPLICATION_BEHAVIOR
        ),
        AgentTerminalReason.BUDGET_EXHAUSTED: (ScenarioExecutionClassification.BUDGET_EXHAUSTED),
        AgentTerminalReason.APPLICATION_EXITED: (ScenarioExecutionClassification.APPLICATION_CRASH),
        AgentTerminalReason.APPLICATION_INACTIVE: (
            ScenarioExecutionClassification.APPLICATION_FAILURE
        ),
        AgentTerminalReason.DEVICE_REFUSED: (ScenarioExecutionClassification.BLOCKED_BY_POLICY),
        AgentTerminalReason.INTERRUPTED: ScenarioExecutionClassification.INTERRUPTED,
        AgentTerminalReason.RUNTIME_FAILURE: (
            ScenarioExecutionClassification.INFRASTRUCTURE_FAILURE
        ),
    }[result.terminal_reason]
    return classification, result.reason


def _action_count(
    route: ProcedureReplayResult | None,
    agent: AgentResult | None,
) -> int:
    return sum(
        item.usage.dispatched_actions
        for item in (route.agent_result if route is not None else None, agent)
        if item is not None
    )


def _model_call_count(
    route: ProcedureReplayResult | None,
    agent: AgentResult | None,
) -> int:
    return sum(
        item.usage.planning_requests + item.usage.grounding_requests
        for item in (route.agent_result if route is not None else None, agent)
        if item is not None
    )


def _evidence_ids(
    application_setup: ApplicationPreconditionReceipt | None,
    route: ProcedureReplayResult | None,
    agent: AgentResult | None,
    receipts: tuple[CapabilityTransactionReceipt, ...],
) -> tuple[RecordIdentifier, ...]:
    identifiers: list[RecordIdentifier] = []
    if application_setup is not None:
        identifiers.append(application_setup.receipt_id)
    if route is not None:
        identifiers.extend(item.observation_id for item in route.milestones)
        identifiers.extend(item.screenshot_artifact_id for item in route.milestones)
    if agent is not None:
        identifiers.append(agent.final_observation_id)
        identifiers.extend(item.step_id for item in agent.steps)
    identifiers.extend(item.transaction_id for item in receipts)
    return tuple(dict.fromkeys(identifiers))
