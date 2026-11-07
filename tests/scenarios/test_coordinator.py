"""Scenario coordination through replay, capabilities, and the shared agent."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from yewo.agent import AgentResult, AgentTerminalReason
from yewo.agent.planning.goal_verifier import GoalVerification
from yewo.agent.replay.procedure.records import (
    MilestoneEvidence,
    ProcedureReplayResult,
    ProcedureReplayTerminal,
)
from yewo.devices.capabilities import (
    CapabilityStateObservation,
    CapabilitySupportResult,
    CapabilitySupportState,
    CapabilityTransactionCoordinator,
)
from yewo.devices.application_preconditions import ApplicationPreconditionReceipt
from yewo.project_config import CapabilityKind, CoverageCategory
from yewo.project_config.capabilities import AppStateCapabilitySpec
from yewo.reporting import ScenarioExecutionClassification
from yewo.scenarios import (
    ScenarioCoordinator,
    ScenarioDefinition,
    ScenarioExecutionChange,
    ScenarioExecutionStore,
    ScenarioSuite,
    evaluate_testing_categories,
    generate_scenarios,
)
from yewo.scenarios.coordinator import _agent_classification
from yewo.scenarios.paths import scenario_execution_path

from agent.discovery.support import agent_result
from agent.replay.support import observations
from scenarios.test_bridge import _bridge, _journey
from scenarios.test_generator import _design


STARTED_AT = datetime(2026, 11, 6, 12, tzinfo=timezone.utc)


def _generated_suite() -> ScenarioSuite:
    check = "The saved item must remain visible after the operation finishes."
    journey = _journey().model_copy(update={"checks": (check,)})
    accepted = _bridge()
    design = _design(accepted, check, category=CoverageCategory.STATE)
    registry = evaluate_testing_categories(journey, accepted, concerns=design.concerns)
    return generate_scenarios(journey, accepted, design, registry).suite


def _replay(
    terminal: ProcedureReplayTerminal = ProcedureReplayTerminal.COMPLETED,
) -> ProcedureReplayResult:
    before, *_middle, after = observations()
    runtime_result = agent_result(before, after)
    satisfied = terminal is ProcedureReplayTerminal.COMPLETED
    verification = (
        GoalVerification(
            verification_id="route_verification",
            objective_id="objective_1",
            final_observation_id=after.observation_id,
            satisfied=satisfied,
            reason="The route audit reached its configured conclusion.",
        )
        if terminal in {ProcedureReplayTerminal.COMPLETED, ProcedureReplayTerminal.REPLAY_COMPLETE}
        else None
    )
    return ProcedureReplayResult(
        procedure_id="procedure_1",
        evidence_id="evidence_1",
        source_run_id="source_run",
        setup_id="setup_1",
        application=before.application,
        terminal=terminal,
        procedure_step_count=1,
        completed_step_count=(
            1
            if terminal
            in {ProcedureReplayTerminal.COMPLETED, ProcedureReplayTerminal.REPLAY_COMPLETE}
            else 0
        ),
        milestones=(
            MilestoneEvidence(
                completion_condition_index=0,
                procedure_index=0,
                source_step=1,
                observation_id=after.observation_id,
                screenshot_artifact_id=after.redacted_screenshot.artifact_id,
                screen_fingerprint=after.screen_fingerprint,
            ),
        )
        if verification is not None
        else (),
        verification=verification,
        reason="The configured route terminal was returned.",
        agent_result=runtime_result,
    )


class _RouteExecutor:
    def __init__(
        self,
        events: list[str],
        replay: ProcedureReplayResult | None = None,
    ) -> None:
        self.events = events
        self.replay = replay or _replay()

    def execute(self, scenario: ScenarioDefinition) -> ProcedureReplayResult:
        self.events.append(f"route:{scenario.scenario_id}")
        return self.replay


class _TaskExecutor:
    def __init__(self, events: list[str], result: AgentResult | None = None) -> None:
        before, *_middle, after = observations()
        self.events = events
        self.result = result or agent_result(before, after)
        self.tasks = []

    def execute(self, task):
        self.events.append("task")
        self.tasks.append(task)
        return self.result


class _CapabilityAdapter:
    def __init__(
        self,
        events: list[str],
        support_state: CapabilitySupportState,
        restoration_fails: bool,
    ) -> None:
        self.events = events
        self.support_state = support_state
        self.restoration_fails = restoration_fails
        self.state = "foreground"

    def support(self, request: AppStateCapabilitySpec) -> CapabilitySupportResult:
        del request
        return CapabilitySupportResult(
            capability=CapabilityKind.APP_STATE,
            state=self.support_state,
            reason="The deterministic target returned its configured support state.",
        )

    def read_state(self, request: AppStateCapabilitySpec) -> CapabilityStateObservation:
        del request
        return CapabilityStateObservation(value=self.state, source="deterministic target")

    def validate_original_state(self, request, original_state):
        del request, original_state
        return self.support(AppStateCapabilitySpec(action="background_resume"))

    def requested_state(self, request: AppStateCapabilitySpec) -> str:
        del request
        return "resumed"

    def apply(self, request: AppStateCapabilitySpec) -> None:
        del request
        self.events.append("capability:apply")
        self.state = "resumed"

    def application_verified(self, request, observed_state) -> bool:
        del request
        return observed_state.value == "resumed"

    def restore(self, request, original_state) -> None:
        del request
        self.events.append("capability:restore")
        if self.restoration_fails:
            raise RuntimeError("private restoration failure")
        self.state = original_state.value

    def restoration_verified(self, request, original_state, observed_state) -> bool:
        del request
        return observed_state == original_state


class _CapabilityPreparer:
    def __init__(
        self,
        events: list[str],
        support_state: CapabilitySupportState = CapabilitySupportState.SUPPORTED,
        restoration_fails: bool = False,
    ) -> None:
        self.events = events
        self.support_state = support_state
        self.restoration_fails = restoration_fails

    def prepare(self, request):
        adapter = _CapabilityAdapter(
            self.events,
            self.support_state,
            self.restoration_fails,
        )
        return CapabilityTransactionCoordinator(
            "target_1",
            adapter,
            identifier_factory=lambda: "capability_transaction",
        ).prepare(request)


class _ApplicationPreparer:
    def __init__(self, events: list[str], *, verified: bool = True) -> None:
        self.events = events
        self.verified = verified

    def establish(self, requested_state):
        normalized = requested_state or "preserve"
        self.events.append(f"application:{normalized}")
        return ApplicationPreconditionReceipt(
            receipt_id="application_setup",
            requested_state=normalized,
            verified=self.verified,
            reason=(
                "The requested application state was verified."
                if self.verified
                else "The requested application state was not verified."
            ),
        )


def _coordinator(
    tmp_path: Path,
    events: list[str],
    *,
    replay: ProcedureReplayResult | None = None,
    support_state: CapabilitySupportState = CapabilitySupportState.SUPPORTED,
    restoration_fails: bool = False,
    application_verified: bool = True,
) -> tuple[ScenarioCoordinator, _TaskExecutor]:
    task_executor = _TaskExecutor(events)
    times = iter((STARTED_AT, STARTED_AT + timedelta(seconds=2)))
    return (
        ScenarioCoordinator(
            _RouteExecutor(events, replay),
            task_executor,
            _ApplicationPreparer(events, verified=application_verified),
            _CapabilityPreparer(events, support_state, restoration_fails),
            ScenarioExecutionStore(tmp_path),
            now=lambda: next(times),
        ),
        task_executor,
    )


def test_accepted_route_completes_without_a_second_agent_loop(tmp_path: Path) -> None:
    suite = _generated_suite()
    scenario = next(
        item
        for item in suite.scenarios
        if item.execution_change is ScenarioExecutionChange.ACCEPTED_ROUTE
    )
    events: list[str] = []
    coordinator, task_executor = _coordinator(tmp_path, events)

    result = coordinator.execute(suite, scenario)

    assert result.execution.classification is ScenarioExecutionClassification.COMPLETED
    assert result.task is None
    assert result.agent is None
    assert task_executor.tasks == []
    assert events == ["application:clear_data", f"route:{scenario.scenario_id}"]
    assert result.route is not None
    assert result.execution.action_count == result.route.agent_result.usage.dispatched_actions
    assert (tmp_path / scenario_execution_path(scenario.scenario_id)).is_file()


def test_lifecycle_capability_runs_after_route_and_restores_after_agent(tmp_path: Path) -> None:
    suite = _generated_suite()
    scenario = next(
        item
        for item in suite.scenarios
        if item.execution_change is ScenarioExecutionChange.LIFECYCLE_INTERRUPTION
    )
    events: list[str] = []
    coordinator, task_executor = _coordinator(tmp_path, events)

    result = coordinator.execute(suite, scenario)

    assert result.execution.classification is ScenarioExecutionClassification.COMPLETED
    assert events == [
        "application:clear_data",
        f"route:{scenario.scenario_id}",
        "capability:apply",
        "task",
        "capability:restore",
    ]
    assert result.capability_receipts[0].restoration.verified
    assert result.task is not None
    assert result.task == task_executor.tasks[0]
    assert result.task.objective == (
        "Execute the lifecycle scenario variation: " + scenario.variation
    )
    assert result.task.open_work[0] == scenario.variation
    assert any("already applied" in fact for fact in result.task.known_facts)
    hidden_outcomes = {
        outcome.statement
        for obligation in scenario.obligations
        for outcome in obligation.expected_outcomes
    }
    task_text = result.task.model_dump_json()
    assert all(statement not in task_text for statement in hidden_outcomes)


def test_unsupported_capability_stops_before_route_mutation(tmp_path: Path) -> None:
    suite = _generated_suite()
    scenario = next(
        item
        for item in suite.scenarios
        if item.execution_change is ScenarioExecutionChange.LIFECYCLE_INTERRUPTION
    )
    events: list[str] = []
    coordinator, _ = _coordinator(
        tmp_path,
        events,
        support_state=CapabilitySupportState.UNSUPPORTED,
    )

    result = coordinator.execute(suite, scenario)

    assert result.execution.classification is ScenarioExecutionClassification.UNSUPPORTED_CAPABILITY
    assert events == []
    assert result.route is None
    assert result.capability_receipts == ()


def test_nonconcrete_variation_is_truthfully_left_unexecuted(tmp_path: Path) -> None:
    generated = _generated_suite()
    baseline = next(
        item
        for item in generated.scenarios
        if item.execution_change is ScenarioExecutionChange.ACCEPTED_ROUTE
    )
    scenario = ScenarioDefinition.model_validate(
        {
            **baseline.model_dump(mode="python"),
            "scenario_id": "input_variation_scenario",
            "execution_change": ScenarioExecutionChange.INPUT_VARIATION,
            "variation": "Change the input at a discovery-selected procedure step.",
        }
    )
    suite = ScenarioSuite.model_validate(
        {**generated.model_dump(mode="python"), "scenarios": (scenario,)}
    )
    events: list[str] = []
    coordinator, _ = _coordinator(tmp_path, events)

    result = coordinator.execute(suite, scenario)

    assert result.execution.classification is ScenarioExecutionClassification.INSUFFICIENT_DISCOVERY
    assert events == []


def test_route_regression_is_application_failure_not_yewo_failure(tmp_path: Path) -> None:
    suite = _generated_suite()
    scenario = next(
        item
        for item in suite.scenarios
        if item.execution_change is ScenarioExecutionChange.ACCEPTED_ROUTE
    )
    events: list[str] = []
    coordinator, _ = _coordinator(
        tmp_path,
        events,
        replay=_replay(ProcedureReplayTerminal.REPLAY_COMPLETE),
    )

    result = coordinator.execute(suite, scenario)

    assert result.execution.classification is ScenarioExecutionClassification.APPLICATION_FAILURE
    assert events == ["application:clear_data", f"route:{scenario.scenario_id}"]


def test_failed_application_setup_stops_before_capability_or_route(tmp_path: Path) -> None:
    suite = _generated_suite()
    scenario = next(
        item
        for item in suite.scenarios
        if item.execution_change is ScenarioExecutionChange.LIFECYCLE_INTERRUPTION
    )
    events: list[str] = []
    coordinator, _ = _coordinator(
        tmp_path,
        events,
        application_verified=False,
    )

    result = coordinator.execute(suite, scenario)

    assert result.execution.classification is ScenarioExecutionClassification.INFRASTRUCTURE_FAILURE
    assert events == ["application:clear_data"]
    assert result.route is None


def test_restoration_failure_is_recorded_without_replacing_execution(tmp_path: Path) -> None:
    suite = _generated_suite()
    scenario = next(
        item
        for item in suite.scenarios
        if item.execution_change is ScenarioExecutionChange.LIFECYCLE_INTERRUPTION
    )
    events: list[str] = []
    coordinator, _ = _coordinator(
        tmp_path,
        events,
        restoration_fails=True,
    )

    result = coordinator.execute(suite, scenario)

    assert result.execution.classification is ScenarioExecutionClassification.COMPLETED
    assert result.execution.restoration_failure_reason is not None
    assert result.capability_receipts[0].restoration.verified is False
    assert events[-1] == "capability:restore"


def test_agent_runtime_failure_preserves_its_reason() -> None:
    before, *_middle, after = observations()
    result = agent_result(before, after).model_copy(
        update={
            "terminal_reason": AgentTerminalReason.RUNTIME_FAILURE,
            "reason": "The model service was unavailable.",
        }
    )

    classification, reason = _agent_classification(result)

    assert classification is ScenarioExecutionClassification.INFRASTRUCTURE_FAILURE
    assert reason == "The model service was unavailable."


def test_agent_returned_interruption_is_classified_without_key_error() -> None:
    before, *_middle, after = observations()
    result = agent_result(before, after).model_copy(
        update={
            "terminal_reason": AgentTerminalReason.INTERRUPTED,
            "reason": "The run was interrupted.",
        }
    )

    classification, reason = _agent_classification(result)

    assert classification is ScenarioExecutionClassification.INTERRUPTED
    assert reason == "The run was interrupted."
