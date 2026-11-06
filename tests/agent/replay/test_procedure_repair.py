"""Bounded repair and deterministic compiled-procedure rejoin tests."""

from typing import cast

from yewo.agent import AgentTerminalReason
from yewo.agent.actuation import ActionKind, ActionProposal, LiteralText
from yewo.agent.planning.history import PlanningHistoryEntry
from yewo.agent.planning.evidence import PlannerDecisionMetadata, PlannerDecisionSource
from yewo.agent.planning.goal_verifier import GoalVerification
from yewo.agent.planning.records import (
    PlanningRequest,
    TaskProgressContext,
)
from yewo.agent.replay.procedure.execution import _can_audit_replay, _procedure_terminal
from yewo.agent.replay.procedure.records import (
    ProcedureDivergence,
    ProcedureRepairEvidence,
    ProcedureRepairTarget,
    ProcedureReplayTerminal,
)
from yewo.agent.replay.procedure.repair import BoundedProcedureRepairActionSource
from yewo.agent.replay.procedure.source import CompiledProcedureActionSource
from yewo.agent.replay.reconciliation.compiler import compile_replay_procedure
from yewo.agent.replay.reconciliation.records import ReplayProcedure
from yewo.grounding import GroundingRequest
from yewo.observations import Observation

from agent.discovery.support import agent_result
from agent.replay.support import (
    evidence as _evidence,
    observations as _observations,
    response as _response,
)


class RepairPlanner:
    """Return one harmless device action for each deterministic model decision."""

    def __init__(self) -> None:
        self.request_count = 0
        self.requests: list[PlanningRequest] = []

    def required_model_requests(self, request: PlanningRequest) -> int:
        del request
        return 1

    def plan(self, request: PlanningRequest) -> ActionProposal:
        self.request_count += 1
        self.requests.append(request)
        return ActionProposal(
            proposal_id=f"repair_{self.request_count}",
            source_observation_id=request.observation.observation_id,
            action=ActionKind.WAIT,
            duration_seconds=0.1,
            expected_outcome="The controlled screen remains available.",
            reason="Wait for the controlled repair test.",
        )


class FixedRepairPlanner(RepairPlanner):
    """Return one caller-supplied proposal for a repair policy test."""

    def __init__(self, proposal: ActionProposal) -> None:
        super().__init__()
        self._proposal = proposal

    def plan(self, request: PlanningRequest) -> ActionProposal:
        self.request_count += 1
        self.requests.append(request)
        return self._proposal.model_copy(
            update={"source_observation_id": request.observation.observation_id}
        )


class KeyboardProcedureSource:
    """Expose one keyboard repair target without model-owned decisions."""

    procedure_id = "procedure_keyboard"
    application_id = "com.example.testing_app"
    completed_step_count = 0
    failed_procedure_index = 3
    milestones = ()
    next_procedure_index = 3
    replay_complete = False

    def __init__(self, target: ProcedureRepairTarget, observation: Observation) -> None:
        self.repair_target = target
        self._rejoined = False
        self.divergence = ProcedureDivergence(
            procedure_id=self.procedure_id,
            procedure_index=3,
            source_step=4,
            guard="expectation",
            expected_state_id=target.expected_state_id,
            actual_observation_id=observation.observation_id,
            actual_fingerprint=observation.screen_fingerprint,
            reason="The keyboard closed after focus.",
        )

    def required_model_requests(self, request: PlanningRequest) -> int:
        del request
        return 0

    def plan(self, request: PlanningRequest) -> ActionProposal:
        if self._rejoined:
            return ActionProposal(
                proposal_id="resumed_type",
                source_observation_id=request.observation.observation_id,
                action=ActionKind.TYPE,
                target_expression="the focused Password text field",
                text=LiteralText(value="testpass123"),
                expected_outcome="The password is entered.",
                reason="Resume the compiled procedure.",
            )
        return ActionProposal(
            proposal_id="keyboard_divergence",
            source_observation_id=request.observation.observation_id,
            action=ActionKind.CANNOT_PROGRESS,
            reason="The keyboard state diverged.",
        )

    def decision_metadata(self) -> PlannerDecisionMetadata:
        return PlannerDecisionMetadata(
            source=PlannerDecisionSource.COMPILED_PROCEDURE,
            reused_flow_id=self.procedure_id,
        )

    def matches_repair_target(self, observation: Observation) -> bool:
        return observation.keyboard_visible is True

    def rejoin(self, observation: Observation) -> bool:
        self._rejoined = observation.keyboard_visible is True
        return self._rejoined

    def grounding_match(self, request: GroundingRequest) -> None:
        del request
        return None

    def authorize_keyboard_repair_grounding(
        self,
        proposal: ActionProposal,
        observation: Observation,
    ) -> bool:
        del proposal, observation
        return False

    def repair_would_repeat_failed_action(self, proposal: ActionProposal) -> bool:
        del proposal
        return False


def _repair_source(
    *,
    maximum_actions: int = 8,
    maximum_total_actions: int | None = None,
    planner: RepairPlanner | None = None,
) -> tuple[BoundedProcedureRepairActionSource, RepairPlanner]:
    evidence = _evidence()
    procedure = compile_replay_procedure(evidence, _response())
    assert isinstance(procedure, ReplayProcedure)
    planner = planner or RepairPlanner()
    source = BoundedProcedureRepairActionSource(
        CompiledProcedureActionSource(procedure, evidence),
        planner,
        maximum_actions_per_attempt=maximum_actions,
        maximum_total_actions=maximum_total_actions or maximum_actions,
    )
    return source, planner


def _request(
    observation: Observation,
    history: tuple[PlanningHistoryEntry, ...] = (),
) -> PlanningRequest:
    return PlanningRequest(
        observation=observation,
        objective="Replay the compiled procedure.",
        completion_conditions=("Every compiled expectation passes.",),
        available_actions=frozenset(ActionKind),
        history=history,
    )


def _outcome(
    proposal: ActionProposal,
    before: Observation,
    after: Observation,
) -> PlanningHistoryEntry:
    outcome = (
        agent_result(
            before,
            after,
            action=proposal.action,
            terminal=AgentTerminalReason.BUDGET_EXHAUSTED,
        )
        .steps[0]
        .outcome
    )
    assert outcome is not None
    return PlanningHistoryEntry(proposal=proposal, outcome=outcome)


def test_source_divergence_repairs_only_to_the_same_compiled_index() -> None:
    source, planner = _repair_source()
    wrong = _observations()[-1]

    assert not source.requires_screen_analysis()
    assert source.required_model_requests(_request(wrong)) == 0
    boundary = source.plan(_request(wrong))
    assert boundary.action is ActionKind.WAIT
    assert source.divergence is not None
    assert source.requires_screen_analysis()

    outer_request = _request(wrong).model_copy(
        update={
            "progress_context": TaskProgressContext(
                recovery_guidance=("Do not reopen Edit Record.",)
            )
        }
    )
    assert source.required_model_requests(outer_request) == 1
    repair_action = source.plan(outer_request)
    assert repair_action.action is ActionKind.WAIT
    assert planner.requests[0].objective.startswith("Restore the application")
    assert planner.requests[0].reference_screenshot is not None
    assert any(
        fact.startswith("Expected repair state metadata:")
        for fact in planner.requests[0].known_facts
    )
    remaining_fact = next(
        fact
        for fact in planner.requests[0].known_facts
        if fact.startswith("Remaining compiled step 0:")
    )
    assert "action=tap" in remaining_fact
    assert "expected outcome=The editor opens." in remaining_fact
    assert any(
        "completion conditions=1 of 1: The named item is saved." in fact
        for fact in planner.requests[0].known_facts
    )
    assert source.last_planning_model_request_count() == 1
    verification = source.action_verification_context(outer_request)
    assert verification.request.objective.startswith("Restore the application")
    assert verification.request.progress_context is None
    assert verification.request.screen_facts == ()
    assert verification.request.open_work == ()
    assert not verification.assess_task_progress

    launch = _observations()[0]
    assert source.required_model_requests(_request(launch)) == 0
    replay_action = source.plan(_request(launch))
    assert not source.requires_screen_analysis()

    assert replay_action.action is ActionKind.TAP
    assert source.decision_metadata().reused_segment_index == 0
    assert source.repair_evidence is not None
    assert source.repair_evidence.succeeded
    assert source.repair_evidence.rejoined_observation_id == launch.observation_id


def test_expectation_divergence_does_not_repeat_the_dispatched_action() -> None:
    source, unused_planner = _repair_source()
    before = _observations()[0]
    expected_destination = _observations()[2]
    wrong_destination = _observations()[-1]

    dispatched = source.plan(_request(before))
    boundary = source.plan(
        _request(
            wrong_destination,
            (_outcome(dispatched, before, wrong_destination),),
        )
    )
    assert boundary.action is ActionKind.WAIT
    assert source.completed_step_count == 0

    resumed = source.plan(_request(expected_destination))

    assert resumed.action is ActionKind.TYPE
    assert source.decision_metadata().reused_segment_index == 1
    assert resumed.proposal_id != dispatched.proposal_id
    assert source.repair_evidence is not None
    assert source.repair_evidence.target.resume_procedure_index == 1
    assert source.repair_evidence.target.established_by_action is ActionKind.TAP
    remaining = source.repair_evidence.target.remaining_step_summaries
    assert remaining[0].action is ActionKind.TYPE
    assert remaining[0].expected_outcome == "The name field contains the text."

    next_destination = _observations()[3]
    suffix_action = source.plan(
        _request(
            next_destination,
            (_outcome(resumed, expected_destination, next_destination),),
        )
    )
    assert suffix_action.action is ActionKind.TAP
    assert source.decision_metadata().reused_segment_index == 2
    assert source.completed_step_count == 2


def test_two_divergences_produce_two_independent_repairs() -> None:
    source, unused_planner = _repair_source()
    launch, first_destination, second_destination, wrong = (
        _observations()[0],
        _observations()[2],
        _observations()[3],
        _observations()[-1],
    )

    assert source.plan(_request(wrong)).action is ActionKind.WAIT
    first_action = source.plan(_request(launch))
    assert first_action.action is ActionKind.TAP
    second_action = source.plan(
        _request(first_destination, (_outcome(first_action, launch, first_destination),))
    )
    assert second_action.action is ActionKind.TYPE

    second_boundary = source.plan(
        _request(wrong, (_outcome(second_action, first_destination, wrong),))
    )

    assert second_boundary.action is ActionKind.WAIT
    assert len(source.divergences) == 2
    assert len(source.repairs) == 2
    assert source.repairs[0].succeeded
    assert not source.repairs[1].succeeded

    resumed = source.plan(_request(second_destination))

    assert resumed.action is ActionKind.TAP
    assert len(source.repairs) == 2
    assert all(repair.succeeded for repair in source.repairs)


def test_trusted_system_surface_is_dismissed_without_planner_repair() -> None:
    source, planner = _repair_source()
    overlay = _observations()[0].model_copy(
        update={
            "observation_id": "system_overlay",
            "foreground_application_id": "com.android.systemui",
            "platform_context": "notification_shade",
        }
    )

    action = source.plan(_request(overlay))

    assert action.action is ActionKind.BACK
    assert planner.request_count == 0
    assert source.divergence is None
    assert len(source.environment_events) == 1
    assert source.environment_events[0].surface == "system_ui"


def test_keyboard_repair_resets_focus_without_a_planner_call() -> None:
    observation = _observations()[0].model_copy(
        update={"observation_id": "keyboard_closed", "keyboard_visible": False}
    )
    target = ProcedureRepairTarget(
        expected_state_id="state_password_focused",
        resume_procedure_index=4,
        presentation=observation.presentation_key.model_copy(update={"keyboard_visible": True}),
        established_by_action=ActionKind.FOCUS,
        established_by_target_expression="the Password text field",
        keyboard_reset_target_expression="the Email text field",
        reason="The keyboard closed after focus.",
    )
    planner = RepairPlanner()
    procedure_source = cast(
        CompiledProcedureActionSource,
        KeyboardProcedureSource(target, observation),
    )
    source = BoundedProcedureRepairActionSource(procedure_source, planner)

    boundary = source.plan(_request(observation))
    assert boundary.action is ActionKind.WAIT
    assert source.required_model_requests(_request(observation)) == 0
    required_focus = source.plan(_request(observation))
    assert required_focus.action is ActionKind.FOCUS
    assert required_focus.target_expression == "the Password text field"

    keyboard_open = observation.model_copy(
        update={"observation_id": "keyboard_open", "keyboard_visible": True}
    )
    assert source.required_model_requests(_request(keyboard_open)) == 0
    resumed = source.plan(_request(keyboard_open))

    assert resumed.action is ActionKind.TYPE
    assert planner.request_count == 0
    assert source.repair_evidence is not None
    assert source.repair_evidence.action_count == 2
    assert source.repair_evidence.succeeded


def test_repair_refuses_exact_retry_of_failed_non_navigation_action() -> None:
    evidence = _evidence()
    procedure = compile_replay_procedure(evidence, _response())
    assert isinstance(procedure, ReplayProcedure)
    failed_step = procedure.steps[0].model_copy(update={"procedure_effect": "state_setting"})
    procedure = procedure.model_copy(update={"steps": (failed_step, *procedure.steps[1:])})
    retry = failed_step.proposal.model_copy(update={"proposal_id": "unsafe_retry"})
    planner = FixedRepairPlanner(retry)
    source = BoundedProcedureRepairActionSource(
        CompiledProcedureActionSource(procedure, evidence),
        planner,
    )
    before = _observations()[0]
    wrong = _observations()[-1]
    dispatched = source.plan(_request(before))
    boundary = source.plan(_request(wrong, (_outcome(dispatched, before, wrong),)))

    assert boundary.action is ActionKind.WAIT
    assert source.required_model_requests(_request(wrong)) == 1
    stopped = source.plan(_request(wrong))

    assert stopped.action is ActionKind.CANNOT_PROGRESS
    assert stopped.reason is not None and "refused to repeat" in stopped.reason
    assert source.repair_evidence is not None
    assert source.repair_evidence.action_count == 1


def test_repair_action_limit_stops_without_general_task_completion() -> None:
    source, planner = _repair_source(maximum_actions=2)
    wrong = _observations()[-1]

    boundary = source.plan(_request(wrong))
    assert boundary.action is ActionKind.WAIT
    assert source.required_model_requests(_request(wrong)) == 1
    repair_action = source.plan(_request(wrong))
    assert repair_action.action is ActionKind.WAIT
    assert source.required_model_requests(_request(wrong)) == 0

    stopped = source.plan(_request(wrong))

    assert stopped.action is ActionKind.CANNOT_PROGRESS
    assert planner.request_count == 1
    assert source.repair_evidence is not None
    assert not source.repair_evidence.succeeded
    assert source.repair_evidence.action_count == 2
    assert source.repair_evidence.reason.endswith("bounded action allowance.")


def test_completed_attempt_does_not_disable_later_environment_recovery() -> None:
    source, unused_planner = _repair_source(
        maximum_actions=8,
        maximum_total_actions=32,
    )
    expected_source = _observations()[0]
    wrong = _observations()[-1]

    boundary = source.plan(_request(wrong))
    assert boundary.action is ActionKind.WAIT
    for _unused_index in range(7):
        assert source.plan(_request(wrong)).action is ActionKind.WAIT
    rejoined = source.plan(_request(expected_source))
    assert rejoined.action is not ActionKind.CANNOT_PROGRESS
    assert source.repairs[-1].succeeded
    assert source.repairs[-1].action_count == 8

    system_surface = expected_source.model_copy(
        update={
            "observation_id": "obs_system_surface",
            "foreground_application_id": "com.android.systemui",
        }
    )
    environment_action = source.plan(_request(system_surface))

    assert environment_action.action is ActionKind.BACK
    assert len(source.environment_events) == 1


def test_task_completion_after_repair_remains_a_diverged_procedure() -> None:
    source, unused_planner = _repair_source()
    wrong = _observations()[-1]
    source.plan(_request(wrong))
    divergence = source.divergence
    assert divergence is not None

    terminal = _procedure_terminal(
        AgentTerminalReason.DONE_REQUESTED,
        source,
        divergence,
        source.repair_evidence,
        None,
    )

    assert terminal is ProcedureReplayTerminal.DIVERGED


def test_successful_repair_can_be_audited_and_complete() -> None:
    source = cast(
        CompiledProcedureActionSource,
        type("CompletedSource", (), {"replay_complete": True})(),
    )
    divergence = ProcedureDivergence(
        procedure_id="procedure_1",
        procedure_index=1,
        source_step=2,
        guard="expectation",
        expected_state_id="state_2",
        actual_observation_id="obs_wrong",
        actual_fingerprint=_observations()[0].screen_fingerprint,
        reason="The expectation diverged before repair.",
    )
    repair = ProcedureRepairEvidence(
        target=ProcedureRepairTarget(
            expected_state_id="state_2",
            resume_procedure_index=2,
            presentation=_observations()[0].presentation_key,
            reason=divergence.reason,
        ),
        maximum_actions=8,
        action_count=3,
        model_call_count=0,
        succeeded=True,
        rejoined_observation_id="obs_rejoined",
        reason="The deterministic rejoin guard accepted the observation.",
    )
    verification = GoalVerification(
        verification_id="verification_repaired",
        objective_id="objective_1",
        final_observation_id="obs_final",
        satisfied=True,
        reason="The repaired replay satisfied the task.",
    )

    assert _can_audit_replay(
        AgentTerminalReason.DONE_REQUESTED,
        source,
        divergence,
        repair,
    )
    assert (
        _procedure_terminal(
            AgentTerminalReason.DONE_REQUESTED,
            source,
            divergence,
            repair,
            verification,
        )
        is ProcedureReplayTerminal.COMPLETED
    )
