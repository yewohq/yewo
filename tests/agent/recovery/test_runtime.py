"""Recovery frontier matching and effective-budget credits."""

import pytest
from pydantic import ValidationError

from yewo.agent.actuation import ActionKind
from yewo.observations import ScreenComparisonPolicy
from yewo.agent.planning.records import (
    PlanningRequest,
    TaskProgressContext,
)
from yewo.agent.recovery.advisor import RecoveryAssessment, RecoveryDisposition
from yewo.agent.recovery.records import RecoveryBudgetCredit
from yewo.agent.recovery.runtime import RecoveryRunState
from yewo.agent.runtime.records import AgentResult, AgentStep, AgentTerminalReason, AgentUsage
from yewo.observations import Observation

from agent.actuation.support import proposal
from agent.discovery.support import observation


class ContinuingAdvisor:
    """Return one configured recovery continuation."""

    def assess(
        self,
        planning_request: PlanningRequest,
        failed_step: AgentStep,
        recovery_observation: Observation,
    ) -> RecoveryAssessment:
        del planning_request, failed_step, recovery_observation
        return RecoveryAssessment(
            assessment_id="assessment_1",
            failed_proposal_id="proposal_failed",
            pre_failure_observation_id="obs_frontier",
            recovery_observation_id="obs_fresh",
            disposition=RecoveryDisposition.CONTINUE,
            reason="The task can return to the recorded frontier.",
            remaining_work=("Restore the form state.",),
            planner_guidance=("Do not repeat the failed submission unchanged.",),
            candidate_causes=("A stale internal value may have caused the failure.",),
            model_request_id="request_1",
            model_call_count=1,
        )


def _failed_step() -> AgentStep:
    failed_proposal = proposal(
        ActionKind.TAP,
        proposal_id="proposal_failed",
        source_observation_id="obs_frontier",
    )
    return AgentStep(
        step_id="step_failed",
        step_index=1,
        source_observation_id="obs_frontier",
        proposal=failed_proposal,
    )


def _planning_request(frontier: Observation) -> PlanningRequest:
    return PlanningRequest(
        observation=frontier,
        objective="Save the record.",
        completion_conditions=("It is saved.",),
        available_actions=frozenset({ActionKind.TAP}),
    )


def test_recovery_credit_ends_when_the_recorded_frontier_matches() -> None:
    frontier = observation("obs_frontier", "1111111111111111")
    fresh = observation("obs_fresh", "0000000000000000", offset_seconds=1)
    intermediate = observation("obs_middle", "2222222222222222", offset_seconds=2)
    matched = observation("obs_matched", "1111111111111111", offset_seconds=3)
    state = RecoveryRunState(ContinuingAdvisor(), ScreenComparisonPolicy())

    assessment = state.assess(
        _planning_request(frontier),
        _failed_step(),
        fresh,
        completed_step_count=4,
        now=10,
    )
    state.credit_planning_requests(2)
    state.credit_analysis_requests(1)
    state.credit_grounding_requests(1)
    state.credit_dispatched_action()
    state.observe(intermediate, completed_step_count=5, now=12)

    assert assessment is not None
    assert state.is_active
    assert state.progress_context() is not None
    state.observe(matched, completed_step_count=6, now=14)
    credit = state.credit(completed_step_count=6, now=20)

    assert not state.is_active
    assert credit.steps == 2
    assert credit.planning_requests == 3
    assert credit.analysis_requests == 1
    assert credit.grounding_requests == 1
    assert credit.dispatched_actions == 1
    assert credit.duration_seconds == 4


def test_recovery_credit_is_capped_at_ten_steps() -> None:
    frontier = observation("obs_frontier", "1111111111111111")
    fresh = observation("obs_fresh", "0000000000000000", offset_seconds=1)
    still_different = observation("obs_other", "2222222222222222", offset_seconds=2)
    state = RecoveryRunState(ContinuingAdvisor(), ScreenComparisonPolicy())
    state.assess(
        _planning_request(frontier),
        _failed_step(),
        fresh,
        completed_step_count=2,
        now=0,
    )

    state.observe(still_different, completed_step_count=12, now=20)

    assert not state.is_active
    assert state.credit(completed_step_count=20, now=30).steps == 10


def test_credit_is_pure_and_settle_ends_a_step_capped_interval() -> None:
    frontier = observation("obs_frontier", "1111111111111111")
    fresh = observation("obs_fresh", "0000000000000000", offset_seconds=1)
    state = RecoveryRunState(ContinuingAdvisor(), ScreenComparisonPolicy())
    state.assess(
        _planning_request(frontier),
        _failed_step(),
        fresh,
        completed_step_count=2,
        now=0,
    )

    credit = state.credit(completed_step_count=12, now=20)

    assert state.is_active
    assert credit.steps == 10
    state.settle(completed_step_count=12, now=20)
    assert not state.is_active
    assert state.credit(completed_step_count=20, now=30).steps == 10


def test_recovery_credit_cannot_exceed_gross_usage() -> None:
    with pytest.raises(ValidationError, match="cannot exceed gross usage"):
        AgentResult(
            terminal_reason=AgentTerminalReason.CANNOT_PROGRESS,
            reason="The task stopped.",
            final_observation_id="obs_1",
            usage=AgentUsage(
                planning_requests=0,
                grounding_requests=0,
                dispatched_actions=0,
            ),
            recovery_budget_credit=RecoveryBudgetCredit(planning_requests=1),
            steps=(),
        )


def test_advisor_text_reaches_the_planner_as_guidance_not_as_a_work_item() -> None:
    frontier = observation("obs_frontier", "0000000000000000")
    fresh = observation("obs_fresh", "1111111111111111", offset_seconds=1)
    state = RecoveryRunState(ContinuingAdvisor(), ScreenComparisonPolicy())
    state.assess(
        _planning_request(frontier),
        _failed_step(),
        fresh,
        completed_step_count=4,
        now=10,
    )

    context = state.progress_context()

    assert context is not None
    assert context.remaining_work == ()
    assert "Remaining work: Restore the form state." in context.recovery_guidance
    assert "Do not repeat the failed submission unchanged." in context.recovery_guidance
    assert "Unproven recovery cause: A stale internal value may have caused the failure." in (
        context.recovery_guidance
    )


def test_failed_branch_constraint_survives_full_recovery_guidance() -> None:
    frontier = observation("obs_frontier", "0000000000000000")
    fresh = observation("obs_fresh", "1111111111111111", offset_seconds=1)
    state = RecoveryRunState(ContinuingAdvisor(), ScreenComparisonPolicy())
    state.assess(
        _planning_request(frontier),
        _failed_step(),
        fresh,
        completed_step_count=4,
        now=10,
    )
    base = TaskProgressContext(
        recovery_guidance=tuple(f"Existing guidance {index}." for index in range(20))
    )

    context = state.progress_context(base)

    assert context is not None
    assert context.recovery_guidance[0] == (
        "Do not repeat the application-failing action without a material state change."
    )
    assert len(context.recovery_guidance) == 20


class ManyWorkItemsAdvisor:
    """Return a continuation whose remaining work exceeds the guidance window."""

    def assess(
        self,
        planning_request: PlanningRequest,
        failed_step: AgentStep,
        recovery_observation: Observation,
    ) -> RecoveryAssessment:
        del planning_request, failed_step, recovery_observation
        return RecoveryAssessment(
            assessment_id="assessment_1",
            failed_proposal_id="proposal_failed",
            pre_failure_observation_id="obs_frontier",
            recovery_observation_id="obs_fresh",
            disposition=RecoveryDisposition.CONTINUE,
            reason="The task can still reach the remaining work.",
            remaining_work=tuple(f"Remaining item {index}." for index in range(40)),
            planner_guidance=("The platform rejected the submission.",),
            candidate_causes=(),
            model_request_id="request_1",
        )


def test_established_guidance_outranks_remaining_work_in_the_guidance_window() -> None:
    frontier = observation("obs_frontier", "0000000000000000")
    fresh = observation("obs_fresh", "1111111111111111", offset_seconds=1)
    state = RecoveryRunState(ManyWorkItemsAdvisor(), ScreenComparisonPolicy())
    state.assess(
        _planning_request(frontier),
        _failed_step(),
        fresh,
        completed_step_count=4,
        now=10,
    )

    context = state.progress_context()

    assert context is not None
    assert len(context.recovery_guidance) == 20
    assert "The platform rejected the submission." in context.recovery_guidance
    assert "Remaining work: Remaining item 0." in context.recovery_guidance
