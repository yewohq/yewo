"""Run-scoped pre-dispatch and post-recovery model decisions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from yewo.agent.actuation.diagnostics import ActionApplicationFailure
from yewo.agent.actuation.proposal_verifier import ProposalVerifier
from yewo.agent.actuation import ActionProposal, ActionRefusal
from yewo.agent.narration import narrate, narrate_recovery_assessment
from yewo.agent.planning.planner import (
    resolve_action_verification_context,
)
from yewo.agent.planning.records import (
    PlanningRequest,
)
from yewo.agent.planning.evidence import PlannerDecisionSource
from yewo.agent.recovery.advisor import RecoveryAssessment, RecoveryDisposition
from yewo.agent.recovery.runtime import RecoveryRunState
from yewo.agent.runtime.records import AgentModePolicy, AgentStep, invocation_budget_reason
from yewo.observations import Observation


@dataclass(frozen=True)
class PreDispatchVerification:
    """Budget and refusal result for one optional discovery verification."""

    request_count: int = 0
    refusal: ActionRefusal | None = None
    budget_reason: str | None = None
    progress_assessed: bool = False
    completed_work_items: tuple[str, ...] = ()
    proposal_work_item: str | None = None
    restarts_completed_work_item: str | None = None


def goal_verification_budget_reason(
    policy: AgentModePolicy,
    model_requests: int,
    elapsed_seconds: float,
) -> str | None:
    """Return why a final auditor request cannot start."""
    if elapsed_seconds >= policy.maximum_duration_seconds:
        return "The duration budget ended before final verification."
    if (
        policy.maximum_model_requests is not None
        and model_requests >= policy.maximum_model_requests
    ):
        return "The model-request budget ended before final verification."
    return None


def verify_discovery_action(
    verifier: ProposalVerifier | None,
    discovery_mode: bool,
    decision_source: PlannerDecisionSource,
    planning_source: object,
    planning_request: PlanningRequest,
    proposal: ActionProposal,
    mode_policy: AgentModePolicy,
    recovery: RecoveryRunState,
    completed_step_count: int,
    started_at: float,
    planning_requests: int,
    grounding_requests: int,
    now: float,
) -> PreDispatchVerification:
    """Assess model-selected discovery navigation within the shared budget."""
    if not discovery_mode or verifier is None or decision_source is not PlannerDecisionSource.MODEL:
        return PreDispatchVerification()
    request_count = verifier.required_model_requests(proposal)
    if not request_count:
        return PreDispatchVerification()
    verification_was_recovery = recovery.is_active
    credit = recovery.credit(completed_step_count, now)
    budget_reason = invocation_budget_reason(
        mode_policy,
        max(0.0, now - started_at - credit.duration_seconds),
        planning_requests
        + grounding_requests
        - credit.planning_requests
        - credit.analysis_requests
        - credit.grounding_requests,
        0 if verification_was_recovery else request_count,
    )
    if budget_reason is not None:
        return PreDispatchVerification(budget_reason=budget_reason)
    context = resolve_action_verification_context(planning_source, planning_request)
    assessment = verifier.verify(context.request, proposal)
    recovery.credit_planning_requests(request_count)
    narrate(
        f"Step proposal check (before dispatch) {assessment.disposition.value}: {assessment.reason}"
    )
    return PreDispatchVerification(
        request_count=request_count,
        refusal=assessment.refusal(),
        progress_assessed=context.assess_task_progress,
        completed_work_items=(
            assessment.completed_work_items if context.assess_task_progress else ()
        ),
        proposal_work_item=(
            assessment.proposal_work_item if context.assess_task_progress else None
        ),
        restarts_completed_work_item=(
            assessment.restarts_completed_work_item if context.assess_task_progress else None
        ),
    )


def assess_recovered_failure(
    recovery: RecoveryRunState,
    failure: ActionApplicationFailure | None,
    request: PlanningRequest,
    step: AgentStep,
    observation: Observation,
    step_count: int,
    monotonic: Callable[[], float],
) -> RecoveryAssessment | None:
    """Run the single advisor only after a successful application relaunch."""
    if failure is None or not failure.recovery_succeeded:
        return None
    return recovery.assess(
        request,
        step,
        observation,
        step_count,
        monotonic(),
    )


def recovery_stop_reason(
    recovery: RecoveryRunState,
    failure: ActionApplicationFailure | None,
    request: PlanningRequest,
    step: AgentStep,
    observation: Observation,
    step_count: int,
    monotonic: Callable[[], float],
) -> tuple[str | None, int]:
    """Narrate one recovery assessment and report its stop reason and model cost."""
    assessment = assess_recovered_failure(
        recovery, failure, request, step, observation, step_count, monotonic
    )
    if assessment is None:
        return None, 0
    narrate_recovery_assessment(assessment)
    if assessment.disposition is RecoveryDisposition.STOP:
        return assessment.reason, 1
    return None, 1
