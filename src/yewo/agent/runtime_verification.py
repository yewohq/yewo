"""Run-scoped pre-dispatch and post-recovery model decisions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from yewo.agent.action_diagnostics import ActionApplicationFailure
from yewo.agent.action_verifier import ActionVerifier
from yewo.agent.actuation import ActionProposal, ActionRefusal
from yewo.agent.narration import narrate
from yewo.agent.planner import PlanningRequest
from yewo.agent.planning_evidence import PlannerDecisionSource
from yewo.agent.recovery_advisor import RecoveryAssessment
from yewo.agent.recovery_runtime import RecoveryRunState
from yewo.agent.runtime_records import AgentModePolicy, AgentStep, invocation_budget_reason
from yewo.observations import Observation


@dataclass(frozen=True)
class PreDispatchVerification:
    """Budget and refusal result for one optional discovery verification."""

    request_count: int = 0
    refusal: ActionRefusal | None = None
    budget_reason: str | None = None


def verify_discovery_action(
    verifier: ActionVerifier | None,
    discovery_mode: bool,
    decision_source: PlannerDecisionSource,
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
    if (
        not discovery_mode
        or verifier is None
        or decision_source is not PlannerDecisionSource.MODEL
    ):
        return PreDispatchVerification()
    request_count = verifier.required_model_requests(proposal)
    if not request_count:
        return PreDispatchVerification()
    verification_was_recovery = recovery.is_active
    recovery.credit_planning_requests(request_count)
    credit = recovery.credit(completed_step_count, now)
    budget_reason = invocation_budget_reason(
        mode_policy,
        max(0.0, now - started_at - credit.duration_seconds),
        planning_requests
        + grounding_requests
        - credit.planning_requests
        - credit.grounding_requests,
        0 if verification_was_recovery else request_count,
    )
    if budget_reason is not None:
        return PreDispatchVerification(budget_reason=budget_reason)
    assessment = verifier.verify(planning_request, proposal)
    narrate(
        "Discovery action verification "
        f"{assessment.disposition.value}: {assessment.reason}"
    )
    return PreDispatchVerification(
        request_count=request_count,
        refusal=assessment.refusal(),
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
