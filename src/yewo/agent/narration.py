"""Human-readable Device Agent narration for one run."""

from __future__ import annotations

from collections.abc import Callable

from yewo.agent.actuation.diagnostics import ActionApplicationFailure
from yewo.agent.actuation import (
    ActionOutcome,
    ActionProposal,
    ActionRefusal,
    AuthorizedAction,
    DispatchReceipt,
)
from yewo.grounding import GroundedPoint, GroundedRegion, GroundingEvidence
from yewo.observations import Observation
from yewo.run_narration import current_narrator
from yewo.agent.planning.planner import ActionProposalSource
from yewo.agent.planning.evidence import (
    guard_planner_decision,
    PlannerDecisionMetadata,
    PlannerDecisionMetadataSource,
    PlannerDecisionSource,
)
from yewo.agent.recovery.advisor import RecoveryAssessment
from yewo.agent.recovery.records import RecoveryBudgetCredit


_SOURCE = "device-agent"


def narrate_memory_eviction(kind: str, record_ids: tuple[str, ...]) -> None:
    """Record one bounded-memory eviction without exposing application content."""
    narrator = current_narrator()
    if narrator is None or not record_ids:
        return
    narrator.record(
        "agent-memory",
        f"Memory retention removed {len(record_ids)} older {kind} record(s).",
    )
    narrator.trace(
        "agent-memory",
        f"retention=evicted kind={kind} count={len(record_ids)} ids={','.join(record_ids)}",
    )


def narrate(message: str) -> None:
    """Append one Device Agent message when a run narrator is active."""
    narrator = current_narrator()
    if narrator is not None:
        narrator.record(_SOURCE, message)


def trace(message: str) -> None:
    """Append one Device Agent diagnostic when a run narrator is active."""
    narrator = current_narrator()
    if narrator is not None:
        narrator.trace(_SOURCE, message)


def narrate_proposal(
    step_index: int,
    proposal: ActionProposal,
    decision_source: PlannerDecisionSource,
    reused_flow_id: str | None,
) -> None:
    """Describe one planner decision without exposing typed values."""
    if decision_source is PlannerDecisionSource.MODEL:
        narrate(f"Step {step_index} requested a new planner decision.")
    elif decision_source is PlannerDecisionSource.SYSTEM:
        narrate(f"Step {step_index} stopped before dispatch under a runtime guard.")
    else:
        narrate(f"Step {step_index} reused a {decision_source.value.replace('_', ' ')} decision.")
    narrate(f"Step {step_index} decision: {proposal.reason}")
    trace(
        f"step={step_index} proposal={proposal.proposal_id} "
        f"observation={proposal.source_observation_id} action={proposal.action.value} "
        f"planner_source={decision_source.value} "
        f"reused_flow={reused_flow_id or 'none'}"
    )
    action = proposal.action.value
    target = _target_summary(proposal)
    if proposal.expected_outcome is None:
        narrate(f"Step {step_index} requested {action}.")
        return
    narrate(f"Step {step_index} planned {action}{target}. Expected: {proposal.expected_outcome}")


def narrate_selected_proposal(
    planner: ActionProposalSource,
    step_index: int,
    proposal: ActionProposal,
    default_source: PlannerDecisionSource,
    default_reused_flow_id: str | None,
    identifier_factory: Callable[[], str],
    *,
    system_generated: bool = False,
) -> tuple[ActionProposal, PlannerDecisionMetadata]:
    """Describe a proposal with dynamic replay provenance when available."""
    if system_generated:
        metadata = PlannerDecisionMetadata(source=PlannerDecisionSource.SYSTEM)
    elif isinstance(planner, PlannerDecisionMetadataSource):
        metadata = planner.decision_metadata()
    else:
        metadata = PlannerDecisionMetadata(
            source=default_source,
            reused_flow_id=default_reused_flow_id,
        )
    proposal, metadata = guard_planner_decision(proposal, metadata, identifier_factory)
    narrate_proposal(
        step_index,
        proposal,
        metadata.source,
        metadata.reused_flow_id,
    )
    return proposal, metadata


def narrate_refusal(step_index: int, refusal: ActionRefusal) -> None:
    """Describe why policy rejected one proposal."""
    narrate(f"Step {step_index} refused ({refusal.code.value}): {refusal.reason}")
    trace(f"step={step_index} proposal={refusal.proposal_id} refusal={refusal.code.value}")


def narrate_grounding(
    step_index: int,
    label: str,
    evidence: GroundingEvidence | None,
) -> None:
    """Describe the disposition of one optional grounding request."""
    if evidence is not None:
        source = evidence.result.resolution_source.value
        narrate(
            f"Step {step_index} grounded the {label}: "
            f"{evidence.result.disposition.value} via {source}."
        )
        location = evidence.result.location
        if isinstance(location, GroundedPoint):
            geometry = f"point=({location.pixel.x},{location.pixel.y})"
        elif isinstance(location, GroundedRegion):
            region = location.pixel
            geometry = f"region=({region.left},{region.top},{region.right},{region.bottom})"
        else:
            geometry = "geometry=none"
        trace(
            f"step={step_index} grounding={evidence.request.grounding_id} label={label} "
            f"operation={evidence.request.operation.value} "
            f"disposition={evidence.result.disposition.value} {geometry} "
            f"source={source} profile={evidence.result.profile_id} "
            f"model_profile={evidence.result.model_profile_id} "
            f"cached_transition={evidence.result.cached_transition_id or 'none'}"
        )


def narrate_recovery(step_index: int, recovery_observation_id: str | None) -> None:
    """Describe an application recovery when one occurred."""
    if recovery_observation_id is not None:
        narrate(f"Step {step_index} recovered the application to the foreground.")
        trace(f"step={step_index} recovery_observation={recovery_observation_id}")


def narrate_recovery_assessment(assessment: RecoveryAssessment) -> None:
    """Record one bounded model recovery decision and its evidence identity."""
    narrate(f"Recovery advisor chose {assessment.disposition.value}: {assessment.reason}")
    trace(
        f"recovery_assessment={assessment.assessment_id} "
        f"disposition={assessment.disposition.value} "
        f"failed_proposal={assessment.failed_proposal_id} "
        f"frontier_observation={assessment.pre_failure_observation_id} "
        f"recovery_observation={assessment.recovery_observation_id}"
    )


def narrate_recovery_unavailable() -> None:
    """Record that recovery advice failed and the run continues without it."""
    narrate("Recovery advice was unavailable. The agent continues without it.")
    trace("recovery_assessment=none disposition=unavailable")


def narrate_recovery_credit(credit: RecoveryBudgetCredit) -> None:
    """Trace gross-to-effective allowance without changing provider usage."""
    if credit.assessment_id is None:
        return
    trace(
        f"recovery_credit assessment={credit.assessment_id} steps={credit.steps} "
        f"planning={credit.planning_requests} grounding={credit.grounding_requests} "
        f"actions={credit.dispatched_actions} duration_seconds={credit.duration_seconds:.3f}"
    )


def narrate_observation(observation: Observation, *, label: str) -> None:
    """Describe and index one persisted observation without screenshot content."""
    keyboard = (
        "unknown"
        if observation.keyboard_visible is None
        else "visible"
        if observation.keyboard_visible
        else "not visible"
    )
    narrate(
        f"{label} observation: {observation.application_state.name.lower()}, "
        f"context {observation.platform_context or 'unknown'}, keyboard {keyboard}."
    )
    trace(
        f"observation={observation.observation_id} label={label} "
        f"predecessor={observation.predecessor_observation_id or 'none'} "
        f"application_state={observation.application_state.name.lower()} "
        f"foreground={observation.foreground_application_id or 'unknown'} "
        f"context={observation.platform_context or 'unknown'} "
        f"keyboard={observation.keyboard_visible if observation.keyboard_visible is not None else 'unknown'} "
        f"fingerprint={observation.screen_fingerprint.value} "
        f"masked_pixels_sha256={observation.screen_fingerprint.masked_pixels_sha256} "
        f"screenshot={observation.redacted_screenshot.path}"
    )


def trace_authorization(step_index: int, action: AuthorizedAction) -> None:
    """Index one exact policy authorization without recording typed text."""
    target = f"({action.target.x},{action.target.y})" if action.target is not None else "none"
    trace(
        f"step={step_index} authorization={action.authorization_id} "
        f"proposal={action.proposal_id} action={action.action.value} "
        f"side_effect={action.side_effect.value} source={action.source.value} target={target}"
    )


def trace_dispatch(step_index: int, receipt: DispatchReceipt) -> None:
    """Index one device dispatch receipt."""
    trace(
        f"step={step_index} receipt={receipt.receipt_id} "
        f"authorization={receipt.authorization_id} action={receipt.action.value} "
        f"disposition={receipt.disposition.value}"
    )


def narrate_outcome(step_index: int, outcome: ActionOutcome) -> None:
    """Describe and index one deterministic action outcome."""
    signals = ", ".join(sorted(signal.value for signal in outcome.signals)) or "none"
    narrate(
        f"Step {step_index} outcome {outcome.disposition.value}. "
        f"Signals: {signals}. {outcome.reason}"
    )
    trace(
        f"step={step_index} outcome={outcome.outcome_id} "
        f"authorization={outcome.authorization_id} receipt={outcome.receipt_id} "
        f"source_observation={outcome.source_observation_id} "
        f"settled_observation={outcome.settled_observation_id or 'none'} "
        f"disposition={outcome.disposition.value} signals={signals.replace(', ', ',')}"
    )


def narrate_no_progress(message: str) -> None:
    """Record a deterministic no-progress advisory."""
    narrate(f"No-progress advisory: {message}")
    trace(f"no_progress={message}")


def narrate_application_failure(
    step_index: int,
    failure: ActionApplicationFailure | None,
) -> None:
    """Describe one action-scoped application failure and recovery result."""
    if failure is None:
        return
    recovery = "succeeded" if failure.recovery_succeeded else "did not succeed"
    narrate(
        f"Step {step_index} application diagnostic: "
        f"{failure.summary} Recovery {recovery}."
    )


def narrate_verification(
    *,
    verification_id: str,
    final_observation_id: str,
    satisfied: bool,
    reason: str,
) -> None:
    """Describe and index one independent goal-verification result."""
    verdict = "satisfied" if satisfied else "not satisfied"
    narrate(f"Independent goal verification {verdict}: {reason}")
    trace(
        f"verification={verification_id} final_observation={final_observation_id} "
        f"satisfied={str(satisfied).lower()}"
    )


def _target_summary(proposal: ActionProposal) -> str:
    if proposal.target_expression is not None:
        target = f" on {proposal.target_expression}"
        if proposal.destination_expression is not None:
            target += f" toward {proposal.destination_expression}"
        return target
    if proposal.direction is not None:
        return f" {proposal.direction.value}"
    return ""
