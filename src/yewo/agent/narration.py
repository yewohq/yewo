"""Human-readable Device Agent narration for one run."""

from __future__ import annotations

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


_SOURCE = "device-agent"


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


def narrate_proposal(step_index: int, proposal: ActionProposal) -> None:
    """Describe one planner decision without exposing typed values."""
    narrate(f"Step {step_index} decision: {proposal.reason}")
    trace(
        f"step={step_index} proposal={proposal.proposal_id} "
        f"observation={proposal.source_observation_id} action={proposal.action.value}"
    )
    action = proposal.action.value
    target = _target_summary(proposal)
    if proposal.expected_outcome is None:
        narrate(f"Step {step_index} requested {action}.")
        return
    narrate(f"Step {step_index} planned {action}{target}. Expected: {proposal.expected_outcome}")


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
        narrate(f"Step {step_index} grounded the {label}: {evidence.result.disposition.value}.")
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
            f"profile={evidence.result.profile_id} model_profile={evidence.result.model_profile_id}"
        )


def narrate_recovery(step_index: int, recovery_observation_id: str | None) -> None:
    """Describe an application recovery when one occurred."""
    if recovery_observation_id is not None:
        narrate(f"Step {step_index} recovered the application to the foreground.")
        trace(f"step={step_index} recovery_observation={recovery_observation_id}")


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
