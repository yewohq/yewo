"""Deterministic YewoAgent discovery evidence builders."""

from datetime import datetime, timedelta
from hashlib import sha256

from yewo.agent import (
    AgentResult,
    AgentStep,
    AgentTerminalReason,
    AgentUsage,
)
from yewo.agent.actuation import (
    ACTION_DEFINITIONS,
    ActionBudget,
    ActionKind,
    ActionOutcome,
    ActionOutcomeDisposition,
    ActionOutcomeSignal,
    ActionPolicy,
    ActionRefusal,
    AuthorizedAction,
    DeclaredValueReference,
    DispatchDisposition,
    DispatchReceipt,
)
from yewo.artifacts import ArtifactRetention
from yewo.observations import Observation, ScreenFingerprint

from agent.actuation.support import limits, point_grounding, proposal
from observations.factories import build_observation, build_screenshot


def observation(
    observation_id: str,
    fingerprint: str,
    *,
    predecessor_observation_id: str | None = None,
    offset_seconds: int = 0,
    mask_sha256: str | None = None,
) -> Observation:
    screenshot = build_screenshot(artifact_id=f"shot_{observation_id}")
    base = build_observation(
        observation_id=observation_id,
        screenshot=screenshot,
        predecessor_observation_id=predecessor_observation_id,
        captured_at=build_observation().captured_at + timedelta(seconds=offset_seconds),
    )
    return base.model_copy(
        update={
            "screen_fingerprint": ScreenFingerprint(
                value=fingerprint,
                masked_pixels_sha256=sha256(fingerprint.encode()).hexdigest(),
                system_region_mask_sha256=mask_sha256,
                source_artifact_id=screenshot.artifact_id,
            )
        }
    )


def agent_result(
    before: Observation,
    after: Observation,
    *,
    disposition: ActionOutcomeDisposition = ActionOutcomeDisposition.OBSERVED,
    terminal: AgentTerminalReason = AgentTerminalReason.DONE_REQUESTED,
    action: ActionKind = ActionKind.TAP,
    declared_value_name: str | None = None,
) -> AgentResult:
    policy_observation = (
        before.model_copy(update={"keyboard_visible": True})
        if action in {ActionKind.TYPE, ActionKind.CLEAR}
        else before
    )
    action_proposal = proposal(
        action,
        proposal_id="proposal_1",
        source_observation_id=before.observation_id,
        **(
            {"text": DeclaredValueReference(value_name=declared_value_name)}
            if declared_value_name is not None
            else {}
        ),
    )
    grounding = None
    if action_proposal.target_expression is not None:
        target_operation = ACTION_DEFINITIONS[action].target_operation
        if target_operation is None:
            raise AssertionError("The test action has no target operation.")
        grounding = point_grounding(
            screenshot_sha256=before.screenshot.sha256,
            expression=action_proposal.target_expression,
            operation=target_operation,
        )
        grounding = grounding.model_copy(
            update={
                "request": grounding.request.model_copy(update={"screenshot": before.screenshot})
            }
        )
    authorized = ActionPolicy(limits(), lambda: "authorization_1").authorize(
        action_proposal,
        policy_observation,
        ActionBudget(dispatched_actions=0),
        target_grounding=grounding,
    )
    if isinstance(authorized, ActionRefusal):
        raise AssertionError(authorized.reason)
    receipt = _receipt(authorized, before.captured_at)
    signals = (
        frozenset({ActionOutcomeSignal.PIXELS_CHANGED})
        if disposition is ActionOutcomeDisposition.OBSERVED
        else frozenset()
    )
    settled_observation_id = (
        None if disposition is ActionOutcomeDisposition.UNAVAILABLE else after.observation_id
    )
    outcome = ActionOutcome(
        outcome_id="outcome_1",
        authorization_id=authorized.authorization_id,
        receipt_id=receipt.receipt_id,
        action=authorized.action,
        source_observation_id=before.observation_id,
        settled_observation_id=settled_observation_id,
        disposition=disposition,
        signals=signals,
        reason="Deterministic outcome evidence.",
        retention=ArtifactRetention.RUN,
    )
    action_step = AgentStep(
        step_id="step_1",
        step_index=1,
        source_observation_id=before.observation_id,
        proposal=action_proposal,
        target_grounding=grounding,
        authorization=authorized,
        receipt=receipt,
        settlement_observation_ids=(after.observation_id,)
        if settled_observation_id is not None
        else (),
        settled_observation_id=settled_observation_id,
        outcome=outcome,
    )
    steps = [action_step]
    if terminal in {
        AgentTerminalReason.DONE_REQUESTED,
        AgentTerminalReason.CANNOT_PROGRESS,
    }:
        terminal_action = (
            ActionKind.DONE
            if terminal is AgentTerminalReason.DONE_REQUESTED
            else ActionKind.CANNOT_PROGRESS
        )
        terminal_proposal = proposal(
            terminal_action,
            proposal_id="proposal_2",
            source_observation_id=after.observation_id,
        )
        steps.append(
            AgentStep(
                step_id="step_2",
                step_index=2,
                source_observation_id=after.observation_id,
                proposal=terminal_proposal,
                terminal_reason=terminal,
            )
        )
    return AgentResult(
        terminal_reason=terminal,
        reason="Deterministic agent terminal.",
        final_observation_id=after.observation_id,
        usage=AgentUsage(
            steps=len(steps),
            planning_requests=len(steps),
            grounding_requests=1 if grounding is not None else 0,
            dispatched_actions=1,
        ),
        steps=tuple(steps),
    )


def _receipt(action: AuthorizedAction, dispatched_at: datetime) -> DispatchReceipt:
    """Dispatch at the source capture, so a settled timing measures the step and not the fixture."""
    return DispatchReceipt(
        dispatched_at=dispatched_at,
        receipt_id="receipt_1",
        authorization_id=action.authorization_id,
        proposal_id=action.proposal_id,
        action=action.action,
        disposition=DispatchDisposition.DISPATCHED,
        retention=action.retention,
    )
