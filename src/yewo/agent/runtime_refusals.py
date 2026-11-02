"""Record planner refusals and correctable device rejections."""

from __future__ import annotations

from yewo.agent.actuation import (
    ActionProposal,
    ActionRefusal,
    ActionRefusalCode,
    AuthorizedAction,
    DispatchDisposition,
    DispatchReceipt,
)
from yewo.agent.narration import narrate, narrate_no_progress, narrate_refusal
from yewo.agent.no_progress import ActionNoProgressTracker, record_refusal_with_facts
from yewo.agent.planning_evidence import PlannerDecisionMetadata
from yewo.agent.planning_history import (
    PlanningHistoryEntry,
    build_dispatch_rejection_history_entry,
    build_refusal_history_entry,
)
from yewo.agent.runtime_records import AgentStep, AgentStepStore
from yewo.grounding import GroundingEvidence
from yewo.observations import Observation


def record_action_refusal(
    step_store: AgentStepStore,
    steps: list[AgentStep],
    history: list[PlanningHistoryEntry],
    no_progress: ActionNoProgressTracker,
    step_index: int,
    observation: Observation,
    proposal: ActionProposal,
    decision: PlannerDecisionMetadata,
    refusal: ActionRefusal,
    target_grounding: GroundingEvidence | None = None,
    destination_grounding: GroundingEvidence | None = None,
) -> str | None:
    """Persist one pre-dispatch refusal and record established no-progress evidence."""
    step = step_store.record(
        step_index,
        observation,
        proposal,
        decision,
        target_grounding=target_grounding,
        destination_grounding=destination_grounding,
        refusal=refusal,
    )
    steps.append(step)
    narrate_refusal(step_index, refusal)
    history.append(build_refusal_history_entry(proposal, refusal, observation))
    if refusal.code is ActionRefusalCode.VERIFICATION_UNAVAILABLE:
        return None
    return record_refusal_with_facts(
        no_progress,
        proposal,
        observation,
        refusal,
        narrate_no_progress,
    )


def record_dispatch_rejection(
    step_store: AgentStepStore,
    steps: list[AgentStep],
    history: list[PlanningHistoryEntry],
    no_progress: ActionNoProgressTracker,
    step_index: int,
    observation: Observation,
    proposal: ActionProposal,
    decision: PlannerDecisionMetadata,
    target_grounding: GroundingEvidence | None,
    destination_grounding: GroundingEvidence | None,
    authorization: AuthorizedAction,
    receipt: DispatchReceipt,
) -> str | None:
    """Persist one correctable rejection that occurred before device mutation."""
    step = step_store.record(
        step_index,
        observation,
        proposal,
        decision,
        target_grounding=target_grounding,
        destination_grounding=destination_grounding,
        authorization=authorization,
        receipt=receipt,
    )
    steps.append(step)
    reason = receipt.reason or "The device rejected correctable action input."
    narrate(f"Step {step_index} dispatch rejected: {reason}")
    history.append(
        build_dispatch_rejection_history_entry(
            proposal,
            receipt,
            target_grounding,
            observation,
        )
    )
    refusal = ActionRefusal(
        proposal_id=proposal.proposal_id,
        code={
            DispatchDisposition.INPUT_REJECTED: ActionRefusalCode.INVALID_INPUT,
            DispatchDisposition.TARGET_REJECTED: ActionRefusalCode.INVALID_GROUNDING,
        }[receipt.disposition],
        reason=reason,
    )
    return record_refusal_with_facts(
        no_progress,
        proposal,
        observation,
        refusal,
        narrate_no_progress,
    )
