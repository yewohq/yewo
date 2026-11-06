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
from yewo.agent.narration import narrate, narrate_no_progress_facts, narrate_refusal
from yewo.agent.planning.evidence import PlannerDecisionMetadata
from yewo.agent.planning.history import (
    PlanningHistoryEntry,
    build_dispatch_rejection_history_entry,
    build_refusal_history_entry,
)
from yewo.agent.runtime.failure import (
    ModelServiceUnavailableError,
    persist_model_service_failure,
)
from yewo.agent.runtime.records import AgentStep, AgentStepStore
from yewo.agent.runtime.progress import RunProgressLedger
from yewo.grounding import GroundingEvidence
from yewo.observations import Observation


def record_action_refusal(
    step_store: AgentStepStore,
    steps: list[AgentStep],
    history: list[PlanningHistoryEntry],
    progress_ledger: RunProgressLedger,
    step_index: int,
    observation: Observation,
    proposal: ActionProposal,
    decision: PlannerDecisionMetadata,
    refusal: ActionRefusal,
    target_grounding: GroundingEvidence | None = None,
    destination_grounding: GroundingEvidence | None = None,
    *,
    planning_requests: int,
    grounding_requests: int,
    dispatched_actions: int,
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
    progress_facts, terminal_reason = progress_ledger.record_refusal(
        proposal,
        observation,
        refusal,
        decision.source,
    )
    narrate_no_progress_facts(progress_facts)
    if terminal_reason is not None and progress_ledger.refusal_deadlock_is_transient:
        summary = persist_model_service_failure(
            step_store.root,
            "A required model service remained unavailable. " + terminal_reason,
            observation.observation_id,
            steps,
            planning_requests,
            grounding_requests,
            dispatched_actions,
        )
        raise ModelServiceUnavailableError(summary)
    return terminal_reason


def record_dispatch_rejection(
    step_store: AgentStepStore,
    steps: list[AgentStep],
    history: list[PlanningHistoryEntry],
    progress_ledger: RunProgressLedger,
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
    progress_facts, terminal_reason = progress_ledger.record_refusal(
        proposal,
        observation,
        refusal,
        decision.source,
    )
    narrate_no_progress_facts(progress_facts)
    return terminal_reason
