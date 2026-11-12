"""Ground one proposal and report how the loop should answer a failure to."""

from __future__ import annotations

from dataclasses import dataclass

from yewo.agent.actuation import ActionProposal
from yewo.agent.narration import narrate
from yewo.agent.planning.evidence import PlannerDecisionMetadata
from yewo.agent.planning.planner import PlanningHistoryEntry
from yewo.agent.recovery.runtime import RecoveryRunState
from yewo.agent.runtime import grounding
from yewo.agent.runtime.progress import RunProgressLedger
from yewo.agent.runtime.records import (
    AgentModePolicy,
    AgentStep,
    AgentStepStore,
    AgentTerminalReason,
    grounding_budget_reason,
)
from yewo.agent.runtime.refusals import record_grounding_refusal
from yewo.grounding import GroundingEvidence, GroundingRequest, VisualGrounder
from yewo.grounding.records import GroundingInputError
from yewo.grounding.service import GroundingEvidenceError
from yewo.observations import Observation


@dataclass(frozen=True)
class ResolvedGrounding:
    """Grounding evidence, or how the loop should answer a failure to produce it."""

    target: GroundingEvidence | None = None
    destination: GroundingEvidence | None = None
    terminal: tuple[AgentTerminalReason, str] | None = None
    replan: bool = False


def resolve_proposal_grounding_phase(
    grounder: VisualGrounder,
    step_store: AgentStepStore,
    progress_ledger: RunProgressLedger,
    recovery: RecoveryRunState,
    mode_policy: AgentModePolicy,
    target_request: GroundingRequest | None,
    destination_request: GroundingRequest | None,
    steps: list[AgentStep],
    history: list[PlanningHistoryEntry],
    budget_time: float,
    step_index: int,
    observation: Observation,
    proposal: ActionProposal,
    decision: PlannerDecisionMetadata,
    *,
    planning_requests: int,
    grounding_requests: int,
    dispatched_actions: int,
) -> ResolvedGrounding:
    """Ground one proposal, or report how the loop should answer a failure to."""
    target: GroundingEvidence | None = None
    destination: GroundingEvidence | None = None
    try:
        budget = grounding_phase_budget_reason(
            grounder,
            recovery,
            mode_policy,
            target_request,
            destination_request,
            len(steps),
            budget_time,
            planning_requests,
            grounding_requests,
        )
        if budget is not None:
            steps.append(
                step_store.record(
                    step_index,
                    observation,
                    proposal,
                    decision,
                    AgentTerminalReason.BUDGET_EXHAUSTED,
                )
            )
            return ResolvedGrounding(terminal=(AgentTerminalReason.BUDGET_EXHAUSTED, budget))
        target, destination = grounding.resolve_proposal_grounding(
            grounder,
            target_request,
            destination_request,
        )
    except (GroundingEvidenceError, GroundingInputError) as error:
        reason = record_grounding_refusal(
            step_store,
            steps,
            history,
            progress_ledger,
            step_index,
            observation,
            proposal,
            decision,
            error,
            planning_requests=planning_requests,
            grounding_requests=grounding_requests,
            dispatched_actions=dispatched_actions,
        )
        if reason is not None:
            return ResolvedGrounding(terminal=(AgentTerminalReason.CANNOT_PROGRESS, reason))
        return ResolvedGrounding(replan=True)
    except BaseException as error:
        narrate(f"Step {step_index} grounding failed: {type(error).__name__}.")
        step_store.record_partial_or_raise(
            step_index,
            observation,
            proposal,
            decision,
            target,
            destination,
            None,
            None,
            AgentTerminalReason.RUNTIME_FAILURE,
            steps,
            error,
        )
        raise
    return ResolvedGrounding(target=target, destination=destination)


def grounding_phase_budget_reason(
    grounder: VisualGrounder,
    recovery: RecoveryRunState,
    mode_policy: AgentModePolicy,
    target_request: GroundingRequest | None,
    destination_request: GroundingRequest | None,
    completed_step_count: int,
    budget_time: float,
    planning_requests: int,
    grounding_requests: int,
) -> str | None:
    """Reserve the grounding calls this proposal needs before any of them is made."""
    reservation = sum(
        grounder.required_model_requests(request)
        for request in (target_request, destination_request)
        if request is not None
    )
    was_recovery = recovery.is_active
    credit = recovery.credit(completed_step_count, budget_time)
    return grounding_budget_reason(
        mode_policy,
        planning_requests - credit.planning_requests - credit.analysis_requests,
        grounding_requests - credit.grounding_requests,
        0 if was_recovery else reservation,
    )
