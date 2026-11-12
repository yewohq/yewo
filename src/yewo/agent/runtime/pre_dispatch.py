"""Compose deterministic pre-dispatch policy with optional model verification."""

from __future__ import annotations

from dataclasses import dataclass

from yewo.agent.actuation import (
    ACTION_DEFINITIONS,
    ActionBudget,
    ActionHandling,
    ActionPolicy,
    ActionProposal,
    ActionRefusal,
)
from yewo.agent.actuation.proposal_verifier import ProposalVerifier
from yewo.agent.planning.evidence import PlannerDecisionSource
from yewo.agent.planning.records import PlanningRequest
from yewo.agent.recovery.runtime import RecoveryRunState
from yewo.agent.runtime.progress import RunProgressLedger, gate_before_grounding
from yewo.agent.runtime.proposal_coherence import proposal_coherence_refusal
from yewo.agent.runtime.records import AgentModePolicy, AgentTerminalReason, proposal_terminal
from yewo.agent.runtime.verification import PreDispatchVerification, verify_discovery_action


@dataclass(frozen=True)
class PreparedAction:
    """Deterministic gate and optional discovery verification result."""

    proposal: ActionProposal
    refusal: ActionRefusal | None
    verification_request_count: int = 0
    budget_reason: str | None = None
    verification: PreDispatchVerification = PreDispatchVerification()


@dataclass(frozen=True)
class PreDispatchTerminal:
    """A planner terminal or an exhausted device-action budget."""

    terminal_reason: AgentTerminalReason
    reason: str


def pre_dispatch_terminal(
    proposal: ActionProposal,
    dispatched_actions: int,
    credited_actions: int,
    maximum_actions: int,
) -> PreDispatchTerminal | None:
    """Resolve terminal proposals and the final action-budget gate together."""
    terminal = proposal_terminal(proposal)
    if terminal is not None:
        return PreDispatchTerminal(
            terminal,
            proposal.reason or "The planner requested a terminal.",
        )
    if dispatched_actions - credited_actions >= maximum_actions:
        return PreDispatchTerminal(
            AgentTerminalReason.BUDGET_EXHAUSTED,
            "No device-action budget remains.",
        )
    return None


def prepare_action_for_grounding(
    progress_ledger: RunProgressLedger,
    action_policy: ActionPolicy,
    action_budget: ActionBudget,
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
) -> PreparedAction:
    """Apply deterministic policy before spending a discovery-verifier request."""
    definition = ACTION_DEFINITIONS[proposal.action]
    if definition.handling is ActionHandling.TERMINAL:
        verification = verify_discovery_action(
            verifier,
            discovery_mode,
            decision_source,
            planning_source,
            planning_request,
            proposal,
            mode_policy,
            recovery,
            completed_step_count,
            started_at,
            planning_requests,
            grounding_requests,
            now,
        )
        progress_refusal = progress_ledger.record_verification(
            verification,
            planning_request.observation,
            proposal,
        )
        return PreparedAction(
            proposal,
            verification.refusal or progress_refusal,
            verification.request_count,
            verification.budget_reason,
            verification,
        )
    proposal, refusal = gate_before_grounding(
        progress_ledger,
        action_policy,
        proposal,
        planning_request.observation,
        action_budget,
        decision_source,
    )
    if refusal is not None:
        return PreparedAction(proposal, refusal)
    coherence_refusal = proposal_coherence_refusal(
        proposal,
        planning_request.current_screen_analysis,
    )
    if coherence_refusal is not None:
        return PreparedAction(proposal, coherence_refusal)
    verification = verify_discovery_action(
        verifier,
        discovery_mode,
        decision_source,
        planning_source,
        planning_request,
        proposal,
        mode_policy,
        recovery,
        completed_step_count,
        started_at,
        planning_requests,
        grounding_requests,
        now,
    )
    progress_refusal = progress_ledger.record_verification(
        verification,
        planning_request.observation,
        proposal,
    )
    return PreparedAction(
        proposal,
        verification.refusal or progress_refusal,
        verification.request_count,
        verification.budget_reason,
        verification,
    )
