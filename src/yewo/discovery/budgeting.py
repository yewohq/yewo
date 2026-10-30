"""Pure independent-budget accounting for bounded discovery."""

from __future__ import annotations

from yewo.agent.actuation.safety import ActionAuthorization
from yewo.discovery.contracts import (
    DiscoveryBlocker,
    DiscoveryBudgetDimension,
    DiscoveryBudgetUsage,
    DiscoveryRun,
    SideEffectClass,
    TransitionAttemptUsage,
)
from yewo.discovery.runtime import (
    DiscoveryActionCandidate,
    DiscoveryProposalReceipt,
    DiscoveryProposer,
)

AuthorizedFrontier = dict[
    str, tuple[DiscoveryActionCandidate, ActionAuthorization]
]


def can_spend(
    run: DiscoveryRun, dimension: DiscoveryBudgetDimension, amount: int
) -> bool:
    current = getattr(run.usage, dimension.value)
    limit = getattr(run.budgets, dimension.value)
    return current + amount <= limit


def spend(
    usage: DiscoveryBudgetUsage,
    dimension: DiscoveryBudgetDimension,
    amount: int,
) -> DiscoveryBudgetUsage:
    current = getattr(usage, dimension.value)
    return usage.model_copy(update={dimension.value: current + amount})


def proposal_budget(
    run: DiscoveryRun, proposer: DiscoveryProposer
) -> DiscoveryBudgetDimension | None:
    if not can_spend(
        run,
        DiscoveryBudgetDimension.MODEL_PROPOSALS,
        proposer.model_calls_per_proposal,
    ):
        return DiscoveryBudgetDimension.MODEL_PROPOSALS
    if not can_spend(
        run,
        DiscoveryBudgetDimension.COST_MICRO_USD,
        proposer.maximum_cost_micro_usd,
    ):
        return DiscoveryBudgetDimension.COST_MICRO_USD
    return None


def record_proposal(
    run: DiscoveryRun,
    proposal: DiscoveryProposalReceipt,
    proposer: DiscoveryProposer,
) -> tuple[DiscoveryBudgetUsage, DiscoveryBlocker | None]:
    if (
        proposal.model_calls < 0
        or proposal.model_calls > proposer.model_calls_per_proposal
        or proposal.cost_micro_usd < 0
        or proposal.cost_micro_usd > proposer.maximum_cost_micro_usd
    ):
        return run.usage, DiscoveryBlocker.INFRASTRUCTURE
    usage = spend(
        run.usage,
        DiscoveryBudgetDimension.MODEL_PROPOSALS,
        proposal.model_calls,
    )
    usage = spend(
        usage,
        DiscoveryBudgetDimension.COST_MICRO_USD,
        proposal.cost_micro_usd,
    )
    return usage, None


def charge_unreported_proposal_spend(
    run: DiscoveryRun, proposer: DiscoveryProposer
) -> tuple[
    DiscoveryBudgetUsage | None,
    DiscoveryBlocker | None,
    DiscoveryBudgetDimension | None,
]:
    """Validate and charge spend that no proposal receipt carried."""
    model_calls, micro_usd = proposer.drain_unreported_spend()
    if not model_calls and not micro_usd:
        return None, None, None
    if (
        model_calls < 0
        or model_calls > proposer.model_calls_per_proposal
        or micro_usd < 0
        or micro_usd > proposer.maximum_cost_micro_usd
    ):
        return None, DiscoveryBlocker.INFRASTRUCTURE, None
    if not can_spend(
        run, DiscoveryBudgetDimension.MODEL_PROPOSALS, model_calls
    ):
        return None, None, DiscoveryBudgetDimension.MODEL_PROPOSALS
    if not can_spend(run, DiscoveryBudgetDimension.COST_MICRO_USD, micro_usd):
        return None, None, DiscoveryBudgetDimension.COST_MICRO_USD
    usage = spend(run.usage, DiscoveryBudgetDimension.MODEL_PROPOSALS, model_calls)
    return spend(usage, DiscoveryBudgetDimension.COST_MICRO_USD, micro_usd), None, None


def pre_proposal_action_budget(run: DiscoveryRun) -> DiscoveryBudgetDimension | None:
    if not can_spend(run, DiscoveryBudgetDimension.ACTIONS, 1):
        return DiscoveryBudgetDimension.ACTIONS
    if not can_spend(run, DiscoveryBudgetDimension.OBSERVATIONS, 1):
        return DiscoveryBudgetDimension.OBSERVATIONS
    return None


def budget_authorized(
    run: DiscoveryRun,
    authorized: AuthorizedFrontier,
) -> tuple[AuthorizedFrontier, DiscoveryBudgetDimension | None]:
    exhausted = pre_proposal_action_budget(run)
    if exhausted is not None:
        return authorized, exhausted
    if can_spend(run, DiscoveryBudgetDimension.SIDE_EFFECT_ACTIONS, 1):
        return authorized, None
    read_only = {
        frontier_id: value
        for frontier_id, value in authorized.items()
        if value[0].side_effect is SideEffectClass.READ_ONLY
    }
    if read_only:
        return read_only, None
    return authorized, DiscoveryBudgetDimension.SIDE_EFFECT_ACTIONS


def action_budget(
    run: DiscoveryRun, candidate: DiscoveryActionCandidate
) -> DiscoveryBudgetDimension | None:
    if not can_spend(run, DiscoveryBudgetDimension.ACTIONS, 1):
        return DiscoveryBudgetDimension.ACTIONS
    if (
        candidate.side_effect is not SideEffectClass.READ_ONLY
        and not can_spend(run, DiscoveryBudgetDimension.SIDE_EFFECT_ACTIONS, 1)
    ):
        return DiscoveryBudgetDimension.SIDE_EFFECT_ACTIONS
    if not can_spend(run, DiscoveryBudgetDimension.OBSERVATIONS, 1):
        return DiscoveryBudgetDimension.OBSERVATIONS
    if not run.usage.can_attempt_transition(
        candidate.attempt_identity, run.budgets
    ):
        return DiscoveryBudgetDimension.ACTIONS
    return None


def record_action_spend(
    run: DiscoveryRun, candidate: DiscoveryActionCandidate
) -> DiscoveryBudgetUsage:
    usage = spend(run.usage, DiscoveryBudgetDimension.ACTIONS, 1)
    if candidate.side_effect is not SideEffectClass.READ_ONLY:
        usage = spend(usage, DiscoveryBudgetDimension.SIDE_EFFECT_ACTIONS, 1)
    attempts = {
        item.transition_identity: item.attempts
        for item in usage.transition_attempts
    }
    attempts[candidate.attempt_identity] = attempts.get(candidate.attempt_identity, 0) + 1
    return usage.model_copy(
        update={
            "transition_attempts": tuple(
                TransitionAttemptUsage(transition_identity=identity, attempts=count)
                for identity, count in sorted(attempts.items())
            )
        }
    )
