"""Settle journey goals from accepted capability evidence.

A goal closes from a receipt, never from a claim. The controller decides when
to ask. This decides what one answer is allowed to change.

Closing a goal also retires frontier work that serves only settled goals.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from yewo.discovery.contracts import (
    CapabilityState,
    JourneyGoal,
    RelevantFrontierEntry,
)
from yewo.discovery.frontier import retire_settled_offers
from yewo.discovery.runtime import CapabilityEvidenceReceipt


@dataclass(frozen=True)
class GoalSettlement:
    """The goals and the frontier one set of receipts leaves behind."""

    goals: tuple[JourneyGoal, ...]
    frontier: tuple[RelevantFrontierEntry, ...]


def settle_goals(
    receipts: Sequence[CapabilityEvidenceReceipt],
    goals: tuple[JourneyGoal, ...],
    frontier: tuple[RelevantFrontierEntry, ...],
    observation_id: str,
    transition_identity: str | None,
) -> GoalSettlement:
    """Close every goal these receipts answer, and retire the offers they spent.

    Args:
        receipts: What the runtime reported about the observation it reached.
        goals: The run's journey goals, in order.
        frontier: The run's frontier so far.
        observation_id: The observation the controller accepted.
        transition_identity: The transition the controller accepted, if any.

    Returns:
        The goals and the frontier after these receipts.

    Raises:
        ValueError: A receipt names another observation, an unknown goal, or a
            transition the controller did not accept.
    """
    settled_goals = list(goals)
    goals_by_id = {goal.goal_id: goal for goal in settled_goals}
    goal_index_by_id = {
        goal.goal_id: position for position, goal in enumerate(settled_goals)
    }
    for receipt in receipts:
        if receipt.observation_id != observation_id:
            raise ValueError("Capability evidence requires its exact observation.")
        goal = goals_by_id.get(receipt.goal_id)
        if goal is None:
            raise ValueError("Capability evidence names an unknown goal.")
        if goal.is_terminal:
            continue
        state = _capability_state_for_receipt(receipt, transition_identity)
        if goal.state is CapabilityState.PENDING:
            goal = goal.activate()
        closed = goal.close(state, (observation_id,))
        goals_by_id[closed.goal_id] = closed
        settled_goals[goal_index_by_id[closed.goal_id]] = closed
    updated_goals = tuple(settled_goals)
    return GoalSettlement(
        updated_goals, retire_settled_offers(frontier, updated_goals)
    )


def _capability_state_for_receipt(
    receipt: CapabilityEvidenceReceipt, transition_identity: str | None
) -> CapabilityState:
    if receipt.kind == "route_verified":
        if transition_identity is None or receipt.transition_identity != transition_identity:
            raise ValueError("Route capability evidence requires its accepted transition.")
        return CapabilityState.ROUTE_VERIFIED
    if receipt.transition_identity is not None:
        raise ValueError("Observed capability evidence cannot claim a transition.")
    return CapabilityState.REACHED
