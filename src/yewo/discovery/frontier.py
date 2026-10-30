"""Which offered controls become work, and which journey goals that opens.

The controller owns the loop and the decision to act. This owns the
bookkeeping one observation causes: what a batch is allowed to offer, which
offers are new, and which pending goals an offer activates.

Here rather than in the controller. This is the one place a runtime's offers
are checked against the observation that produced them. A runtime offering a
control that observation did not report is refused. So no later step can act
on a target the run has no evidence for.
"""

from __future__ import annotations

from dataclasses import dataclass

from yewo.discovery.contracts import (
    CapabilityState,
    DiscoveryObservation,
    FrontierState,
    JourneyGoal,
    RelevantFrontierEntry,
)
from yewo.discovery.runtime import DiscoveryActionCandidate


@dataclass(frozen=True)
class FrontierExtension:
    """The goals and frontier one observation leaves behind."""

    goals: tuple[JourneyGoal, ...]
    frontier: tuple[RelevantFrontierEntry, ...]


def extend_frontier(
    observation: DiscoveryObservation,
    candidates: tuple[DiscoveryActionCandidate, ...],
    goals: tuple[JourneyGoal, ...],
    frontier: tuple[RelevantFrontierEntry, ...],
) -> FrontierExtension:
    """Add this observation's new offers, activating the goals they serve.

    Args:
        observation: The accepted observation these offers came from.
        candidates: What the runtime offered on that observation.
        goals: The run's journey goals, in order.
        frontier: The run's frontier so far.

    Returns:
        The goals and the frontier after this observation.

    Raises:
        ValueError: An offer names another observation or another screen
            state. Or it names a control the observation did not report as
            actionable. Or it names a goal this run does not hold.
    """
    actionable = {
        control.control_identity
        for control in observation.controls
        if control.actionable
    }
    goals_by_id = {goal.goal_id: goal for goal in goals}
    settled_goals = list(goals)
    entries = list(frontier)
    offered = {entry.frontier_id for entry in entries}
    for candidate in candidates:
        _validate_offer(candidate, observation, actionable, goals_by_id)
        unresolved = tuple(
            goal_id
            for goal_id in candidate.relevant_goal_ids
            if not goals_by_id[goal_id].is_terminal
        )
        if not unresolved or candidate.frontier_id in offered:
            continue
        for goal_id in unresolved:
            goal = goals_by_id[goal_id]
            if goal.state is CapabilityState.PENDING:
                activated = goal.activate()
                goals_by_id[goal_id] = activated
                settled_goals[settled_goals.index(goal)] = activated
        entries.append(
            RelevantFrontierEntry(
                frontier_id=candidate.frontier_id,
                source_observation_id=candidate.source_observation_id,
                control_identity=candidate.control_identity,
                relevant_goal_ids=unresolved,
            )
        )
        offered.add(candidate.frontier_id)
    return FrontierExtension(tuple(settled_goals), tuple(entries))


def _validate_offer(
    candidate: DiscoveryActionCandidate,
    observation: DiscoveryObservation,
    actionable: set[str],
    goals_by_id: dict[str, JourneyGoal],
) -> None:
    if candidate.source_observation_id != observation.observation_id:
        raise ValueError("A frontier candidate requires its exact source observation.")
    if candidate.source_screen_state_id != observation.screen_state_id:
        raise ValueError("A frontier candidate requires its exact source state.")
    if candidate.control_identity not in actionable:
        raise ValueError("A frontier candidate requires an actionable observed control.")
    if not candidate.relevant_goal_ids or not set(
        candidate.relevant_goal_ids
    ).issubset(goals_by_id):
        raise ValueError("A frontier candidate requires known journey goals.")


def retire_settled_offers(
    frontier: tuple[RelevantFrontierEntry, ...],
    goals: tuple[JourneyGoal, ...],
) -> tuple[RelevantFrontierEntry, ...]:
    """Close offers whose every goal has settled, so none is left waiting.

    An offer is recorded while its goal is open. The goal can settle from the
    very observation that produced the offer. A run may not hold available work
    whose every goal is already answered. Closed as attempted, which is the
    state a run uses when it stops pursuing work it still had.
    """
    settled = {goal.goal_id for goal in goals if goal.is_terminal}
    return tuple(
        entry.close(FrontierState.ATTEMPTED)
        if entry.state is FrontierState.AVAILABLE
        and set(entry.relevant_goal_ids).issubset(settled)
        else entry
        for entry in frontier
    )
