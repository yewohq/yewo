"""Close discovery frontier entries and unresolved goals truthfully."""

from typing import Literal

from yewo.discovery.contracts import (
    CapabilityState,
    FrontierState,
    JourneyGoal,
    RelevantFrontierEntry,
)

ClosableFrontierState = Literal[
    FrontierState.DECLINED,
    FrontierState.ATTEMPTED,
    FrontierState.VERIFIED,
    FrontierState.POLICY_REFUSED,
    FrontierState.NAVIGATION_UNAVAILABLE,
]

WindowClosureState = Literal[
    FrontierState.DECLINED,
    FrontierState.NAVIGATION_UNAVAILABLE,
]


def close_frontier_entry(
    frontier: tuple[RelevantFrontierEntry, ...],
    frontier_id: str,
    state: ClosableFrontierState,
    transition_identity: str | None = None,
) -> tuple[RelevantFrontierEntry, ...]:
    return tuple(
        entry.close(state, transition_identity)
        if entry.frontier_id == frontier_id
        else entry
        for entry in frontier
    )


def close_available_frontier_ids(
    frontier: tuple[RelevantFrontierEntry, ...],
    frontier_ids: set[str],
    state: WindowClosureState,
) -> tuple[RelevantFrontierEntry, ...]:
    return tuple(
        entry.close(state)
        if entry.frontier_id in frontier_ids
        and entry.state is FrontierState.AVAILABLE
        else entry
        for entry in frontier
    )


def close_all_available_frontier(
    frontier: tuple[RelevantFrontierEntry, ...],
    state: WindowClosureState,
) -> tuple[RelevantFrontierEntry, ...]:
    return tuple(
        entry.close(state) if entry.state is FrontierState.AVAILABLE else entry
        for entry in frontier
    )


def close_unresolved_goals(
    goals: tuple[JourneyGoal, ...],
    state: CapabilityState,
) -> tuple[JourneyGoal, ...]:
    closed = []
    for goal in goals:
        if goal.is_terminal:
            closed.append(goal)
            continue
        if goal.state is CapabilityState.PENDING and state not in (
            CapabilityState.NOT_EVALUATED,
            CapabilityState.BLOCKED_BY_POLICY,
            CapabilityState.BLOCKED_BY_SETUP,
        ):
            goal = goal.activate()
        closed.append(goal.close(state))
    return tuple(closed)
