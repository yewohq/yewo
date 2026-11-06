"""Interpret live history for one pending compiled procedure action."""

from yewo.agent.planning.history import PlanningHistoryEntry


def find_action_feedback(
    history: tuple[PlanningHistoryEntry, ...],
    proposal_id: str,
) -> PlanningHistoryEntry | None:
    """Return the latest feedback for one live compiled proposal."""
    return next(
        (entry for entry in reversed(history) if entry.proposal.proposal_id == proposal_id),
        None,
    )


def application_recovered(feedback: PlanningHistoryEntry) -> bool:
    """Return whether recovery invalidated the pending procedure position."""
    failure = feedback.application_failure
    return failure is not None and failure.recovery_succeeded
