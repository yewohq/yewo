"""Derive workflow-stage advancement from trusted confirmation outcomes."""

from __future__ import annotations

from yewo.agent.actuation import (
    ActionKind,
    ActionOutcome,
    ActionOutcomeDisposition,
    ActionProposal,
)
from yewo.agent.screen_analysis import (
    CurrentScreenAnalysis,
    ScreenSemanticRole,
    ScreenWorkflowRole,
)
from yewo.observations import Observation


def confirmed_stage(
    proposal: ActionProposal,
    work_item: str | None,
    analysis: CurrentScreenAnalysis | None,
    before: Observation,
    after: Observation,
    outcome: ActionOutcome,
) -> str | None:
    """Describe an observed confirmation without claiming whole-task completion."""
    if (
        proposal.action is not ActionKind.TAP
        or proposal.target_element_ref is None
        or analysis is None
        or outcome.disposition is not ActionOutcomeDisposition.OBSERVED
        or before.screen_fingerprint.value == after.screen_fingerprint.value
    ):
        return None
    target = next(
        (
            element
            for element in analysis.elements
            if element.element_ref == proposal.target_element_ref
        ),
        None,
    )
    if target is None or target.kind.semantic is not ScreenSemanticRole.BUTTON:
        return None
    if target.workflow_role is not ScreenWorkflowRole.ADVANCE_STAGE:
        return None
    work = f" for {work_item!r}" if work_item is not None else ""
    return (
        f"The {target.name!r} confirmation advanced one workflow stage{work}. "
        "Do not reopen that accepted stage unless newer evidence shows rejection."
    )


__all__ = ["confirmed_stage"]
