"""Deterministic proposal checks that do not require application interpretation."""

from __future__ import annotations

from yewo.agent.actuation import (
    ActionKind,
    ActionProposal,
    ActionRefusal,
    ActionRefusalCode,
    LiteralText,
)
from yewo.agent.screen_analysis import (
    CurrentScreenAnalysis,
    ScreenElement,
)


def proposal_coherence_refusal(
    proposal: ActionProposal,
    analysis: CurrentScreenAnalysis | None,
) -> ActionRefusal | None:
    """Reject contradictions established by the proposal and shared screen map."""
    if analysis is None:
        return None
    target = _target_element(proposal, analysis)
    if target is None:
        return None
    if proposal.action is ActionKind.TYPE and (
        isinstance(proposal.text, LiteralText)
        and target.state is not None
        and _normalized(proposal.text.value) == _normalized(target.state)
    ):
        return _refusal(
            proposal,
            f"The target {target.name!r} already contains the proposed value.",
        )
    return None


def _target_element(
    proposal: ActionProposal,
    analysis: CurrentScreenAnalysis,
) -> ScreenElement | None:
    if proposal.target_element_ref is None:
        return None
    return next(
        (item for item in analysis.elements if item.element_ref == proposal.target_element_ref),
        None,
    )


def _normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def _refusal(proposal: ActionProposal, reason: str) -> ActionRefusal:
    return ActionRefusal(
        proposal_id=proposal.proposal_id,
        code=ActionRefusalCode.INVALID_CONSTRAINT,
        reason=reason,
    )


__all__ = ["proposal_coherence_refusal"]
