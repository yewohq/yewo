"""Identify an interaction without interpreting application vocabulary."""

from __future__ import annotations

from yewo.agent.actuation import ActionProposal
from yewo.agent.memory.records import ScreenComparisonPolicy
from yewo.agent.memory.screens import observations_match
from yewo.observations import Observation


class AmbiguousInteractionState:
    """Retain ambiguous interaction families until observed evidence clears them."""

    def __init__(self, comparison: ScreenComparisonPolicy) -> None:
        self._comparison = comparison
        self._attempts: list[tuple[ActionProposal, Observation]] = []

    def reset(self) -> None:
        self._attempts = []

    def record(self, proposal: ActionProposal, observation: Observation) -> int:
        self._attempts.append((proposal, observation))
        return self.count(proposal, observation)

    def count(self, proposal: ActionProposal, observation: Observation) -> int:
        return matching_interaction_count(
            proposal,
            observation,
            self._attempts,
            self._comparison,
        )

    def clear_observed(self, proposal: ActionProposal, observation: Observation) -> None:
        self._attempts = [
            (prior, screen)
            for prior, screen in self._attempts
            if not (
                same_interaction_family(proposal, prior)
                and observations_match(observation, screen, self._comparison)
            )
        ]

    def prohibited_proposals(self, observation: Observation) -> tuple[ActionProposal, ...]:
        return tuple(
            proposal
            for proposal, screen in self._attempts
            if observations_match(observation, screen, self._comparison)
            if self.count(proposal, observation) >= 2
        )


def interaction_intent_key(proposal: ActionProposal) -> str:
    """Return one stable key from the proposal's structured action fields."""
    target = (
        normalize_target_expression(proposal.target_expression)
        if proposal.target_expression is not None
        else (
            f"ref:{proposal.target_element_ref}"
            if proposal.target_element_ref is not None
            else ""
        )
    )
    destination = (
        normalize_target_expression(proposal.destination_expression)
        if proposal.destination_expression is not None
        else (
            f"ref:{proposal.destination_element_ref}"
            if proposal.destination_element_ref is not None
            else ""
        )
    )
    fields = (
        proposal.action.value,
        target,
        destination,
        proposal.direction.value if proposal.direction is not None else "",
        proposal.distance.value if proposal.distance is not None else "",
        str(proposal.duration_seconds) if proposal.duration_seconds is not None else "",
    )
    return "|".join(fields)


def same_interaction_family(first: ActionProposal, second: ActionProposal) -> bool:
    """Return whether two proposals contain the same structured interaction."""
    return interaction_intent_key(first) == interaction_intent_key(second)


def matching_interaction_count(
    proposal: ActionProposal,
    observation: Observation,
    attempts: list[tuple[ActionProposal, Observation]],
    comparison: ScreenComparisonPolicy,
) -> int:
    """Count one interaction family on equivalent logical screens."""
    return sum(
        same_interaction_family(proposal, prior)
        and observations_match(observation, screen, comparison)
        for prior, screen in attempts
    )


def normalize_target_expression(expression: str | None) -> str:
    if expression is None:
        return ""
    return " ".join(expression.casefold().split())


__all__ = [
    "AmbiguousInteractionState",
    "interaction_intent_key",
    "matching_interaction_count",
    "same_interaction_family",
]
