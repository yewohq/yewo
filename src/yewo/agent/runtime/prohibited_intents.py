"""Project no-progress attempts into bounded planner prohibitions."""

from __future__ import annotations

from collections.abc import Callable

from yewo.agent.actuation import ActionProposal
from yewo.agent.runtime.interaction_intent import (
    AmbiguousInteractionState,
    interaction_intent_key,
)
from yewo.observations import Observation


def collect_prohibited_intents(
    observation: Observation,
    matches: Callable[[Observation, Observation], bool],
    attempts: list[tuple[ActionProposal, Observation]],
    refusals: list[tuple[ActionProposal, str, Observation, int]],
    exhausted_transitions: list[tuple[ActionProposal, Observation]],
    focus_prohibitions: list[tuple[ActionProposal, Observation]],
    ambiguous: AmbiguousInteractionState,
    route_keys: tuple[str, ...],
    maximum: int,
) -> tuple[str, ...]:
    """Return one deduplicated prohibition list for the current logical screen."""
    proposals = (
        *(proposal for proposal, screen in attempts if matches(observation, screen)),
        *(
            proposal
            for proposal, unused_code, screen, unused_count in refusals
            if matches(observation, screen)
        ),
        *(proposal for proposal, screen in exhausted_transitions if matches(observation, screen)),
        *(proposal for proposal, screen in focus_prohibitions if matches(observation, screen)),
        *ambiguous.prohibited_proposals(observation),
    )
    keys = tuple(
        dict.fromkeys((*route_keys, *(interaction_intent_key(item) for item in proposals)))
    )
    return keys[-maximum:]


__all__ = ["collect_prohibited_intents"]
