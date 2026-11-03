"""Observation-reference ownership for retained YewoAgent memory."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from yewo.artifacts import RecordIdentifier

if TYPE_CHECKING:
    from yewo.agent.memory.records import AcceptedTransition, UnsuccessfulAttempt


def referenced_memory_observation_ids(
    transitions: Sequence[AcceptedTransition],
    unsuccessful_attempts: Sequence[UnsuccessfulAttempt],
) -> set[RecordIdentifier]:
    """Return observation variants pinned by retained memory evidence."""
    referenced = {
        *(
            observation_id
            for transition in transitions
            for observation_id in (
                transition.source_observation_id,
                transition.destination_observation_id,
            )
        ),
        *(attempt.source_observation_id for attempt in unsuccessful_attempts),
    }
    for transition in transitions:
        for grounding in (transition.target_grounding, transition.destination_grounding):
            if grounding is not None:
                referenced.add(
                    grounding.source_observation_id or transition.source_observation_id
                )
    return referenced


def validate_retained_observations(observation_ids: list[RecordIdentifier]) -> None:
    """Require every retained observation to belong to one logical screen."""
    if len(observation_ids) != len(set(observation_ids)):
        raise ValueError("A memory observation must belong to one logical screen.")
