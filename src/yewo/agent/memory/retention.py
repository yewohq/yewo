"""Bounded transition retention for reusable YewoAgent memory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from yewo.agent.narration import narrate_memory_eviction

if TYPE_CHECKING:
    from yewo.agent.memory.records import AcceptedTransition


def retain_transitions(
    transitions: list[AcceptedTransition],
    maximum: int,
) -> tuple[AcceptedTransition, ...]:
    """Keep the newest accepted transitions within the retention bound."""
    retained = tuple(transitions[-maximum:] if maximum else ())
    retained_ids = {item.transition_id for item in retained}
    narrate_memory_eviction(
        "transition",
        tuple(
            item.transition_id
            for item in transitions
            if item.transition_id not in retained_ids
        ),
    )
    return retained
