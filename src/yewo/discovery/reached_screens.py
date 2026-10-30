"""Retain report-only capability candidates for reached screens.

A run that misses its goal still learned which screens exist and how it got
to them. These records carry no requirement authority: a reachable screen is
not a product rule that it should be reachable. They exist so a partial
outcome is reportable rather than only present in the raw map.

The isolation check prevents scenario-owned values from entering shared app
knowledge.
"""

from __future__ import annotations

from collections.abc import Sequence

from yewo.discovery.artifacts import validate_shared_knowledge_payload
from yewo.discovery.contracts import CapabilityCandidate, DiscoveryRun


def merge_reached_screen_candidates(
    run: DiscoveryRun,
    observed: Sequence[CapabilityCandidate],
    scenario_owned_values: Sequence[str] = (),
) -> tuple[CapabilityCandidate, ...] | None:
    """The run's records plus the ones it has not seen, or nothing new.

    Args:
        run: The open discovery run.
        observed: What this observation reported reaching.
        scenario_owned_values: Values one scenario owns, which shared app
            knowledge may never carry.

    Returns:
        Every record the run should now hold, or None when none is new.

    Raises:
        ValueError: A record names a value one scenario owns.
    """
    known = {candidate.candidate_id for candidate in run.capability_candidates}
    fresh = tuple(
        candidate for candidate in observed if candidate.candidate_id not in known
    )
    if not fresh:
        return None
    validate_shared_knowledge_payload(
        [candidate.model_dump(mode="json") for candidate in fresh],
        scenario_owned_values=scenario_owned_values,
    )
    return (*run.capability_candidates, *fresh)
