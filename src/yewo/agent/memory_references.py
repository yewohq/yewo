"""Observation-reference ownership for retained YewoAgent memory."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from yewo.artifacts import RecordIdentifier

if TYPE_CHECKING:
    from yewo.agent.app_memory import (
        AcceptedTransition,
        CapabilityEvidence,
        UnsuccessfulAttempt,
        VerifiedRoute,
    )


def referenced_memory_observation_ids(
    transitions: Sequence[AcceptedTransition],
    unsuccessful_attempts: Sequence[UnsuccessfulAttempt],
    routes: Sequence[VerifiedRoute],
    capabilities: Sequence[CapabilityEvidence],
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
        *(item for route in routes for item in route.evidence_observation_ids),
        *(item for capability in capabilities for item in capability.evidence_observation_ids),
    }
    for transition in transitions:
        for grounding in (transition.target_grounding, transition.destination_grounding):
            if grounding is not None:
                referenced.add(
                    grounding.source_observation_id or transition.source_observation_id
                )
    return referenced


def validate_retained_observations(
    observation_ids: list[RecordIdentifier],
    routes: tuple[VerifiedRoute, ...],
    capabilities: tuple[CapabilityEvidence, ...],
) -> None:
    """Require route and capability observations to remain available."""
    if len(observation_ids) != len(set(observation_ids)):
        raise ValueError("A memory observation must belong to one logical screen.")
    known_observations = set(observation_ids)
    if any(
        observation_id not in known_observations
        for route in routes
        for observation_id in route.evidence_observation_ids
    ):
        raise ValueError("Every route must reference retained observations.")
    if any(
        observation_id not in known_observations
        for capability in capabilities
        for observation_id in capability.evidence_observation_ids
    ):
        raise ValueError("Every capability must reference retained observations.")


def validate_route_evidence(
    routes: tuple[VerifiedRoute, ...],
    capabilities: tuple[CapabilityEvidence, ...],
    transitions_by_id: dict[RecordIdentifier, AcceptedTransition],
) -> None:
    """Bind route observations to transitions and capabilities to routes."""
    routes_by_id = {route.route_id: route for route in routes}
    for route in routes:
        route_transitions = [
            transitions_by_id[segment.transition_id] for segment in route.segments
        ]
        expected_observations = (
            route_transitions[0].source_observation_id,
            *(transition.destination_observation_id for transition in route_transitions),
        )
        if route.evidence_observation_ids != tuple(dict.fromkeys(expected_observations)):
            raise ValueError("Route observations must match its accepted transitions.")
    if any(
        capability.evidence_observation_ids
        != routes_by_id[capability.route_id].evidence_observation_ids
        for capability in capabilities
    ):
        raise ValueError("Capability observations must match its verified route.")
