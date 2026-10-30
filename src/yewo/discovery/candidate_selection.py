"""Authorize and order exact-screen discovery candidates."""

from dataclasses import dataclass

from yewo.agent.actuation.safety import (
    ActionAuthorization,
    ActionSafetyPolicy,
    ActionSource,
)
from yewo.discovery.contracts import DiscoveryRun, FrontierState, SideEffectClass
from yewo.discovery.control_identity import discovery_control_identity
from yewo.discovery.runtime import DiscoveryActionCandidate, DiscoveryObservationBatch

AuthorizedCandidate = tuple[DiscoveryActionCandidate, ActionAuthorization]


@dataclass(frozen=True)
class CandidateSelection:
    candidate: DiscoveryActionCandidate
    authorization: ActionAuthorization
    replay_destination_state_id: str | None = None


def select_available_candidates(
    run: DiscoveryRun,
    batch: DiscoveryObservationBatch,
) -> tuple[DiscoveryActionCandidate, ...]:
    frontier_state_by_id = {entry.frontier_id: entry.state for entry in run.frontier}
    return tuple(
        candidate
        for candidate in batch.candidates
        if frontier_state_by_id.get(candidate.frontier_id) is FrontierState.AVAILABLE
    )


def authorize_candidates(
    run: DiscoveryRun,
    batch: DiscoveryObservationBatch,
    candidates: tuple[DiscoveryActionCandidate, ...],
    action_policy: ActionSafetyPolicy,
    allowed_side_effects: frozenset[SideEffectClass],
) -> dict[str, AuthorizedCandidate]:
    authorized: dict[str, AuthorizedCandidate] = {}
    for candidate in candidates:
        if candidate.side_effect not in allowed_side_effects:
            continue
        if not run.usage.can_attempt_transition(
            candidate.attempt_identity, run.budgets
        ):
            continue
        try:
            authorization = action_policy.authorize(
                candidate.action,
                source=ActionSource.DISCOVERY,
                observation_id=candidate.source_observation_id,
                screen_phash=candidate.source_phash,
                screen_size=candidate.screen_size,
                context_id=run.header.artifact_id,
                keyboard_visible=batch.observation.keyboard_visible,
                expected_destination_phash=candidate.expected_destination_phash,
            )
        except (TypeError, ValueError):
            continue
        if (
            discovery_control_identity(
                candidate.source_screen_state_id,
                authorization.action,
                candidate.screen_size,
            )
            != candidate.control_identity
        ):
            continue
        authorized[candidate.frontier_id] = (candidate, authorization)
    return authorized


def prioritize_authorized_candidates(
    batch: DiscoveryObservationBatch,
    authorized: dict[str, AuthorizedCandidate],
) -> tuple[AuthorizedCandidate, ...]:
    controls_by_id = {
        control.control_identity: control for control in batch.observation.controls
    }

    def priority(item: AuthorizedCandidate) -> tuple[bool, int, int, str]:
        candidate = item[0]
        control = controls_by_id[candidate.control_identity]
        return (
            not bool(control.text_digest),
            control.bounds.top,
            control.bounds.left,
            candidate.frontier_id,
        )

    return tuple(sorted(authorized.values(), key=priority))
