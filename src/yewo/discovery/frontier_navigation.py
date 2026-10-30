"""Find verified route prefixes back to discovery work."""

from collections import deque
from dataclasses import dataclass, replace
from enum import Enum

from yewo.agent.actuation.safety import ActionSafetyPolicy
from yewo.discovery.candidate_selection import CandidateSelection, authorize_candidates
from yewo.discovery.contracts import DiscoveryRun, FrontierState, SideEffectClass
from yewo.discovery.runtime import DiscoveryObservationBatch


class FrontierReachability(str, Enum):
    """Why no replay step was returned. The two reasons are not one answer.

    Standing on the screen that owns open work is not the same as having no
    route to any. A caller that cannot tell them apart reads work in front of
    it as work it can never reach. It then writes off every other screen too.
    """

    WORK_IS_HERE = "work_is_here"
    NO_ROUTE = "no_route"


@dataclass(frozen=True)
class FrontierReplayStep:
    control_identity: str
    destination_screen_state_id: str
    expected_destination_phash: str


def select_frontier_replay_step(
    run: DiscoveryRun,
    current_screen_state_id: str,
) -> FrontierReplayStep | FrontierReachability:
    observations_by_id = {
        observation.observation_id: observation for observation in run.observations
    }
    target_screen_state_ids = {
        observations_by_id[entry.source_observation_id].screen_state_id
        for entry in run.frontier
        if entry.state is FrontierState.AVAILABLE
    }
    if current_screen_state_id in target_screen_state_ids:
        return FrontierReachability.WORK_IS_HERE

    verified_transition_ids = {
        entry.transition_identity
        for entry in run.frontier
        if entry.state is FrontierState.VERIFIED
        and entry.transition_identity is not None
    }
    edges_by_source: dict[str, list[FrontierReplayStep]] = {}
    for transition in run.transitions:
        if transition.transition_identity not in verified_transition_ids:
            continue
        source = observations_by_id[transition.source_observation_id]
        destination = observations_by_id[transition.destination_observation_id]
        edges_by_source.setdefault(source.screen_state_id, []).append(
            FrontierReplayStep(
                control_identity=transition.control_identity,
                destination_screen_state_id=destination.screen_state_id,
                expected_destination_phash=destination.phash,
            )
        )

    pending: deque[tuple[str, FrontierReplayStep | None]] = deque(
        [(current_screen_state_id, None)]
    )
    visited = {current_screen_state_id}
    while pending:
        screen_state_id, first_step = pending.popleft()
        for step in sorted(
            edges_by_source.get(screen_state_id, ()),
            key=lambda item: (
                item.destination_screen_state_id,
                item.control_identity,
            ),
        ):
            if step.destination_screen_state_id in visited:
                continue
            route_first_step = first_step or step
            if step.destination_screen_state_id in target_screen_state_ids:
                return route_first_step
            visited.add(step.destination_screen_state_id)
            pending.append((step.destination_screen_state_id, route_first_step))
    return FrontierReachability.NO_ROUTE


def stranded_frontier_ids(
    run: DiscoveryRun,
    batch: DiscoveryObservationBatch,
) -> set[str]:
    """Open offers on the screen in front of the run that it cannot act on.

    An offer names a control on a screen. Standing on that screen while the
    frame does not carry that control means the run is as close as it can get.
    It still cannot take the offer. Below the fold and behind an overlay both
    read this way. So does an offer left under an earlier naming scheme.

    Left open, one such offer makes the screen read as work in hand. The run
    then stops looking for a route to work it could still reach.
    """
    offerable = {candidate.control_identity for candidate in batch.candidates}
    observations_by_id = {
        observation.observation_id: observation for observation in run.observations
    }
    return {
        entry.frontier_id
        for entry in run.frontier
        if entry.state is FrontierState.AVAILABLE
        and observations_by_id[entry.source_observation_id].screen_state_id
        == batch.observation.screen_state_id
        and entry.control_identity not in offerable
    }


def select_frontier_replay_candidate(
    run: DiscoveryRun,
    batch: DiscoveryObservationBatch,
    action_policy: ActionSafetyPolicy,
    allowed_side_effects: frozenset[SideEffectClass],
) -> CandidateSelection | FrontierReachability:
    replay = select_frontier_replay_step(
        run,
        batch.observation.screen_state_id,
    )
    if isinstance(replay, FrontierReachability):
        return replay
    matching = tuple(
        replace(
            candidate,
            expected_destination_phash=replay.expected_destination_phash,
        )
        for candidate in batch.candidates
        if candidate.control_identity == replay.control_identity
    )
    authorized = authorize_candidates(
        run,
        batch,
        matching,
        action_policy,
        allowed_side_effects,
    )
    if not authorized:
        return FrontierReachability.NO_ROUTE
    candidate, authorization = next(iter(authorized.values()))
    return CandidateSelection(
        candidate,
        authorization,
        replay.destination_screen_state_id,
    )
