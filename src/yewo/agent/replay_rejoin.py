"""Stitch retained checkpoints and rejoin valid downstream replay segments."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yewo.agent.actuation import ActionKind, ActionProposal, DeclaredValueReference, LiteralText
from yewo.agent.app_memory import (
    AcceptedTransition,
    AppMemory,
    CapabilityEvidence,
    ObservationVariant,
    RouteSegment,
    SemanticRouteAction,
    VerifiedRoute,
    derived_memory_identifier,
)
from yewo.agent.discovery_checkpoint import (
    DiscoveryCheckpoint,
    build_discovery_checkpoint,
    checkpoint_segments,
    retain_discovery_graph,
)
from yewo.agent.planner import ActionProposalSource, PlanningRequest
from yewo.agent.planning_evidence import (
    PlannerDecisionMetadata,
    PlannerDecisionSource,
    resolve_planner_decision_metadata,
)
from yewo.agent.screen_memory import (
    ExactObservationStateKey,
    exact_variant_state_key,
    observation_matches_replay_variant,
    retain_observation_variants,
)
from yewo.artifacts import RecordIdentifier, new_record_identifier
from yewo.observations import Observation


MAXIMUM_STITCHED_REPLAY_SEGMENTS, MAXIMUM_REPLAY_SEARCH_OBSERVATIONS = 100, 100


class ReplayRejoinPlan(BaseModel):
    """A downstream retained path that repair planning can rejoin."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path_id: RecordIdentifier
    decision_source: PlannerDecisionSource
    transition_ids: tuple[RecordIdentifier, ...] = Field(min_length=1, max_length=500)
    minimum_segment_index: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_minimum_segment(self) -> ReplayRejoinPlan:
        if self.minimum_segment_index >= len(self.transition_ids):
            raise ValueError("A replay rejoin must retain a downstream segment.")
        if self.decision_source is PlannerDecisionSource.MODEL:
            raise ValueError("A replay rejoin requires retained-path provenance.")
        return self


def build_replay_rejoin_plans(
    path_id: RecordIdentifier,
    decision_source: PlannerDecisionSource,
    replay_transition_ids: tuple[RecordIdentifier, ...],
    downstream_transition_ids: tuple[RecordIdentifier, ...],
    rejoin_segment_index: int | None,
    *,
    replay_diverged: bool,
) -> tuple[ReplayRejoinPlan, ...]:
    """Retain the untried replay suffixes that repair planning can rejoin."""
    plans = []
    if (
        replay_diverged
        and rejoin_segment_index is not None and rejoin_segment_index < len(replay_transition_ids)
    ):
        plans.append(
            ReplayRejoinPlan(
                path_id=path_id,
                decision_source=decision_source,
                transition_ids=replay_transition_ids,
                minimum_segment_index=rejoin_segment_index,
            )
        )
    if downstream_transition_ids:
        plans.append(
            ReplayRejoinPlan(
                path_id=path_id,
                decision_source=decision_source,
                transition_ids=downstream_transition_ids,
                minimum_segment_index=0,
            )
        )
    return tuple(plans)


@dataclass(frozen=True)
class _PendingReplayAction:
    proposal_id: str
    segment_index: int
    segment_count: int


@dataclass(frozen=True)
class CheckpointReplaySelection:
    """One contiguous replay prefix and an optional downstream suffix."""

    replay_segments: tuple[RouteSegment, ...]
    downstream_segments: tuple[RouteSegment, ...] = ()


class ReplayRejoinActionSource:
    """Use model repair until the live screen rejoins a downstream path guard."""

    def __init__(
        self,
        source: ActionProposalSource,
        memory: AppMemory,
        plan: ReplayRejoinPlan,
        *,
        identifier_factory: Callable[[], str] = new_record_identifier,
    ) -> None:
        self._source = source
        self._screen_comparison = memory.compatibility.screen_comparison
        self._transitions = {item.transition_id: item for item in memory.transitions}
        self._variants = {
            variant.observation_id: variant
            for screen in memory.screens
            for variant in screen.variants
        }
        self._segments = _segments_for_transition_ids(memory, plan.transition_ids)
        self._path_id = plan.path_id
        self._decision_source = plan.decision_source
        self._minimum_segment_index = plan.minimum_segment_index
        self._identifier_factory = identifier_factory
        self._pending: _PendingReplayAction | None = None
        self._last_metadata = PlannerDecisionMetadata(source=PlannerDecisionSource.MODEL)

    def required_model_requests(self, request: PlanningRequest) -> int:
        """Count no model call when a downstream source guard matches."""
        minimum_index = self._minimum_after_pending()
        if self._select_rejoin_index(request, minimum_index) is not None:
            return 0
        return self._source.required_model_requests(request)

    def plan(self, request: PlanningRequest) -> ActionProposal:
        """Replay a matching downstream segment or delegate repair planning."""
        self._minimum_segment_index = self._minimum_after_pending()
        self._pending = None
        segment_index = self._select_rejoin_index(request, self._minimum_segment_index)
        if segment_index is None:
            proposal = self._source.plan(request)
            self._last_metadata = resolve_planner_decision_metadata(self._source)
            return proposal
        action, segment_count = replay_semantic_action(self._segments, segment_index)
        proposal_id = self._identifier_factory()
        self._pending = _PendingReplayAction(
            proposal_id=proposal_id,
            segment_index=segment_index,
            segment_count=segment_count,
        )
        self._minimum_segment_index = segment_index
        self._last_metadata = PlannerDecisionMetadata(
            source=self._decision_source,
            reused_flow_id=self._path_id,
            reused_transition_ids=tuple(
                segment.transition_id
                for segment in self._segments[segment_index : segment_index + segment_count]
            ),
            reused_segment_index=segment_index,
        )
        return build_replay_action_proposal(
            action,
            proposal_id,
            request.observation.observation_id,
        )

    def decision_metadata(self) -> PlannerDecisionMetadata:
        """Return provenance for the proposal selected by the last call."""
        return self._last_metadata

    def _minimum_after_pending(self) -> int:
        """Consume one retained attempt before repair can reuse later segments."""
        pending = self._pending
        if pending is None:
            return self._minimum_segment_index
        return pending.segment_index + pending.segment_count

    def _select_rejoin_index(
        self,
        request: PlanningRequest,
        minimum_index: int,
    ) -> int | None:
        for segment_index in range(minimum_index, len(self._segments)):
            segment = self._segments[segment_index]
            transition = self._transitions[segment.transition_id]
            source = self._variants[transition.source_observation_id]
            if not observation_matches_replay_variant(
                request.observation,
                source,
                self._screen_comparison,
            ):
                continue
            action, unused_count = replay_semantic_action(self._segments, segment_index)
            if action.action not in request.available_actions:
                return None
            return segment_index
        return None


def select_checkpoint_replay_segments(
    memory: AppMemory,
    checkpoint: DiscoveryCheckpoint,
    observation: Observation,
    *,
    allow_declared_values: bool = True,
    target_transition_id: RecordIdentifier | None = None,
) -> CheckpointReplaySelection:
    """Select the shortest guarded path to a reconciled replay frontier."""
    retained = checkpoint_segments(memory, checkpoint)
    transitions = {item.transition_id: item for item in memory.transitions}
    matching_state_keys = tuple(
        exact_variant_state_key(variant)
        for screen in memory.screens
        for variant in screen.variants
        if observation_matches_replay_variant(
            observation,
            variant,
            memory.compatibility.screen_comparison,
        )
    )
    paths = _reachable_paths(memory, matching_state_keys, allow_declared_values)
    if target_transition_id is not None:
        target_transition = transitions.get(target_transition_id)
        if target_transition is None:
            return CheckpointReplaySelection((), retained)
        target_key = exact_variant_state_key(
            _resolve_observation_variant(
                memory,
                target_transition.destination_observation_id,
            )
        )
        target_path = paths.get(target_key)
        return (
            CheckpointReplaySelection(target_path)
            if target_path is not None
            else CheckpointReplaySelection((), retained)
        )
    for segment_index, segment in enumerate(retained):
        transition = transitions[segment.transition_id]
        if (
            exact_variant_state_key(
                _resolve_observation_variant(memory, transition.source_observation_id)
            )
            in matching_state_keys
        ):
            return CheckpointReplaySelection(retained[segment_index:])
    final_transition = transitions[retained[-1].transition_id]
    if (
        exact_variant_state_key(
            _resolve_observation_variant(memory, final_transition.destination_observation_id)
        )
        in matching_state_keys
    ):
        return CheckpointReplaySelection(())
    first_transition = transitions[retained[0].transition_id]
    checkpoint_source_key = exact_variant_state_key(
        _resolve_observation_variant(memory, first_transition.source_observation_id)
    )
    prefix = paths.get(checkpoint_source_key)
    if prefix is not None:
        stitched = (*prefix, *retained)
        if len(stitched) <= MAXIMUM_STITCHED_REPLAY_SEGMENTS:
            return CheckpointReplaySelection(stitched)
    return CheckpointReplaySelection((), retained)


def replay_semantic_action(
    segments: Sequence[RouteSegment],
    segment_index: int,
) -> tuple[SemanticRouteAction, int]:
    """Return one stored action, collapsing a compatible focus and type pair."""
    segment = segments[segment_index]
    next_index = segment_index + 1
    if segment.action.action is not ActionKind.FOCUS or next_index >= len(segments):
        return segment.action, 1
    next_segment = segments[next_index]
    if (
        next_segment.action.action is ActionKind.TYPE
        and next_segment.action.input_contract is not None
        and next_segment.action.target_description == segment.action.target_description
        and segment.destination_screen_id == next_segment.source_screen_id
    ):
        return next_segment.action, 2
    return segment.action, 1


def build_replay_action_proposal(
    action: SemanticRouteAction,
    proposal_id: str,
    observation_id: str,
) -> ActionProposal:
    """Convert one retained semantic action into a current proposal."""
    return ActionProposal(
        proposal_id=proposal_id,
        source_observation_id=observation_id,
        action=action.action,
        target_expression=action.target_description,
        destination_expression=action.destination_description,
        expected_outcome=action.expected_outcome,
        reason="Follow the stored semantic path toward its destination.",
        direction=action.direction,
        distance=action.distance,
        duration_seconds=action.duration_seconds,
        text=_replay_text(action),
    )


def combine_replayed_checkpoint_prefix(
    memory: AppMemory,
    objective_id: RecordIdentifier,
    objective: str,
    replay_transition_ids: tuple[RecordIdentifier, ...],
    source_run_id: RecordIdentifier,
    updated_at: datetime,
) -> AppMemory:
    """Prepend a verified replay prefix to this run's repaired checkpoint."""
    if not replay_transition_ids:
        return memory
    repaired = next(
        (
            checkpoint
            for checkpoint in memory.checkpoints
            if checkpoint.objective_id == objective_id and checkpoint.source_run_id == source_run_id
        ),
        None,
    )
    if repaired is None:
        return memory
    transitions = {item.transition_id: item for item in memory.transitions}
    path_ids = (*replay_transition_ids, *repaired.transition_ids)
    if any(transition_id not in transitions for transition_id in path_ids):
        return memory
    path = [transitions[transition_id] for transition_id in path_ids]
    if any(
        previous.destination_screen_id != current.source_screen_id
        for previous, current in zip(path, path[1:], strict=False)
    ):
        return memory
    combined = build_discovery_checkpoint(
        objective_id,
        objective,
        path,
        source_run_id,
        updated_at,
    )
    if combined is None:
        return memory
    checkpoints = tuple(
        checkpoint for checkpoint in memory.checkpoints if checkpoint.objective_id != objective_id
    ) + (combined,)
    return AppMemory.model_validate(
        {
            **memory.model_dump(mode="python"),
            "checkpoints": checkpoints,
        }
    )


def combine_repaired_route(
    memory: AppMemory,
    original: VerifiedRoute,
    completed_segment_count: int,
    repaired_route_id: str | None,
) -> tuple[AppMemory, VerifiedRoute, CapabilityEvidence] | None:
    """Combine a verified replay prefix with one independently verified repair."""
    if completed_segment_count == 0 or repaired_route_id is None:
        return None
    repaired = next(
        (route for route in memory.routes if route.route_id == repaired_route_id),
        None,
    )
    prefix = original.segments[:completed_segment_count]
    if (
        repaired is None
        or not prefix
        or prefix[-1].destination_screen_id != repaired.start_screen_id
    ):
        return None
    segments = (*prefix, *repaired.segments)
    transitions = {item.transition_id: item for item in memory.transitions}
    observation_ids = (
        transitions[prefix[0].transition_id].source_observation_id,
        *(transitions[item.transition_id].destination_observation_id for item in segments),
    )
    combined = VerifiedRoute(
        route_id=derived_memory_identifier(
            "repaired-route",
            original.route_id,
            repaired.route_id,
            *(item.transition_id for item in segments),
        ),
        objective_id=repaired.objective_id,
        objective=repaired.objective,
        start_screen_id=original.start_screen_id,
        destination_screen_id=repaired.destination_screen_id,
        segments=segments,
        evidence_observation_ids=tuple(dict.fromkeys(observation_ids)),
    )
    capability = CapabilityEvidence(
        capability_id=derived_memory_identifier(
            "repaired-capability",
            repaired.objective_id,
            combined.route_id,
        ),
        objective_id=repaired.objective_id,
        statement=f"The application can reach: {repaired.objective}",
        route_id=combined.route_id,
        evidence_observation_ids=combined.evidence_observation_ids,
    )
    retained_graph = retain_discovery_graph(
        list(memory.transitions),
        [*memory.routes, combined],
        list(memory.checkpoints),
        [*memory.capabilities, capability],
        memory.compatibility.retention.maximum_transitions,
    )
    if combined.route_id not in {
        route.route_id for route in retained_graph.routes
    } or capability.capability_id not in {
        item.capability_id for item in retained_graph.capabilities
    }:
        return None
    retained_screens = retain_observation_variants(
        list(memory.screens),
        retained_graph.transitions,
        memory.unsuccessful_attempts,
        retained_graph.routes,
        retained_graph.capabilities,
        memory.compatibility.retention.maximum_variants_per_screen,
    )
    updated = AppMemory.model_validate(
        {
            **memory.model_dump(mode="python"),
            "screens": retained_screens,
            "transitions": retained_graph.transitions,
            "checkpoints": retained_graph.checkpoints,
            "routes": retained_graph.routes,
            "capabilities": retained_graph.capabilities,
        }
    )
    return updated, combined, capability


def _reachable_paths(
    memory: AppMemory,
    start_state_keys: tuple[ExactObservationStateKey, ...],
    allow_declared_values: bool,
) -> dict[ExactObservationStateKey, tuple[RouteSegment, ...]]:
    variants = {
        variant.observation_id: variant for screen in memory.screens for variant in screen.variants
    }
    transitions_by_source: dict[ExactObservationStateKey, list[AcceptedTransition]] = {}
    for transition in reversed(memory.transitions):
        source_key = exact_variant_state_key(variants[transition.source_observation_id])
        transitions_by_source.setdefault(source_key, []).append(transition)
    queue = deque(start_state_keys)
    paths: dict[ExactObservationStateKey, tuple[RouteSegment, ...]] = {
        state_key: () for state_key in start_state_keys
    }
    while queue:
        source_state_key = queue.popleft()
        for transition in transitions_by_source.get(source_state_key, ()):
            if not allow_declared_values and transition.action.declared_value_name is not None:
                continue
            destination_key = exact_variant_state_key(
                variants[transition.destination_observation_id]
            )
            if destination_key in paths:
                continue
            path = (
                *paths[source_state_key],
                RouteSegment(
                    transition_id=transition.transition_id,
                    source_screen_id=transition.source_screen_id,
                    destination_screen_id=transition.destination_screen_id,
                    action=transition.action,
                ),
            )
            if len(path) > MAXIMUM_STITCHED_REPLAY_SEGMENTS:
                continue
            if len(paths) >= MAXIMUM_REPLAY_SEARCH_OBSERVATIONS:
                return paths
            paths[destination_key] = path
            queue.append(destination_key)
    return paths


def _resolve_observation_variant(
    memory: AppMemory,
    observation_id: str,
) -> ObservationVariant:
    return next(
        variant
        for screen in memory.screens
        for variant in screen.variants
        if variant.observation_id == observation_id
    )


def _segments_for_transition_ids(
    memory: AppMemory,
    transition_ids: tuple[str, ...],
) -> tuple[RouteSegment, ...]:
    transitions = {transition.transition_id: transition for transition in memory.transitions}
    segments = tuple(
        RouteSegment(
            transition_id=transition_id,
            source_screen_id=transitions[transition_id].source_screen_id,
            destination_screen_id=transitions[transition_id].destination_screen_id,
            action=transitions[transition_id].action,
        )
        for transition_id in transition_ids
        if transition_id in transitions
    )
    if len(segments) != len(transition_ids):
        raise ValueError("A replay rejoin references unavailable transition evidence.")
    if any(
        previous.destination_screen_id != current.source_screen_id
        for previous, current in zip(segments, segments[1:], strict=False)
    ):
        raise ValueError("A replay rejoin path must be contiguous.")
    return segments


def _replay_text(
    action: SemanticRouteAction,
) -> DeclaredValueReference | LiteralText | None:
    if action.declared_value_name is not None:
        return DeclaredValueReference(value_name=action.declared_value_name)
    if action.disposable_text is not None:
        return LiteralText(value=action.disposable_text)
    return None
