"""Bounded resumable paths from incomplete discovery runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yewo.agent.actuation import ActionOutcomeSignal, SideEffectClass
from yewo.agent.narration import narrate_memory_eviction
from yewo.artifacts import RecordIdentifier

if TYPE_CHECKING:
    from yewo.agent.app_memory import (
        AcceptedTransition,
        AppMemory,
        CapabilityEvidence,
        RouteSegment,
        VerifiedRoute,
    )


MAXIMUM_DISCOVERY_CHECKPOINTS = 10


@dataclass(frozen=True)
class RetainedDiscoveryGraph:
    """One bounded memory graph with internally retained references."""

    transitions: tuple[AcceptedTransition, ...]
    checkpoints: tuple[DiscoveryCheckpoint, ...]
    routes: tuple[VerifiedRoute, ...]
    capabilities: tuple[CapabilityEvidence, ...]


class DiscoveryCheckpoint(BaseModel):
    """An unverified accepted path retained only to resume discovery."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    checkpoint_id: RecordIdentifier
    objective_id: RecordIdentifier
    objective: str = Field(min_length=1, max_length=4_000)
    start_screen_id: RecordIdentifier
    destination_screen_id: RecordIdentifier
    transition_ids: tuple[RecordIdentifier, ...] = Field(min_length=1)
    source_run_id: RecordIdentifier
    updated_at: datetime

    @model_validator(mode="after")
    def _validate_checkpoint(self) -> Self:
        if self.updated_at.utcoffset() is None:
            raise ValueError("A discovery checkpoint time must include a timezone.")
        return self


def build_discovery_checkpoint(
    objective_id: RecordIdentifier,
    objective: str,
    transitions: list[AcceptedTransition],
    source_run_id: RecordIdentifier,
    updated_at: datetime,
) -> DiscoveryCheckpoint | None:
    """Build one loop-pruned checkpoint from contiguous accepted evidence."""
    from yewo.agent.app_memory import derived_memory_identifier

    retained = _remove_reversible_cycles(transitions)
    if not retained:
        return None
    transition_ids = tuple(item.transition_id for item in retained)
    return DiscoveryCheckpoint(
        checkpoint_id=derived_memory_identifier("checkpoint", objective_id, *transition_ids),
        objective_id=objective_id,
        objective=objective,
        start_screen_id=retained[0].source_screen_id,
        destination_screen_id=retained[-1].destination_screen_id,
        transition_ids=transition_ids,
        source_run_id=source_run_id,
        updated_at=updated_at,
    )


def select_discovery_checkpoint(
    memory: AppMemory,
    objective_id: RecordIdentifier,
    objective: str,
) -> DiscoveryCheckpoint | None:
    """Select the newest compatible checkpoint for one exact objective."""
    return next(
        (
            checkpoint
            for checkpoint in reversed(memory.checkpoints)
            if checkpoint.objective_id == objective_id and checkpoint.objective == objective
        ),
        None,
    )


def select_verified_route(
    memory: AppMemory,
    objective: str,
    *,
    objective_id: RecordIdentifier | None = None,
) -> VerifiedRoute | None:
    """Select the newest capability-backed route for one exact objective."""
    capability_route_ids = {
        capability.route_id
        for capability in memory.capabilities
        if objective_id is None or capability.objective_id == objective_id
    }
    return next(
        (
            route
            for route in reversed(memory.routes)
            if route.objective == objective
            and route.route_id in capability_route_ids
            and (objective_id is None or route.objective_id == objective_id)
        ),
        None,
    )


def checkpoint_segments(
    memory: AppMemory,
    checkpoint: DiscoveryCheckpoint,
) -> tuple[RouteSegment, ...]:
    """Resolve checkpoint transition references into replay segments."""
    from yewo.agent.app_memory import RouteSegment

    transitions = {item.transition_id: item for item in memory.transitions}
    return tuple(
        RouteSegment(
            transition_id=transition_id,
            source_screen_id=transitions[transition_id].source_screen_id,
            destination_screen_id=transitions[transition_id].destination_screen_id,
            action=transitions[transition_id].action,
        )
        for transition_id in checkpoint.transition_ids
    )


def retain_discovery_graph(
    transitions: list[AcceptedTransition],
    routes: list[VerifiedRoute],
    checkpoints: list[DiscoveryCheckpoint],
    capabilities: list[CapabilityEvidence],
    maximum: int,
) -> RetainedDiscoveryGraph:
    """Bound the memory graph without orphaning retained route evidence."""
    retained_routes = list(routes)
    retained_checkpoints = list(checkpoints[-MAXIMUM_DISCOVERY_CHECKPOINTS:])
    retained_capabilities = list(capabilities)
    evicted_checkpoint_ids = [
        item.checkpoint_id for item in checkpoints[:-MAXIMUM_DISCOVERY_CHECKPOINTS]
    ]
    evicted_route_ids: list[RecordIdentifier] = []
    evicted_capability_ids: list[RecordIdentifier] = []
    while len(retained_capabilities) > maximum:
        evicted_capability_ids.append(retained_capabilities.pop(0).capability_id)
    while len(retained_routes) > maximum:
        _evict_route(
            retained_routes,
            retained_capabilities,
            evicted_route_ids,
            evicted_capability_ids,
        )
    required = _required_transition_ids(retained_routes, retained_checkpoints)
    while len(required) > maximum:
        if retained_checkpoints:
            evicted_checkpoint_ids.append(retained_checkpoints.pop(0).checkpoint_id)
        else:
            _evict_route(
                retained_routes,
                retained_capabilities,
                evicted_route_ids,
                evicted_capability_ids,
            )
        required = _required_transition_ids(retained_routes, retained_checkpoints)
    optional = [item for item in transitions if item.transition_id not in required]
    available = maximum - len(required)
    retained_optional = optional[-available:] if available else []
    retained_ids = required | {item.transition_id for item in retained_optional}
    retained_transitions = tuple(
        item for item in transitions if item.transition_id in retained_ids
    )
    narrate_memory_eviction("checkpoint", tuple(evicted_checkpoint_ids))
    narrate_memory_eviction("route", tuple(evicted_route_ids))
    narrate_memory_eviction("capability", tuple(evicted_capability_ids))
    narrate_memory_eviction(
        "transition",
        tuple(
            item.transition_id
            for item in transitions
            if item.transition_id not in retained_ids
        ),
    )
    return RetainedDiscoveryGraph(
        transitions=retained_transitions,
        checkpoints=tuple(retained_checkpoints),
        routes=tuple(retained_routes),
        capabilities=tuple(retained_capabilities),
    )


def _evict_route(
    routes: list[VerifiedRoute],
    capabilities: list[CapabilityEvidence],
    evicted_route_ids: list[RecordIdentifier],
    evicted_capability_ids: list[RecordIdentifier],
) -> None:
    capability_route_ids = {capability.route_id for capability in capabilities}
    removable_index = next(
        (
            index
            for index, route in enumerate(routes)
            if route.route_id not in capability_route_ids
        ),
        0,
    )
    removed_route = routes.pop(removable_index)
    evicted_route_ids.append(removed_route.route_id)
    removed_capabilities = [
        capability for capability in capabilities if capability.route_id == removed_route.route_id
    ]
    evicted_capability_ids.extend(
        capability.capability_id for capability in removed_capabilities
    )
    capabilities[:] = [
        capability for capability in capabilities if capability.route_id != removed_route.route_id
    ]


def _required_transition_ids(
    routes: list[VerifiedRoute],
    checkpoints: list[DiscoveryCheckpoint],
) -> set[RecordIdentifier]:
    return {
        *(segment.transition_id for route in routes for segment in route.segments),
        *(item for checkpoint in checkpoints for item in checkpoint.transition_ids),
    }


def validate_discovery_checkpoints(
    source_run_ids: tuple[RecordIdentifier, ...],
    transitions: tuple[AcceptedTransition, ...],
    checkpoints: tuple[DiscoveryCheckpoint, ...],
) -> None:
    """Require every checkpoint to reference one contiguous accepted path."""
    checkpoint_ids = [checkpoint.checkpoint_id for checkpoint in checkpoints]
    if len(checkpoint_ids) != len(set(checkpoint_ids)):
        raise ValueError("App memory checkpoint identifiers must be unique.")
    transitions_by_id = {item.transition_id: item for item in transitions}
    for checkpoint in checkpoints:
        path = [transitions_by_id.get(item) for item in checkpoint.transition_ids]
        if (
            any(transition is None for transition in path)
            or checkpoint.source_run_id not in source_run_ids
            or path[0] is None
            or path[0].source_screen_id != checkpoint.start_screen_id
            or path[-1] is None
            or path[-1].destination_screen_id != checkpoint.destination_screen_id
        ):
            raise ValueError("A discovery checkpoint must reference its accepted path.")
        accepted_path = [item for item in path if item is not None]
        if any(
            previous.destination_screen_id != current.source_screen_id
            for previous, current in zip(accepted_path, accepted_path[1:], strict=False)
        ):
            raise ValueError("A discovery checkpoint path must be contiguous.")


def _remove_reversible_cycles(
    transitions: list[AcceptedTransition],
) -> list[AcceptedTransition]:
    if not transitions:
        return []
    retained: list[AcceptedTransition] = []
    screen_path = [transitions[0].source_screen_id]
    for transition in transitions:
        if screen_path[-1] != transition.source_screen_id:
            retained = []
            screen_path = [transition.source_screen_id]
        matching_indexes = [
            index
            for index, screen_id in enumerate(screen_path)
            if screen_id == transition.destination_screen_id
        ]
        if transition.destination_screen_id != transition.source_screen_id and matching_indexes:
            destination_index = matching_indexes[-1]
            cycle = (*retained[destination_index:], transition)
            if all(
                item.side_effect is not SideEffectClass.APPLICATION_MUTATION
                and item.outcome_signals <= {ActionOutcomeSignal.SCREEN_CHANGED}
                for item in cycle
            ):
                retained = retained[:destination_index]
                screen_path = screen_path[: destination_index + 1]
                continue
        retained.append(transition)
        screen_path.append(transition.destination_screen_id)
    return retained
