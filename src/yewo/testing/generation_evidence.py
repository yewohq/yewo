"""Verified app knowledge exposed to semantic scenario generation."""

from __future__ import annotations

import hashlib
import json

from pydantic import Field, model_validator

from yewo.discovery.artifacts import DiscoveryContract, verify_artifact
from yewo.discovery.contracts import CapabilityState, SUCCESSFUL_CAPABILITY_STATES
from yewo.discovery.knowledge import AppMap, RouteMemory, RouteMemoryLifecycle


class GenerationCapabilityEvidence(DiscoveryContract):
    """A capability terminal the generator may reference but cannot reinterpret."""

    capability_id: str
    intent: str
    state: CapabilityState
    required: bool
    evidence_observation_ids: tuple[str, ...] = ()

    @property
    def is_available(self) -> bool:
        return self.state in SUCCESSFUL_CAPABILITY_STATES


class GenerationRoutePrefix(DiscoveryContract):
    """A verified route option without coordinates, values, or verdict evidence."""

    route_memory_artifact_id: str
    capability: str
    semantic_instruction: str
    source_node_id: str
    destination_node_id: str
    capability_ids: tuple[str, ...] = Field(min_length=1)
    segment_count: int = Field(gt=0)


REPORTED_ONLY_FIELDS = {"reached_capabilities", "known_from_other_device_shapes"}
"""Fields a run reported about itself, which authorize nothing."""


class ScenarioGenerationEvidence(DiscoveryContract):
    """The only app evidence semantic generation may consume."""

    app_map_artifact_id: str
    app_map_integrity_sha256: str
    capabilities: tuple[GenerationCapabilityEvidence, ...]
    route_prefixes: tuple[GenerationRoutePrefix, ...] = ()
    reached_capabilities: tuple[str, ...] = ()
    """What the run reached that no requirement asked for.

    Reported, never generated from. A reachable screen is not a product rule
    that it should be reachable, so these cannot authorize a scenario. They
    exist so a run that misses its goal says how far it got.
    """

    known_from_other_device_shapes: int = 0
    """How many stored maps already describe this app on another screen shape.

    Nothing is reused from them, because a fingerprint and a coordinate belong
    to one screen size. A non-zero count says this run re-learned an
    application something already knew.
    """

    @model_validator(mode="after")
    def _validate_unique_evidence(self) -> "ScenarioGenerationEvidence":
        capability_ids = [item.capability_id for item in self.capabilities]
        route_ids = [item.route_memory_artifact_id for item in self.route_prefixes]
        if len(capability_ids) != len(set(capability_ids)):
            raise ValueError("Generation capability identifiers must be unique.")
        if len(route_ids) != len(set(route_ids)):
            raise ValueError("Generation route identifiers must be unique.")
        return self

    @property
    def evidence_sha256(self) -> str:
        """Identify what authorizes generation, and nothing a run only reported.

        A lock records this digest and refuses knowledge that does not match.
        A reported field differs between the run that discovered a map and a
        later run that reused it. Including one would make the first run's lock
        unreadable by the second.
        """
        payload = json.dumps(
            self.model_dump(mode="json", exclude=REPORTED_ONLY_FIELDS),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    @property
    def available_capability_ids(self) -> frozenset[str]:
        return frozenset(item.capability_id for item in self.capabilities if item.is_available)

    @property
    def unavailable_required_capabilities(
        self,
    ) -> tuple[GenerationCapabilityEvidence, ...]:
        return tuple(item for item in self.capabilities if item.required and not item.is_available)

    @property
    def route_memory_ids(self) -> frozenset[str]:
        return frozenset(item.route_memory_artifact_id for item in self.route_prefixes)


def build_scenario_generation_evidence(
    app_map: AppMap,
    route_memories: tuple[RouteMemory, ...] = (),
    reached_capabilities: tuple[str, ...] = (),
    known_from_other_device_shapes: int = 0,
) -> ScenarioGenerationEvidence:
    """Project verified knowledge into a narrow, model-safe generation contract."""
    if not verify_artifact(app_map):
        raise ValueError("Scenario generation requires a sealed AppMap.")
    route_prefixes = []
    nodes_by_id = {node.node_id: node for node in app_map.nodes}
    for route_memory in route_memories:
        if not verify_artifact(route_memory):
            raise ValueError("Scenario generation requires sealed RouteMemory.")
        if route_memory.lifecycle is not RouteMemoryLifecycle.ACTIVE:
            raise ValueError("Scenario generation accepts only active RouteMemory.")
        if route_memory.source_app_map_artifact_id != app_map.header.artifact_id:
            raise ValueError("RouteMemory belongs to another AppMap.")
        if not route_memory.header.compatibility.is_compatible_with(app_map.header.compatibility):
            raise ValueError("RouteMemory is incompatible with its AppMap.")
        destination_node_id = route_memory.segments[-1].destination.node_id
        destination_observation_ids = set(nodes_by_id[destination_node_id].observation_ids)
        capability_ids = tuple(
            sorted(
                item.capability_id
                for item in app_map.capabilities
                if item.state in SUCCESSFUL_CAPABILITY_STATES
                and destination_observation_ids.intersection(item.evidence_observation_ids)
            )
        )
        if not capability_ids:
            raise ValueError("RouteMemory destination has no observed AppMap capability.")
        route_prefixes.append(
            GenerationRoutePrefix(
                route_memory_artifact_id=route_memory.header.artifact_id,
                capability=route_memory.capability,
                semantic_instruction=route_memory.semantic_instruction,
                source_node_id=route_memory.start.node_id,
                destination_node_id=destination_node_id,
                capability_ids=capability_ids,
                segment_count=len(route_memory.segments),
            )
        )
    capabilities = tuple(
        GenerationCapabilityEvidence(
            capability_id=item.capability_id,
            intent=item.intent,
            state=item.state,
            required=item.required,
            evidence_observation_ids=item.evidence_observation_ids,
        )
        for item in sorted(app_map.capabilities, key=lambda capability: capability.capability_id)
    )
    return ScenarioGenerationEvidence(
        app_map_artifact_id=app_map.header.artifact_id,
        app_map_integrity_sha256=app_map.header.integrity_sha256 or "",
        capabilities=capabilities,
        route_prefixes=tuple(
            sorted(route_prefixes, key=lambda item: item.route_memory_artifact_id)
        ),
        reached_capabilities=tuple(sorted(set(reached_capabilities))),
        known_from_other_device_shapes=known_from_other_device_shapes,
    )
