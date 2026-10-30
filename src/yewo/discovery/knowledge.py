"""Shared AppMap and guarded RouteMemory contracts."""

from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from yewo.discovery.artifacts import (
    DiscoveryArtifact,
    DiscoveryArtifactHeader,
    DiscoveryContract,
)
from yewo.discovery.contracts import CapabilityState, SideEffectClass


class ScreenPredicate(DiscoveryContract):
    """A secret-safe observed state predicate with no planner-owned screen name."""

    screen_state_id: str
    screen_family_id: str | None = None
    phash: str
    keyboard_visible: bool | None = None
    overlay_visible: bool | None = None
    foreground_app_id: str | None = None


class ControlReacquisition(DiscoveryContract):
    """Evidence needed to reacquire a target live, without stored coordinates."""

    control_identity: str
    role: str
    text_digest: str = ""
    source_observation_id: str


class RouteActionTemplate(DiscoveryContract):
    """Semantic action mechanics without live scenario values or coordinates."""

    action_kind: str
    target: ControlReacquisition


class RouteStateGuard(DiscoveryContract):
    """Accepted exact-state or calibrated-family predicates for one map node."""

    node_id: str
    identity_scope: Literal["exact_state", "calibrated_family"]
    predicates: tuple[ScreenPredicate, ...] = Field(min_length=1)


class RouteMemorySegment(DiscoveryContract):
    """One independently evidenced local transition, never an opaque macro."""

    segment_id: str
    source: RouteStateGuard
    action: RouteActionTemplate
    destination: RouteStateGuard
    completion_predicate_id: str
    side_effect: SideEffectClass
    reversible: bool
    reset_requirement: str | None = None
    authorization_receipt_id: str
    dispatch_receipt_id: str
    settlement_receipt_id: str
    evidence_observation_ids: tuple[str, ...] = Field(min_length=2)


class RouteMemoryLifecycle(str, Enum):
    CANDIDATE = "candidate"
    ACTIVE = "active"
    SUSPECT = "suspect"
    INVALIDATED = "invalidated"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class RouteMemoryCandidate(DiscoveryArtifact):
    """Unpromoted app-level route knowledge with local evidence per segment."""

    header: DiscoveryArtifactHeader
    lifecycle: Literal[
        RouteMemoryLifecycle.CANDIDATE,
        RouteMemoryLifecycle.REJECTED,
    ] = RouteMemoryLifecycle.CANDIDATE
    source_app_map_artifact_id: str
    capability: str
    semantic_instruction: str
    start: RouteStateGuard
    segments: tuple[RouteMemorySegment, ...] = Field(min_length=1)
    lifecycle_reason: str = ""

    @model_validator(mode="after")
    def _validate_route(self) -> "RouteMemoryCandidate":
        if self.header.schema_name != "route-memory-candidate":
            raise ValueError("A RouteMemory candidate requires its own schema.")
        if self.segments[0].source != self.start:
            raise ValueError("The first route segment must start at the route start predicate.")
        segment_ids = [segment.segment_id for segment in self.segments]
        if len(segment_ids) != len(set(segment_ids)):
            raise ValueError("Route segment identifiers must be unique.")
        for previous, current in zip(self.segments, self.segments[1:], strict=False):
            if previous.destination != current.source:
                raise ValueError("Route candidate segments must form a contiguous chain.")
        return self


class RouteMemory(DiscoveryArtifact):
    """Promoted reusable mechanics whose segments require live validation."""

    header: DiscoveryArtifactHeader
    lifecycle: Literal[
        RouteMemoryLifecycle.ACTIVE,
        RouteMemoryLifecycle.SUSPECT,
        RouteMemoryLifecycle.INVALIDATED,
        RouteMemoryLifecycle.SUPERSEDED,
    ] = RouteMemoryLifecycle.ACTIVE
    source_candidate_artifact_id: str
    source_app_map_artifact_id: str
    capability: str
    semantic_instruction: str
    start: RouteStateGuard
    segments: tuple[RouteMemorySegment, ...] = Field(min_length=1)
    lifecycle_reason: str = ""

    @model_validator(mode="after")
    def _validate_route_memory(self) -> "RouteMemory":
        if self.header.schema_name != "route-memory":
            raise ValueError("RouteMemory requires its own schema.")
        if self.segments[0].source != self.start:
            raise ValueError("RouteMemory must start at its declared start guard.")
        segment_ids = [segment.segment_id for segment in self.segments]
        if len(segment_ids) != len(set(segment_ids)):
            raise ValueError("RouteMemory segment identifiers must be unique.")
        for previous, current in zip(self.segments, self.segments[1:], strict=False):
            if previous.destination != current.source:
                raise ValueError("RouteMemory segments must form a contiguous chain.")
        return self


class AppMapNode(DiscoveryContract):
    """One exact state or calibrated family backed by accepted observations."""

    node_id: str
    identity_scope: Literal["exact_state", "calibrated_family"] = "exact_state"
    predicates: tuple[ScreenPredicate, ...] = Field(min_length=1)
    observation_ids: tuple[str, ...] = Field(min_length=1)
    capability_candidate_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_identity_scope(self) -> "AppMapNode":
        state_ids = {predicate.screen_state_id for predicate in self.predicates}
        family_ids = {predicate.screen_family_id for predicate in self.predicates}
        if self.identity_scope == "exact_state" and len(state_ids) != 1:
            raise ValueError("An exact AppMap node requires one screen state identity.")
        if self.identity_scope == "calibrated_family" and (
            None in family_ids or len(family_ids) != 1
        ):
            raise ValueError("A family AppMap node requires one calibrated family identity.")
        return self


class AppMapEdge(DiscoveryContract):
    """One observed transition variant without execution authority."""

    transition_identity: str
    source_node_id: str
    destination_node_id: str
    action_kind: str
    control_identity: str
    control_role: str
    control_text_digest: str = ""
    side_effect: SideEffectClass
    reversible: bool
    authorization_receipt_id: str
    dispatch_receipt_id: str
    settlement_receipt_id: str
    capability_candidate_ids: tuple[str, ...] = ()
    evidence_observation_ids: tuple[str, str]


class AppCapabilityEvidence(DiscoveryContract):
    """One journey goal and its terminal, non-verdict app evidence."""

    capability_id: str
    intent: str = Field(min_length=1)
    state: CapabilityState
    required: bool
    evidence_observation_ids: tuple[str, ...] = ()
    capability_candidate_ids: tuple[str, ...] = ()


class AppMap(DiscoveryArtifact):
    """Rebuildable non-verdict analytical view over immutable app evidence."""

    header: DiscoveryArtifactHeader
    source_discovery_artifact_ids: tuple[str, ...] = Field(min_length=1)
    source_scenario_evidence_artifact_ids: tuple[str, ...] = ()
    identity_calibration_artifact_id: str | None = None
    nodes: tuple[AppMapNode, ...] = Field(min_length=1)
    root_node_ids: tuple[str, ...] = ()
    edges: tuple[AppMapEdge, ...] = ()
    capabilities: tuple[AppCapabilityEvidence, ...] = ()
    route_memory_candidate_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_graph(self) -> "AppMap":
        if self.header.schema_name != "app-map":
            raise ValueError("An AppMap requires the app-map schema.")
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("AppMap node identifiers must be unique.")
        known_nodes = set(node_ids)
        if not set(self.root_node_ids).issubset(known_nodes):
            raise ValueError("Every AppMap root must name a node in the map.")
        if len(self.root_node_ids) != len(set(self.root_node_ids)):
            raise ValueError("AppMap root node identifiers must be unique.")
        nodes_by_id = {node.node_id: node for node in self.nodes}
        observation_ids = [
            observation_id for node in self.nodes for observation_id in node.observation_ids
        ]
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("An AppMap observation must belong to exactly one node.")
        transition_ids = [edge.transition_identity for edge in self.edges]
        if len(transition_ids) != len(set(transition_ids)):
            raise ValueError("AppMap transition identities must be unique.")
        capability_ids = [item.capability_id for item in self.capabilities]
        if len(capability_ids) != len(set(capability_ids)):
            raise ValueError("AppMap capability identifiers must be unique.")
        if len(self.source_discovery_artifact_ids) != len(set(self.source_discovery_artifact_ids)):
            raise ValueError("AppMap source discovery artifacts must be unique.")
        if len(self.source_scenario_evidence_artifact_ids) != len(
            set(self.source_scenario_evidence_artifact_ids)
        ):
            raise ValueError("AppMap source scenario artifacts must be unique.")
        if len(self.route_memory_candidate_ids) != len(set(self.route_memory_candidate_ids)):
            raise ValueError("AppMap RouteMemory candidate identifiers must be unique.")
        for edge in self.edges:
            if (
                edge.source_node_id not in known_nodes
                or edge.destination_node_id not in known_nodes
            ):
                raise ValueError("Every AppMap edge must connect nodes in the same map.")
            source_observation_id, destination_observation_id = edge.evidence_observation_ids
            if source_observation_id not in nodes_by_id[edge.source_node_id].observation_ids:
                raise ValueError("An AppMap edge requires evidence from its source node.")
            if (
                destination_observation_id
                not in nodes_by_id[edge.destination_node_id].observation_ids
            ):
                raise ValueError("An AppMap edge requires evidence from its destination node.")
        known_observations = set(observation_ids)
        for capability in self.capabilities:
            if not set(capability.evidence_observation_ids).issubset(known_observations):
                raise ValueError("An AppMap capability requires accepted observations.")
        return self
