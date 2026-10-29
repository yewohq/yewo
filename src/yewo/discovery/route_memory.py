"""Compile, promote, and govern guarded RouteMemory artifacts."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import Literal

from pydantic import Field

from yewo.discovery.artifacts import (
    ArtifactProvenance,
    ArtifactReference,
    DiscoveryArtifactHeader,
    DiscoveryContract,
    seal_artifact,
    verify_artifact,
)
from yewo.discovery.contracts import SideEffectClass
from yewo.discovery.knowledge import (
    AppMap,
    AppMapEdge,
    AppMapNode,
    ControlReacquisition,
    RouteActionTemplate,
    RouteMemory,
    RouteMemoryCandidate,
    RouteMemoryLifecycle,
    RouteMemorySegment,
    RouteStateGuard,
)
from yewo.identity import derived_uuid, new_uuid

DEFAULT_ROUTE_SIDE_EFFECTS = frozenset({SideEffectClass.READ_ONLY, SideEffectClass.REVERSIBLE})


class RouteMemoryRequest(DiscoveryContract):
    """Requirement-owned capability request grounded by AppMap mechanics."""

    capability: str = Field(min_length=1)
    semantic_instruction: str = Field(min_length=1)
    source_node_id: str
    destination_node_id: str
    max_segments: int = Field(gt=0)
    allowed_side_effects: frozenset[SideEffectClass] = DEFAULT_ROUTE_SIDE_EFFECTS


class RouteMemoryPromotionReceipt(DiscoveryContract):
    """Post-run authority that may promote one compiled candidate."""

    receipt_id: str = Field(default_factory=new_uuid)
    source_candidate_artifact_id: str
    authority: Literal["post_run_policy"] = "post_run_policy"
    verified_segment_ids: tuple[str, ...] = Field(min_length=1)
    evidence_receipt_ids: tuple[str, ...] = Field(min_length=1)
    approved: Literal[True] = True


def compile_route_memory_candidate(
    app_map: AppMap,
    request: RouteMemoryRequest,
    *,
    forbidden_scenario_values: Sequence[str] = (),
) -> RouteMemoryCandidate:
    """Compile a shortest deterministic chain of independently verified edges."""
    if not verify_artifact(app_map):
        raise ValueError("RouteMemory compilation requires a sealed AppMap.")
    nodes_by_id = {node.node_id: node for node in app_map.nodes}
    if request.source_node_id not in nodes_by_id:
        raise ValueError("RouteMemory source node is absent from the AppMap.")
    if request.destination_node_id not in nodes_by_id:
        raise ValueError("RouteMemory destination node is absent from the AppMap.")
    path = _select_edge_path(app_map, request)
    if not path:
        raise ValueError("No policy-eligible AppMap path satisfies the RouteMemory request.")
    segments = tuple(
        _compile_segment(edge, nodes_by_id, app_map.header.artifact_id) for edge in path
    )
    source_ids = app_map.header.provenance.source_run_ids
    artifact_id = derived_uuid(
        "route-memory-candidate-v1",
        app_map.header.artifact_id,
        request.capability,
        request.semantic_instruction,
        request.source_node_id,
        request.destination_node_id,
        *(edge.transition_identity for edge in path),
    )
    header = DiscoveryArtifactHeader(
        artifact_id=artifact_id,
        schema_name="route-memory-candidate",
        compatibility=app_map.header.compatibility,
        provenance=ArtifactProvenance(
            source_run_ids=source_ids,
            sources=(
                ArtifactReference(
                    artifact_id=app_map.header.artifact_id,
                    sha256=app_map.header.integrity_sha256 or "",
                ),
            ),
        ),
        redaction=app_map.header.redaction,
    )
    candidate = RouteMemoryCandidate(
        header=header,
        source_app_map_artifact_id=app_map.header.artifact_id,
        capability=request.capability,
        semantic_instruction=request.semantic_instruction,
        start=segments[0].source,
        segments=segments,
    )
    return seal_artifact(
        candidate,
        forbidden_scenario_values=forbidden_scenario_values,
    )


def promote_route_memory(
    candidate: RouteMemoryCandidate,
    receipt: RouteMemoryPromotionReceipt,
) -> RouteMemory:
    if not verify_artifact(candidate):
        raise ValueError("RouteMemory promotion requires a sealed candidate.")
    if candidate.lifecycle is not RouteMemoryLifecycle.CANDIDATE:
        raise ValueError("Only an active RouteMemory candidate may be promoted.")
    if receipt.source_candidate_artifact_id != candidate.header.artifact_id:
        raise ValueError("The promotion receipt belongs to another candidate.")
    expected_segment_ids = tuple(segment.segment_id for segment in candidate.segments)
    if receipt.verified_segment_ids != expected_segment_ids:
        raise ValueError("RouteMemory promotion requires ordered verification of every segment.")
    expected_evidence_ids = tuple(
        receipt_id
        for segment in candidate.segments
        for receipt_id in (
            segment.authorization_receipt_id,
            segment.dispatch_receipt_id,
            segment.settlement_receipt_id,
        )
    )
    if receipt.evidence_receipt_ids != expected_evidence_ids:
        raise ValueError(
            "RouteMemory promotion requires every ordered authorization, dispatch, "
            "and settlement receipt."
        )
    artifact_id = derived_uuid(
        "route-memory-v1",
        candidate.header.artifact_id,
        receipt.receipt_id,
    )
    header = candidate.header.model_copy(
        update={
            "artifact_id": artifact_id,
            "schema_name": "route-memory",
            "provenance": ArtifactProvenance(
                source_run_ids=candidate.header.provenance.source_run_ids,
                sources=(
                    ArtifactReference(
                        artifact_id=candidate.header.artifact_id,
                        sha256=candidate.header.integrity_sha256 or "",
                    ),
                ),
            ),
            "integrity_sha256": None,
        }
    )
    return seal_artifact(
        RouteMemory(
            header=header,
            source_candidate_artifact_id=candidate.header.artifact_id,
            source_app_map_artifact_id=candidate.source_app_map_artifact_id,
            capability=candidate.capability,
            semantic_instruction=candidate.semantic_instruction,
            start=candidate.start,
            segments=candidate.segments,
        )
    )


def reject_route_memory_candidate(
    candidate: RouteMemoryCandidate,
    reason: str,
    *,
    authority: str,
) -> RouteMemoryCandidate:
    """Persist an immutable post-run rejection decision for one candidate."""
    if authority != "post_run_policy":
        raise ValueError("Only post-run policy may reject a RouteMemory candidate.")
    if not verify_artifact(candidate):
        raise ValueError("RouteMemory rejection requires a sealed candidate.")
    if candidate.lifecycle is not RouteMemoryLifecycle.CANDIDATE:
        raise ValueError("Only an active RouteMemory candidate may be rejected.")
    header = candidate.header.model_copy(
        update={
            "artifact_id": derived_uuid(
                "route-memory-candidate-rejection-v1",
                candidate.header.artifact_id,
                reason,
            ),
            "provenance": ArtifactProvenance(
                source_run_ids=candidate.header.provenance.source_run_ids,
                sources=(
                    ArtifactReference(
                        artifact_id=candidate.header.artifact_id,
                        sha256=candidate.header.integrity_sha256 or "",
                    ),
                ),
            ),
            "integrity_sha256": None,
        }
    )
    return seal_artifact(
        candidate.model_copy(
            update={
                "header": header,
                "lifecycle": RouteMemoryLifecycle.REJECTED,
                "lifecycle_reason": reason,
            }
        )
    )


def transition_route_memory_lifecycle(
    route_memory: RouteMemory,
    lifecycle: Literal[
        RouteMemoryLifecycle.ACTIVE,
        RouteMemoryLifecycle.SUSPECT,
        RouteMemoryLifecycle.INVALIDATED,
        RouteMemoryLifecycle.SUPERSEDED,
    ],
    reason: str,
    *,
    authority: str,
) -> RouteMemory:
    """Create a new lifecycle artifact; runtime quarantine cannot call this boundary."""
    if authority != "post_run_policy":
        raise ValueError("Only post-run policy may change RouteMemory lifecycle.")
    if not verify_artifact(route_memory):
        raise ValueError("RouteMemory lifecycle changes require a sealed artifact.")
    legal = {
        RouteMemoryLifecycle.ACTIVE: {
            RouteMemoryLifecycle.SUSPECT,
            RouteMemoryLifecycle.INVALIDATED,
            RouteMemoryLifecycle.SUPERSEDED,
        },
        RouteMemoryLifecycle.SUSPECT: {
            RouteMemoryLifecycle.ACTIVE,
            RouteMemoryLifecycle.INVALIDATED,
            RouteMemoryLifecycle.SUPERSEDED,
        },
    }
    if lifecycle not in legal.get(route_memory.lifecycle, set()):
        raise ValueError("The RouteMemory lifecycle transition is not legal.")
    header = route_memory.header.model_copy(
        update={
            "artifact_id": derived_uuid(
                "route-memory-lifecycle-v1",
                route_memory.header.artifact_id,
                lifecycle.value,
                reason,
            ),
            "provenance": ArtifactProvenance(
                source_run_ids=(route_memory.header.artifact_id,),
                sources=(
                    ArtifactReference(
                        artifact_id=route_memory.header.artifact_id,
                        sha256=route_memory.header.integrity_sha256 or "",
                    ),
                ),
            ),
            "integrity_sha256": None,
        }
    )
    return seal_artifact(
        route_memory.model_copy(
            update={
                "header": header,
                "lifecycle": lifecycle,
                "lifecycle_reason": reason,
            }
        )
    )


def _select_edge_path(
    app_map: AppMap,
    request: RouteMemoryRequest,
) -> tuple[AppMapEdge, ...]:
    adjacency: dict[str, list[AppMapEdge]] = {}
    for edge in app_map.edges:
        if edge.side_effect not in request.allowed_side_effects:
            continue
        if edge.side_effect is SideEffectClass.REVERSIBLE and not edge.reversible:
            continue
        adjacency.setdefault(edge.source_node_id, []).append(edge)
    for edges in adjacency.values():
        edges.sort(key=lambda item: item.transition_identity)
    queue: deque[tuple[str, tuple[AppMapEdge, ...]]] = deque([(request.source_node_id, ())])
    visited = {request.source_node_id}
    while queue:
        node_id, path = queue.popleft()
        for edge in adjacency.get(node_id, ()):
            candidate_path = (*path, edge)
            if len(candidate_path) > request.max_segments:
                continue
            if edge.destination_node_id == request.destination_node_id:
                return candidate_path
            if edge.destination_node_id not in visited:
                visited.add(edge.destination_node_id)
                queue.append((edge.destination_node_id, candidate_path))
    return ()


def _compile_segment(
    edge: AppMapEdge,
    nodes_by_id: dict[str, AppMapNode],
    app_map_artifact_id: str,
) -> RouteMemorySegment:
    return RouteMemorySegment(
        segment_id=derived_uuid(
            "route-memory-segment-v1",
            app_map_artifact_id,
            edge.transition_identity,
        ),
        source=_guard(nodes_by_id[edge.source_node_id]),
        action=RouteActionTemplate(
            action_kind=edge.action_kind,
            target=ControlReacquisition(
                control_identity=edge.control_identity,
                role=edge.control_role,
                text_digest=edge.control_text_digest,
                source_observation_id=edge.evidence_observation_ids[0],
            ),
        ),
        destination=_guard(nodes_by_id[edge.destination_node_id]),
        completion_predicate_id=f"app-map-node:{edge.destination_node_id}",
        side_effect=edge.side_effect,
        reversible=edge.reversible,
        authorization_receipt_id=edge.authorization_receipt_id,
        dispatch_receipt_id=edge.dispatch_receipt_id,
        settlement_receipt_id=edge.settlement_receipt_id,
        evidence_observation_ids=edge.evidence_observation_ids,
    )


def _guard(node: AppMapNode) -> RouteStateGuard:
    return RouteStateGuard(
        node_id=node.node_id,
        identity_scope=node.identity_scope,
        predicates=node.predicates,
    )
