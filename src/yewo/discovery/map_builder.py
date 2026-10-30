"""Deterministic AppMap projection over immutable accepted evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from enum import Enum
from typing import TypeVar

from pydantic import Field, model_validator

from yewo.discovery.artifacts import (
    ArtifactProvenance,
    ArtifactRedaction,
    ArtifactReference,
    DiscoveryArtifact,
    DiscoveryArtifactHeader,
    DiscoveryCompatibilityKey,
    verify_artifact,
    seal_artifact,
)
from yewo.discovery.contracts import (
    CapabilityCandidate,
    JourneyGoal,
    DiscoveryObservation,
    DiscoveryRun,
    DiscoveryTransitionEvidence,
)
from yewo.discovery.identity_calibration import (
    IdentityCalibrationResult,
    ScreenFamilyClassificationReceipt,
    require_operational_calibration,
)
from yewo.discovery.knowledge import (
    AppCapabilityEvidence,
    AppMap,
    AppMapEdge,
    AppMapNode,
    ScreenPredicate,
)
from yewo.identity import derived_uuid


class ScenarioMapEvidence(DiscoveryArtifact):
    """Non-verdict scenario observations normalized for AppMap projection."""

    header: DiscoveryArtifactHeader
    observations: tuple[DiscoveryObservation, ...] = Field(min_length=1)
    transitions: tuple[DiscoveryTransitionEvidence, ...] = ()
    capability_candidates: tuple[CapabilityCandidate, ...] = ()

    @model_validator(mode="after")
    def _validate_evidence_graph(self) -> "ScenarioMapEvidence":
        if self.header.schema_name != "scenario-map-evidence":
            raise ValueError("Scenario map evidence requires its own schema.")
        observation_ids = [item.observation_id for item in self.observations]
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("Scenario map observation identifiers must be unique.")
        known_observations = set(observation_ids)
        transition_ids = [item.transition_identity for item in self.transitions]
        if len(transition_ids) != len(set(transition_ids)):
            raise ValueError("Scenario map transition identities must be unique.")
        for transition in self.transitions:
            references = {
                transition.source_observation_id,
                transition.destination_observation_id,
            }
            if not references.issubset(known_observations):
                raise ValueError("Scenario map transitions require exact observations.")
        for candidate in self.capability_candidates:
            if not set(candidate.observation_ids).issubset(known_observations):
                raise ValueError("Scenario capability evidence requires exact observations.")
        return self


class RevalidationScope(str, Enum):
    NONE = "none"
    EDGES = "edges"
    GRAPH = "graph"


class AppMapRevalidationPlan(DiscoveryArtifact):
    """Typed impact of interpreting one map under another compatibility key."""

    header: DiscoveryArtifactHeader
    source_app_map_artifact_id: str
    requested_compatibility: DiscoveryCompatibilityKey
    incompatible_dimensions: tuple[str, ...]
    scope: RevalidationScope
    affected_node_ids: tuple[str, ...] = ()
    affected_edge_ids: tuple[str, ...] = ()
    preserved_node_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_plan(self) -> "AppMapRevalidationPlan":
        if self.header.schema_name != "app-map-revalidation-plan":
            raise ValueError("A revalidation plan requires its own schema.")
        if self.scope is RevalidationScope.NONE and self.incompatible_dimensions:
            raise ValueError("A compatible map cannot require revalidation.")
        if self.scope is not RevalidationScope.NONE and not self.incompatible_dimensions:
            raise ValueError("Revalidation requires an incompatible dimension.")
        return self


EvidenceSource = DiscoveryRun | ScenarioMapEvidence
EvidenceItem = TypeVar("EvidenceItem")


def build_app_map(
    discovery_runs: tuple[DiscoveryRun, ...],
    scenario_evidence: tuple[ScenarioMapEvidence, ...] = (),
    *,
    family_classifications: tuple[ScreenFamilyClassificationReceipt, ...] = (),
    calibration: IdentityCalibrationResult | None = None,
) -> AppMap:
    """Project one reproducible map without changing any source artifact."""
    if not discovery_runs:
        raise ValueError("An AppMap requires at least one discovery artifact.")
    sources: tuple[EvidenceSource, ...] = (*discovery_runs, *scenario_evidence)
    _validate_sources(sources)
    compatibility = sources[0].header.compatibility
    observations = _deduplicate_evidence(
        (item for source in sources for item in source.observations),
        lambda item: item.observation_id,
    )
    transitions = _deduplicate_evidence(
        (item for source in sources for item in source.transitions),
        lambda item: item.transition_identity,
    )
    candidates = _deduplicate_evidence(
        (item for source in sources for item in source.capability_candidates),
        lambda item: item.candidate_id,
    )
    goals = _deduplicate_evidence(
        (item for run in discovery_runs for item in run.goals),
        lambda item: item.goal_id,
    )
    classifications = _validated_classifications(
        observations,
        family_classifications,
        calibration,
    )
    compatibility_digest = _compatibility_digest(compatibility)
    nodes, node_by_observation = _build_nodes(
        observations,
        candidates,
        classifications,
        compatibility_digest,
    )
    edges = _build_edges(
        transitions,
        candidates,
        node_by_observation,
        observations,
    )
    capabilities = _build_capabilities(goals, candidates)
    root_node_ids = tuple(
        sorted(
            {
                node_by_observation[run.observations[0].observation_id]
                for run in discovery_runs
                if run.observations
            }
        )
    )
    source_ids = tuple(sorted(source.header.artifact_id for source in sources))
    calibration_id = (
        calibration.artifact_id if classifications and calibration is not None else None
    )
    artifact_id = derived_uuid(
        "app-map-v1",
        compatibility_digest,
        *source_ids,
        calibration_id,
    )
    redaction_versions = {source.header.redaction.policy_version for source in sources}
    if len(redaction_versions) != 1:
        raise ValueError("AppMap sources require one redaction policy version.")
    header = DiscoveryArtifactHeader(
        artifact_id=artifact_id,
        schema_name="app-map",
        compatibility=compatibility,
        provenance=ArtifactProvenance(
            source_run_ids=source_ids,
            sources=tuple(
                ArtifactReference(
                    artifact_id=source.header.artifact_id,
                    sha256=source.header.integrity_sha256 or "",
                )
                for source in sorted(sources, key=lambda item: item.header.artifact_id)
            ),
        ),
        redaction=ArtifactRedaction(policy_version=redaction_versions.pop()),
    )
    app_map = AppMap(
        header=header,
        source_discovery_artifact_ids=tuple(
            sorted(run.header.artifact_id for run in discovery_runs)
        ),
        source_scenario_evidence_artifact_ids=tuple(
            sorted(item.header.artifact_id for item in scenario_evidence)
        ),
        identity_calibration_artifact_id=calibration_id,
        nodes=nodes,
        root_node_ids=root_node_ids,
        edges=edges,
        capabilities=capabilities,
    )
    return seal_artifact(app_map)


def plan_app_map_revalidation(
    app_map: AppMap,
    requested: DiscoveryCompatibilityKey,
) -> AppMapRevalidationPlan:
    if not verify_artifact(app_map):
        raise ValueError("Revalidation requires a sealed AppMap.")
    dimensions = app_map.header.compatibility.incompatibilities(requested)
    edge_only = {"discovery_policy_version", "control_identity_version"}
    if not dimensions:
        scope = RevalidationScope.NONE
        affected_nodes: tuple[str, ...] = ()
        affected_edges: tuple[str, ...] = ()
        preserved_nodes = tuple(sorted(node.node_id for node in app_map.nodes))
    elif set(dimensions).issubset(edge_only):
        scope = RevalidationScope.EDGES
        affected_nodes = ()
        affected_edges = tuple(sorted(edge.transition_identity for edge in app_map.edges))
        preserved_nodes = tuple(sorted(node.node_id for node in app_map.nodes))
    else:
        scope = RevalidationScope.GRAPH
        affected_nodes = tuple(sorted(node.node_id for node in app_map.nodes))
        affected_edges = tuple(sorted(edge.transition_identity for edge in app_map.edges))
        preserved_nodes = ()
    header = app_map.header.model_copy(
        update={
            "artifact_id": derived_uuid(
                "app-map-revalidation-v1",
                app_map.header.artifact_id,
                _compatibility_digest(requested),
            ),
            "schema_name": "app-map-revalidation-plan",
            "provenance": ArtifactProvenance(
                source_run_ids=(app_map.header.artifact_id,),
                sources=(
                    ArtifactReference(
                        artifact_id=app_map.header.artifact_id,
                        sha256=app_map.header.integrity_sha256 or "",
                    ),
                ),
            ),
            "integrity_sha256": None,
        }
    )
    return seal_artifact(
        AppMapRevalidationPlan(
            header=header,
            source_app_map_artifact_id=app_map.header.artifact_id,
            requested_compatibility=requested,
            incompatible_dimensions=dimensions,
            scope=scope,
            affected_node_ids=affected_nodes,
            affected_edge_ids=affected_edges,
            preserved_node_ids=preserved_nodes,
        )
    )


def _validate_sources(sources: tuple[EvidenceSource, ...]) -> None:
    compatibility = sources[0].header.compatibility
    for source in sources:
        if not verify_artifact(source):
            raise ValueError("AppMap construction requires sealed source evidence.")
        if not source.header.compatibility.is_compatible_with(compatibility):
            raise ValueError("AppMap source evidence has incompatible interpretation.")
        if isinstance(source, DiscoveryRun) and source.terminal is None:
            raise ValueError("AppMap construction requires terminal discovery evidence.")


def _deduplicate_evidence(
    items: Iterable[EvidenceItem],
    identity_of: Callable[[EvidenceItem], str],
) -> tuple[EvidenceItem, ...]:
    by_identity: dict[str, EvidenceItem] = {}
    for item in items:
        identity = identity_of(item)
        existing = by_identity.get(identity)
        if existing is not None and existing != item:
            raise ValueError(f"Conflicting immutable evidence for {identity}.")
        by_identity[identity] = item
    return tuple(by_identity[key] for key in sorted(by_identity))


def _validated_classifications(
    observations: tuple[DiscoveryObservation, ...],
    receipts: tuple[ScreenFamilyClassificationReceipt, ...],
    calibration: IdentityCalibrationResult | None,
) -> dict[str, ScreenFamilyClassificationReceipt]:
    if not receipts:
        return {}
    if calibration is None:
        raise ValueError("Family classification receipts require calibration evidence.")
    require_operational_calibration(calibration)
    known_observations = {item.observation_id for item in observations}
    by_observation = {}
    for receipt in receipts:
        if receipt.observation_id not in known_observations:
            raise ValueError("Family classification requires an accepted observation.")
        if receipt.calibration_artifact_id != calibration.artifact_id:
            raise ValueError("Family classification uses another calibration artifact.")
        if receipt.observation_id in by_observation:
            raise ValueError("An observation can have only one family classification.")
        by_observation[receipt.observation_id] = receipt
    return by_observation


def _build_nodes(observations, candidates, classifications, compatibility_digest):
    grouped = {}
    for observation in observations:
        classification = classifications.get(observation.observation_id)
        key = (
            (
                "family",
                classification.screen_family_id,
            )
            if classification is not None
            else ("state", observation.screen_state_id)
        )
        grouped.setdefault(key, []).append(observation)
    nodes = []
    node_by_observation = {}
    for key in sorted(grouped):
        group = tuple(sorted(grouped[key], key=lambda item: item.observation_id))
        family_id = key[1] if key[0] == "family" else None
        scope = "calibrated_family" if family_id is not None else "exact_state"
        node_id = derived_uuid("app-map-node-v1", compatibility_digest, *key)
        observation_ids = tuple(item.observation_id for item in group)
        candidate_ids = tuple(
            sorted(
                candidate.candidate_id
                for candidate in candidates
                if set(candidate.observation_ids).intersection(observation_ids)
            )
        )
        predicates = tuple(
            ScreenPredicate(
                screen_state_id=item.screen_state_id,
                screen_family_id=family_id,
                phash=item.phash,
                keyboard_visible=item.keyboard_visible,
                overlay_visible=item.overlay_visible,
                foreground_app_id=item.foreground_app_id,
            )
            for item in group
        )
        nodes.append(
            AppMapNode(
                node_id=node_id,
                identity_scope=scope,
                predicates=predicates,
                observation_ids=observation_ids,
                capability_candidate_ids=candidate_ids,
            )
        )
        node_by_observation.update(dict.fromkeys(observation_ids, node_id))
    return tuple(nodes), node_by_observation


def _build_edges(transitions, candidates, node_by_observation, observations):
    observations_by_id = {item.observation_id: item for item in observations}
    edges = []
    for transition in transitions:
        evidence_ids = (
            transition.source_observation_id,
            transition.destination_observation_id,
        )
        candidate_ids = tuple(
            sorted(
                candidate.candidate_id
                for candidate in candidates
                if set(candidate.observation_ids).intersection(evidence_ids)
            )
        )
        source_observation = observations_by_id[transition.source_observation_id]
        control = next(
            (
                item
                for item in source_observation.controls
                if item.control_identity == transition.control_identity
            ),
            None,
        )
        if control is None:
            raise ValueError("An AppMap edge requires its observed source control.")
        edges.append(
            AppMapEdge(
                transition_identity=transition.transition_identity,
                source_node_id=node_by_observation[transition.source_observation_id],
                destination_node_id=node_by_observation[transition.destination_observation_id],
                action_kind=transition.action_kind,
                control_identity=transition.control_identity,
                control_role=control.role,
                control_text_digest=control.text_digest,
                side_effect=transition.side_effect,
                reversible=transition.reversible,
                authorization_receipt_id=transition.authorization_receipt_id,
                dispatch_receipt_id=transition.dispatch_receipt_id,
                settlement_receipt_id=transition.settlement_receipt_id,
                capability_candidate_ids=candidate_ids,
                evidence_observation_ids=evidence_ids,
            )
        )
    return tuple(edges)


def _build_capabilities(
    goals: tuple[JourneyGoal, ...],
    candidates: tuple[CapabilityCandidate, ...],
) -> tuple[AppCapabilityEvidence, ...]:
    return tuple(
        AppCapabilityEvidence(
            capability_id=goal.goal_id,
            intent=goal.statement,
            state=goal.state,
            required=goal.required,
            evidence_observation_ids=goal.evidence_observation_ids,
            capability_candidate_ids=tuple(
                sorted(
                    candidate.candidate_id
                    for candidate in candidates
                    if goal.goal_id in candidate.goal_ids
                )
            ),
        )
        for goal in sorted(goals, key=lambda item: item.goal_id)
    )


def _compatibility_digest(compatibility: DiscoveryCompatibilityKey) -> str:
    payload = json.dumps(
        compatibility.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()
