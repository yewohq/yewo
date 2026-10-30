"""Narrow generation evidence is projected only from sealed app knowledge."""

import pytest

from yewo.discovery.artifacts import seal_artifact
from yewo.discovery.contracts import CapabilityState, SideEffectClass
from yewo.discovery.knowledge import (
    AppCapabilityEvidence,
    AppMap,
    AppMapEdge,
    AppMapNode,
    ScreenPredicate,
)
from yewo.discovery.route_memory import (
    RouteMemoryPromotionReceipt,
    RouteMemoryRequest,
    compile_route_memory_candidate,
    promote_route_memory,
)
from yewo.testing.generation_evidence import (
    GenerationCapabilityEvidence,
    ScenarioGenerationEvidence,
    build_scenario_generation_evidence,
)


def _app_map(header, artifact_id=None):
    source = AppMapNode(
        node_id="list",
        predicates=(ScreenPredicate(screen_state_id="screen-list", phash="0" * 64),),
        observation_ids=("observation-list",),
    )
    destination = AppMapNode(
        node_id="form",
        predicates=(ScreenPredicate(screen_state_id="screen-form", phash="f" * 64),),
        observation_ids=("observation-form",),
    )
    map_header = header("app-map")
    if artifact_id is not None:
        map_header = map_header.model_copy(update={"artifact_id": artifact_id})
    return seal_artifact(
        AppMap(
            header=map_header,
            source_discovery_artifact_ids=("discovery-1",),
            nodes=(source, destination),
            edges=(
                AppMapEdge(
                    transition_identity="transition-form",
                    source_node_id="list",
                    destination_node_id="form",
                    action_kind="tap",
                    control_identity="control-form",
                    control_role="button",
                    side_effect=SideEffectClass.READ_ONLY,
                    reversible=True,
                    authorization_receipt_id="authorization-form",
                    dispatch_receipt_id="dispatch-form",
                    settlement_receipt_id="settlement-form",
                    evidence_observation_ids=("observation-list", "observation-form"),
                ),
            ),
            capabilities=(
                AppCapabilityEvidence(
                    capability_id="capability-form",
                    intent="reach the schedule form",
                    state=CapabilityState.ROUTE_VERIFIED,
                    required=True,
                    evidence_observation_ids=("observation-form",),
                ),
            ),
        )
    )


def _route_memory(app_map):
    candidate = compile_route_memory_candidate(
        app_map,
        RouteMemoryRequest(
            capability="open_schedule_form",
            semantic_instruction="Open the schedule form",
            source_node_id="list",
            destination_node_id="form",
            max_segments=1,
        ),
    )
    evidence_ids = tuple(
        receipt_id
        for segment in candidate.segments
        for receipt_id in (
            segment.authorization_receipt_id,
            segment.dispatch_receipt_id,
            segment.settlement_receipt_id,
        )
    )
    return promote_route_memory(
        candidate,
        RouteMemoryPromotionReceipt(
            source_candidate_artifact_id=candidate.header.artifact_id,
            verified_segment_ids=tuple(segment.segment_id for segment in candidate.segments),
            evidence_receipt_ids=evidence_ids,
        ),
    )


def test_projection_contains_capability_and_route_meaning_but_no_coordinates(header):
    app_map = _app_map(header)
    route_memory = _route_memory(app_map)

    evidence = build_scenario_generation_evidence(app_map, (route_memory,))

    assert evidence.available_capability_ids == frozenset({"capability-form"})
    assert evidence.route_memory_ids == frozenset({route_memory.header.artifact_id})
    assert evidence.route_prefixes[0].segment_count == 1
    assert evidence.route_prefixes[0].semantic_instruction == "Open the schedule form"
    assert "coordinates" not in evidence.model_dump_json()


def test_projection_refuses_unsealed_or_foreign_route_memory(header):
    app_map = _app_map(header)
    route_memory = _route_memory(app_map)
    unsealed = route_memory.model_copy(
        update={"header": route_memory.header.model_copy(update={"integrity_sha256": None})}
    )
    other_map = _app_map(header, artifact_id="another-map")

    with pytest.raises(ValueError, match="sealed RouteMemory"):
        build_scenario_generation_evidence(app_map, (unsealed,))
    with pytest.raises(ValueError, match="another AppMap"):
        build_scenario_generation_evidence(other_map, (route_memory,))


def test_evidence_digest_changes_with_capability_terminal(header):
    available_map = _app_map(header)
    unavailable_map = seal_artifact(
        available_map.model_copy(
            update={
                "header": available_map.header.model_copy(
                    update={"artifact_id": "unavailable-map", "integrity_sha256": None}
                ),
                "capabilities": (
                    available_map.capabilities[0].model_copy(
                        update={
                            "state": CapabilityState.UNREACHABLE,
                            "evidence_observation_ids": (),
                        }
                    ),
                ),
            }
        )
    )

    available = build_scenario_generation_evidence(available_map)
    unavailable = build_scenario_generation_evidence(unavailable_map)

    assert available.evidence_sha256 != unavailable.evidence_sha256
    assert unavailable.unavailable_required_capabilities[0].capability_id == "capability-form"


def test_a_partial_run_says_how_far_it_got():
    """Otherwise a shortfall reads as having learned nothing."""
    from yewo.testing.generation_readiness import assess_generation_readiness
    from yewo.testing.models import UserJourney

    journey = UserJourney(
        id="checkout", name="Checkout", description="A user places an order."
    )
    evidence = ScenarioGenerationEvidence(
        app_map_artifact_id="map-1",
        app_map_integrity_sha256="a" * 64,
        capabilities=(
            GenerationCapabilityEvidence(
                capability_id="goal-1",
                intent="A user places an order.",
                state=CapabilityState.UNREACHABLE,
                required=True,
            ),
        ),
        reached_capabilities=("reach the screen headed Bag",),
        known_from_other_device_shapes=2,
    )

    readiness = assess_generation_readiness(journey, evidence, discovery_completed=True)

    assert not readiness.is_ready
    assert "The run did reach: reach the screen headed Bag." in readiness.reason
    assert "2 stored map(s) already describe this app" in readiness.reason


def test_a_reported_field_does_not_change_what_a_lock_records():
    """A lock refuses knowledge whose digest moved. A report must not move it."""
    base = ScenarioGenerationEvidence(
        app_map_artifact_id="map-1",
        app_map_integrity_sha256="a" * 64,
        capabilities=(),
    )
    reported = base.model_copy(
        update={
            "reached_capabilities": ("reach the screen headed Bag",),
            "known_from_other_device_shapes": 3,
        }
    )

    assert reported.evidence_sha256 == base.evidence_sha256
