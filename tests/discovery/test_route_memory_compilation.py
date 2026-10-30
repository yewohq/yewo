"""RouteMemory compilation and lifecycle authority."""

import pytest

from yewo.discovery.artifacts import seal_artifact, verify_artifact
from yewo.discovery.contracts import SideEffectClass
from yewo.discovery.knowledge import (
    AppMap,
    AppMapEdge,
    AppMapNode,
    RouteMemoryLifecycle,
    ScreenPredicate,
)
from yewo.discovery.route_memory import (
    RouteMemoryPromotionReceipt,
    RouteMemoryRequest,
    compile_route_memory_candidate,
    promote_route_memory,
    reject_route_memory_candidate,
    transition_route_memory_lifecycle,
)


def _node(identifier: str, observation_id: str, digit: str) -> AppMapNode:
    return AppMapNode(
        node_id=identifier,
        predicates=(
            ScreenPredicate(
                screen_state_id=f"screen-{identifier}",
                phash=digit * 64,
                keyboard_visible=False,
                foreground_app_id="com.example.app",
            ),
        ),
        observation_ids=(observation_id,),
    )


def _edge(
    number: int,
    source: str,
    destination: str,
    source_observation: str,
    destination_observation: str,
    *,
    side_effect: SideEffectClass = SideEffectClass.READ_ONLY,
    reversible: bool | None = None,
) -> AppMapEdge:
    return AppMapEdge(
        transition_identity=f"transition-{number}",
        source_node_id=source,
        destination_node_id=destination,
        action_kind="tap",
        control_identity=f"control-{number}",
        control_role="button",
        control_text_digest=f"digest-{number}",
        side_effect=side_effect,
        reversible=(
            side_effect
            in {
                SideEffectClass.READ_ONLY,
                SideEffectClass.REVERSIBLE,
            }
            if reversible is None
            else reversible
        ),
        authorization_receipt_id=f"authorization-{number}",
        dispatch_receipt_id=f"dispatch-{number}",
        settlement_receipt_id=f"settlement-{number}",
        evidence_observation_ids=(source_observation, destination_observation),
    )


def _app_map(header) -> AppMap:
    return seal_artifact(
        AppMap(
            header=header("app-map"),
            source_discovery_artifact_ids=("discovery-1",),
            nodes=(
                _node("list", "observation-1", "0"),
                _node("drawer", "observation-2", "1"),
                _node("form", "observation-3", "2"),
                _node("unsafe", "observation-4", "3"),
            ),
            edges=(
                _edge(1, "list", "drawer", "observation-1", "observation-2"),
                _edge(2, "drawer", "form", "observation-2", "observation-3"),
                _edge(
                    3,
                    "list",
                    "unsafe",
                    "observation-1",
                    "observation-4",
                    side_effect=SideEffectClass.DESTRUCTIVE,
                ),
                _edge(4, "unsafe", "form", "observation-4", "observation-3"),
                _edge(
                    5,
                    "list",
                    "form",
                    "observation-1",
                    "observation-3",
                    side_effect=SideEffectClass.REVERSIBLE,
                    reversible=False,
                ),
            ),
        )
    )


def _request(**updates) -> RouteMemoryRequest:
    values = {
        "capability": "open_schedule_entry_form",
        "semantic_instruction": "Open the schedule entry form",
        "source_node_id": "list",
        "destination_node_id": "form",
        "max_segments": 3,
    }
    values.update(updates)
    return RouteMemoryRequest(**values)


def _evidence_ids(candidate) -> tuple[str, ...]:
    return tuple(
        receipt_id
        for segment in candidate.segments
        for receipt_id in (
            segment.authorization_receipt_id,
            segment.dispatch_receipt_id,
            segment.settlement_receipt_id,
        )
    )


def test_compiler_selects_verified_policy_eligible_path_without_coordinates(header):
    app_map = _app_map(header)

    candidate = compile_route_memory_candidate(app_map, _request())

    assert verify_artifact(candidate)
    assert [segment.action.target.control_identity for segment in candidate.segments] == [
        "control-1",
        "control-2",
    ]
    assert candidate.start.node_id == "list"
    assert "coordinates" not in candidate.model_dump_json()
    assert candidate.header.provenance.sources[0].artifact_id == app_map.header.artifact_id


def test_compiler_is_deterministic_and_enforces_budget_and_side_effect_policy(header):
    app_map = _app_map(header)

    first = compile_route_memory_candidate(app_map, _request())
    second = compile_route_memory_candidate(app_map, _request())

    assert first == second
    with pytest.raises(ValueError, match="No policy-eligible"):
        compile_route_memory_candidate(app_map, _request(max_segments=1))
    with pytest.raises(ValueError, match="scenario value"):
        compile_route_memory_candidate(
            app_map,
            _request(semantic_instruction="Use private-probe-value"),
            scenario_owned_values=("private-probe-value",),
        )


def test_promotion_requires_post_run_verification_of_every_ordered_segment(header):
    candidate = compile_route_memory_candidate(_app_map(header), _request())
    segment_ids = tuple(segment.segment_id for segment in candidate.segments)
    wrong = RouteMemoryPromotionReceipt(
        source_candidate_artifact_id=candidate.header.artifact_id,
        verified_segment_ids=segment_ids[:1],
        evidence_receipt_ids=_evidence_ids(candidate),
    )

    with pytest.raises(ValueError, match="every segment"):
        promote_route_memory(candidate, wrong)
    with pytest.raises(ValueError, match="every ordered authorization"):
        promote_route_memory(
            candidate,
            RouteMemoryPromotionReceipt(
                source_candidate_artifact_id=candidate.header.artifact_id,
                verified_segment_ids=segment_ids,
                evidence_receipt_ids=("incomplete-receipt-chain",),
            ),
        )

    memory = promote_route_memory(
        candidate,
        RouteMemoryPromotionReceipt(
            source_candidate_artifact_id=candidate.header.artifact_id,
            verified_segment_ids=segment_ids,
            evidence_receipt_ids=_evidence_ids(candidate),
        ),
    )

    assert verify_artifact(memory)
    assert memory.lifecycle is RouteMemoryLifecycle.ACTIVE
    assert memory.segments == candidate.segments


def test_only_post_run_policy_can_create_a_new_lifecycle_artifact(header):
    candidate = compile_route_memory_candidate(_app_map(header), _request())
    memory = promote_route_memory(
        candidate,
        RouteMemoryPromotionReceipt(
            source_candidate_artifact_id=candidate.header.artifact_id,
            verified_segment_ids=tuple(segment.segment_id for segment in candidate.segments),
            evidence_receipt_ids=_evidence_ids(candidate),
        ),
    )

    suspect = transition_route_memory_lifecycle(
        memory,
        RouteMemoryLifecycle.SUSPECT,
        "The next shadow run must revalidate this route.",
        authority="post_run_policy",
    )

    assert verify_artifact(suspect)
    assert suspect.lifecycle is RouteMemoryLifecycle.SUSPECT
    assert suspect.header.artifact_id != memory.header.artifact_id
    with pytest.raises(ValueError, match="not legal"):
        transition_route_memory_lifecycle(
            memory,
            RouteMemoryLifecycle.ACTIVE,
            "No transition",
            authority="post_run_policy",
        )
    with pytest.raises(ValueError, match="Only post-run policy"):
        transition_route_memory_lifecycle(
            memory,
            RouteMemoryLifecycle.SUSPECT,
            "Runtime cannot mutate durable memory",
            authority="runtime",
        )


def test_candidate_rejection_is_immutable_and_cannot_be_promoted(header):
    candidate = compile_route_memory_candidate(_app_map(header), _request())

    rejected = reject_route_memory_candidate(
        candidate,
        "Shadow verification contradicted the route.",
        authority="post_run_policy",
    )

    assert rejected.lifecycle is RouteMemoryLifecycle.REJECTED
    assert verify_artifact(rejected)
    assert candidate.lifecycle is RouteMemoryLifecycle.CANDIDATE
    with pytest.raises(ValueError, match="active RouteMemory candidate"):
        promote_route_memory(
            rejected,
            RouteMemoryPromotionReceipt(
                source_candidate_artifact_id=rejected.header.artifact_id,
                verified_segment_ids=tuple(segment.segment_id for segment in rejected.segments),
                evidence_receipt_ids=_evidence_ids(rejected),
            ),
        )
    with pytest.raises(ValueError, match="Only post-run policy"):
        reject_route_memory_candidate(
            candidate,
            "Runtime rejection is forbidden.",
            authority="runtime",
        )
