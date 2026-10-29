import json

import pytest

from yewo.discovery.artifacts import seal_artifact
from yewo.discovery.knowledge import (
    AppMap,
    AppMapNode,
    ControlReacquisition,
    RouteActionTemplate,
    RouteMemory,
    RouteMemoryCandidate,
    RouteMemoryLifecycle,
    RouteMemorySegment,
    RouteStateGuard,
    ScreenPredicate,
)
from yewo.discovery.contracts import SideEffectClass
from yewo.discovery.store import DiscoveryStore


def _app_map(header):
    return seal_artifact(
        AppMap(
            header=header("app-map"),
            source_discovery_artifact_ids=("discovery-1",),
            nodes=(
                AppMapNode(
                    node_id="node-1",
                    predicates=(
                        ScreenPredicate(
                            screen_state_id="screen-1",
                            phash="0" * 16,
                        ),
                    ),
                    observation_ids=("observation-1",),
                ),
            ),
        )
    )


def _route_candidate(header):
    source = RouteStateGuard(
        node_id="node-1",
        identity_scope="exact_state",
        predicates=(ScreenPredicate(screen_state_id="screen-1", phash="0" * 16),),
    )
    destination = RouteStateGuard(
        node_id="node-2",
        identity_scope="exact_state",
        predicates=(ScreenPredicate(screen_state_id="screen-2", phash="f" * 16),),
    )
    return seal_artifact(
        RouteMemoryCandidate(
            header=header("route-memory-candidate"),
            source_app_map_artifact_id="app-map-1",
            capability="open editor",
            semantic_instruction="Open the editor",
            start=source,
            segments=(
                RouteMemorySegment(
                    segment_id="segment-1",
                    source=source,
                    action=RouteActionTemplate(
                        action_kind="tap",
                        target=ControlReacquisition(
                            control_identity="control-1",
                            role="button",
                            source_observation_id="observation-1",
                        ),
                    ),
                    destination=destination,
                    completion_predicate_id="app-map-node:node-2",
                    side_effect=SideEffectClass.READ_ONLY,
                    reversible=True,
                    authorization_receipt_id="authorization-1",
                    dispatch_receipt_id="dispatch-1",
                    settlement_receipt_id="settlement-1",
                    evidence_observation_ids=("observation-1", "observation-2"),
                ),
            ),
        )
    )


def _route_memory(header, *, lifecycle=RouteMemoryLifecycle.ACTIVE):
    candidate = _route_candidate(header)
    return seal_artifact(
        RouteMemory(
            header=header("route-memory"),
            lifecycle=lifecycle,
            source_candidate_artifact_id=candidate.header.artifact_id,
            source_app_map_artifact_id=candidate.source_app_map_artifact_id,
            capability=candidate.capability,
            semantic_instruction=candidate.semantic_instruction,
            start=candidate.start,
            segments=candidate.segments,
        )
    )


def test_store_round_trips_only_verified_compatible_app_maps(tmp_path, header, compatibility):
    store = DiscoveryStore(tmp_path)
    app_map = _app_map(header)

    path = store.save_app_map(app_map)

    assert path.parent.name == "app_maps"
    assert store.find_compatible_app_map(compatibility) == app_map


def test_store_refuses_tampered_app_map(tmp_path, header, compatibility):
    store = DiscoveryStore(tmp_path)
    path = store.save_app_map(_app_map(header))
    payload = json.loads(path.read_text())
    payload["nodes"][0]["predicates"][0]["phash"] = "f" * 16
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="integrity"):
        store.find_compatible_app_map(compatibility)


def test_store_refuses_ambiguous_compatible_maps(tmp_path, header, compatibility):
    store = DiscoveryStore(tmp_path)
    first = _app_map(header)
    second = _app_map(header)
    store.save_app_map(first)
    store.save_app_map(second)

    with pytest.raises(ValueError, match="explicit selection"):
        store.find_compatible_app_map(compatibility)


def test_store_returns_scoped_revalidation_for_incompatible_map(tmp_path, header, compatibility):
    store = DiscoveryStore(tmp_path)
    app_map = _app_map(header)
    store.save_app_map(app_map)
    changed = compatibility.model_copy(update={"discovery_policy_version": "discovery-policy-v2"})

    plans = store.plan_revalidation(changed)

    assert len(plans) == 1
    assert plans[0].source_app_map_artifact_id == app_map.header.artifact_id
    assert plans[0].scope.value == "edges"
    assert plans[0].preserved_node_ids == ("node-1",)


def test_store_round_trips_candidate_and_finds_only_active_compatible_memory(
    tmp_path, header, compatibility
):
    store = DiscoveryStore(tmp_path)
    candidate = _route_candidate(header)
    active = _route_memory(header)
    suspect = _route_memory(header, lifecycle=RouteMemoryLifecycle.SUSPECT)

    candidate_path = store.save_route_memory_candidate(candidate)
    active_path = store.save_route_memory(active)
    store.save_route_memory(suspect)

    assert candidate_path.parent.name == "route_memory_candidates"
    assert active_path.parent.name == "route_memories"
    assert store.find_active_route_memories(compatibility, "open editor") == (active,)
    assert store.find_active_route_memories(compatibility, "another capability") == ()


def test_store_refuses_tampered_route_memory(tmp_path, header, compatibility):
    store = DiscoveryStore(tmp_path)
    path = store.save_route_memory(_route_memory(header))
    payload = json.loads(path.read_text())
    payload["segments"][0]["action"]["target"]["role"] = "changed"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="integrity"):
        store.find_active_route_memories(compatibility, "open editor")


def test_store_loads_exact_knowledge_ids_for_warm_execution(tmp_path, header):
    store = DiscoveryStore(tmp_path)
    app_map = _app_map(header)
    memory = _route_memory(header)
    store.save_app_map(app_map)
    store.save_route_memory(memory)

    assert store.load_app_map(app_map.header.artifact_id) == app_map
    assert store.load_route_memories((memory.header.artifact_id,)) == (memory,)


def test_store_does_not_guess_when_exact_warm_knowledge_is_absent(tmp_path):
    store = DiscoveryStore(tmp_path)

    with pytest.raises(FileNotFoundError, match="AppMap missing-map"):
        store.load_app_map("missing-map")
    with pytest.raises(FileNotFoundError, match="RouteMemory missing-memory"):
        store.load_route_memories(("missing-memory",))
