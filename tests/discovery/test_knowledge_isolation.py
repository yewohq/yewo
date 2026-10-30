"""App knowledge may cross scenarios; scenario evidence may not."""

import pytest
from pydantic import ValidationError

from yewo.discovery.artifacts import (
    parse_artifact,
    seal_artifact,
    serialize_artifact,
    verify_artifact,
)
from yewo.discovery.knowledge import (
    AppMap,
    AppMapEdge,
    AppMapNode,
    ControlReacquisition,
    RouteActionTemplate,
    RouteMemoryCandidate,
    RouteMemorySegment,
    RouteStateGuard,
    ScreenPredicate,
)
from yewo.discovery.contracts import SideEffectClass
from yewo.testing.lock_file import Lock


def _screen(identifier, phash):
    return ScreenPredicate(screen_state_id=identifier, phash=phash)


def _route(header, instruction="Reach the schedule editor"):
    source = RouteStateGuard(
        node_id="node-list",
        identity_scope="exact_state",
        predicates=(_screen("screen-list", "0" * 64),),
    )
    destination = RouteStateGuard(
        node_id="node-editor",
        identity_scope="exact_state",
        predicates=(_screen("screen-editor", "1" * 64),),
    )
    segment = RouteMemorySegment(
        segment_id="segment-1",
        source=source,
        action=RouteActionTemplate(
            action_kind="tap",
            target=ControlReacquisition(
                control_identity="control-edit",
                role="button",
                text_digest="digest",
                source_observation_id="observation-1",
            ),
        ),
        destination=destination,
        completion_predicate_id="app-map-node:node-editor",
        side_effect=SideEffectClass.READ_ONLY,
        reversible=True,
        authorization_receipt_id="authorization-1",
        dispatch_receipt_id="dispatch-1",
        settlement_receipt_id="settlement-1",
        evidence_observation_ids=("observation-1", "observation-2"),
    )
    return RouteMemoryCandidate(
        header=header("route-memory-candidate"),
        source_app_map_artifact_id="app-map-1",
        capability="reach schedule editor",
        semantic_instruction=instruction,
        start=source,
        segments=(segment,),
    )


def test_route_candidate_has_no_coordinate_or_scenario_field(header):
    payload = _route(header).model_dump(mode="json")
    serialized = str(payload)

    for forbidden in ("coordinates", "scenario_uuid", "verdict", "obligations"):
        assert forbidden not in serialized
    with pytest.raises(ValidationError):
        RouteActionTemplate.model_validate(
            {
                **payload["segments"][0]["action"],
                "coordinates": {"x": 1, "y": 2},
            }
        )


def test_route_candidate_rejects_wrong_schema_or_start(header):
    route = _route(header)
    wrong_schema = route.model_dump(mode="json")
    wrong_schema["header"]["schema_name"] = "app-map"
    wrong_start = route.model_dump(mode="json")
    wrong_start["start"]["predicates"][0]["screen_state_id"] = "different"

    with pytest.raises(ValidationError, match="own schema"):
        RouteMemoryCandidate.model_validate(wrong_schema)
    with pytest.raises(ValidationError, match="route start"):
        RouteMemoryCandidate.model_validate(wrong_start)


def test_route_candidate_rejects_duplicate_or_disconnected_segments(header):
    route = _route(header)
    duplicate = route.model_dump(mode="json")
    duplicate["segments"].append(duplicate["segments"][0])
    disconnected = route.model_dump(mode="json")
    second = dict(disconnected["segments"][0])
    second["segment_id"] = "segment-2"
    second["source"] = RouteStateGuard(
        node_id="unrelated",
        identity_scope="exact_state",
        predicates=(_screen("unrelated", "2" * 64),),
    ).model_dump(mode="json")
    disconnected["segments"].append(second)

    with pytest.raises(ValidationError, match="identifiers must be unique"):
        RouteMemoryCandidate.model_validate(duplicate)
    with pytest.raises(ValidationError, match="contiguous"):
        RouteMemoryCandidate.model_validate(disconnected)


def test_shared_knowledge_guard_refuses_scenario_values(header):
    route = _route(header, instruction="Use private-probe-value")

    with pytest.raises(ValueError, match="scenario value"):
        seal_artifact(route, scenario_owned_values=("private-probe-value",))


def test_sealed_route_detects_later_mutation(header):
    route = seal_artifact(_route(header))
    changed = route.model_copy(update={"semantic_instruction": "A changed route"})

    assert verify_artifact(route)
    assert not verify_artifact(changed)


def test_persisted_route_requires_valid_schema_and_integrity(header):
    route = seal_artifact(_route(header))
    content = serialize_artifact(route)

    assert parse_artifact(content, RouteMemoryCandidate) == route
    with pytest.raises(ValueError, match="integrity"):
        serialize_artifact(_route(header))
    with pytest.raises(ValueError, match="integrity"):
        parse_artifact(content.replace(b"schedule editor", b"changed editor"), RouteMemoryCandidate)


def test_app_map_shares_navigation_without_scenario_outcomes(header):
    route = seal_artifact(_route(header))
    source = route.start
    destination = route.segments[0].destination
    app_map = AppMap(
        header=header("app-map"),
        source_discovery_artifact_ids=("discovery-1",),
        nodes=(
            AppMapNode(
                node_id="node-list",
                predicates=source.predicates,
                observation_ids=("observation-1",),
            ),
            AppMapNode(
                node_id="node-editor",
                predicates=destination.predicates,
                observation_ids=("observation-2",),
            ),
        ),
        edges=(
            AppMapEdge(
                transition_identity="transition-1",
                source_node_id="node-list",
                destination_node_id="node-editor",
                action_kind="tap",
                control_identity="control-edit",
                control_role="button",
                control_text_digest="digest",
                side_effect=SideEffectClass.READ_ONLY,
                reversible=True,
                authorization_receipt_id="authorization-1",
                dispatch_receipt_id="dispatch-1",
                settlement_receipt_id="settlement-1",
                evidence_observation_ids=("observation-1", "observation-2"),
            ),
        ),
        route_memory_candidate_ids=(route.header.artifact_id,),
    )
    sealed = seal_artifact(
        app_map,
        scenario_owned_values=("one-exercise", "one-hundred-exercises", "-100"),
    )

    assert verify_artifact(sealed)
    payload = sealed.model_dump(mode="json")
    assert payload["source_scenario_evidence_artifact_ids"] == []
    assert "verdict" not in payload
    assert "expected_outcomes" not in payload


def test_app_map_rejects_edges_to_unknown_nodes(header):
    with pytest.raises(ValidationError, match="same map"):
        AppMap(
            header=header("app-map"),
            source_discovery_artifact_ids=("discovery-1",),
            nodes=(
                AppMapNode(
                    node_id="node-list",
                    predicates=(_screen("screen-list", "0" * 64),),
                    observation_ids=("observation-1",),
                ),
            ),
            edges=(
                AppMapEdge(
                    transition_identity="transition-1",
                    source_node_id="node-list",
                    destination_node_id="missing",
                    action_kind="tap",
                    control_identity="control-edit",
                    control_role="button",
                    side_effect=SideEffectClass.READ_ONLY,
                    reversible=True,
                    authorization_receipt_id="authorization-1",
                    dispatch_receipt_id="dispatch-1",
                    settlement_receipt_id="settlement-1",
                    evidence_observation_ids=("observation-1", "observation-2"),
                ),
            ),
        )


@pytest.mark.parametrize(
    "field,error",
    [
        ("nodes", "node identifiers"),
        ("edges", "transition identities"),
        ("source_discovery_artifact_ids", "source discovery artifacts"),
        ("route_memory_candidate_ids", "RouteMemory candidate identifiers"),
    ],
)
def test_app_map_rejects_duplicate_graph_identities(header, field, error):
    route = seal_artifact(_route(header))
    app_map = AppMap(
        header=header("app-map"),
        source_discovery_artifact_ids=("discovery-1",),
        nodes=(
            AppMapNode(
                node_id="node-1",
                predicates=route.start.predicates,
                observation_ids=("observation-1",),
            ),
        ),
        route_memory_candidate_ids=(route.header.artifact_id,),
    )
    payload = app_map.model_dump(mode="json")
    if field in ("nodes", "edges"):
        if field == "edges":
            edge = AppMapEdge(
                transition_identity="transition-1",
                source_node_id="node-1",
                destination_node_id="node-1",
                action_kind="tap",
                control_identity="control-1",
                control_role="button",
                side_effect=SideEffectClass.READ_ONLY,
                reversible=True,
                authorization_receipt_id="authorization-1",
                dispatch_receipt_id="dispatch-1",
                settlement_receipt_id="settlement-1",
                evidence_observation_ids=("observation-1", "observation-2"),
            ).model_dump(mode="json")
            payload[field] = [edge, edge]
        else:
            payload[field].append(payload[field][0])
    else:
        payload[field].append(payload[field][0])

    with pytest.raises(ValidationError, match=error):
        AppMap.model_validate(payload)


def test_app_map_rejects_wrong_schema_owner(header):
    route = _route(header)
    app_map = AppMap(
        header=header("app-map"),
        source_discovery_artifact_ids=("discovery-1",),
        nodes=(
            AppMapNode(
                node_id="node-1",
                predicates=route.start.predicates,
                observation_ids=("observation-1",),
            ),
        ),
    )
    payload = app_map.model_dump(mode="json")
    payload["header"]["schema_name"] = "discovery-run"

    with pytest.raises(ValidationError, match="app-map schema"):
        AppMap.model_validate(payload)


def test_discovery_cannot_be_added_to_the_product_lock():
    with pytest.raises(ValidationError):
        Lock.model_validate({"entries": [], "discovery": {"capability": "editor"}})
