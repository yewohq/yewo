"""One verified AppMap and RouteMemory pair, as discovery would have sealed it."""

from yewo.discovery.artifacts import seal_artifact
from yewo.discovery.contracts import CapabilityState
from yewo.discovery.control_surface import control_text_digest
from yewo.discovery.knowledge import (
    AppMap,
    AppCapabilityEvidence,
    AppMapNode,
    ControlReacquisition,
    RouteActionTemplate,
    RouteMemory,
    RouteMemorySegment,
    RouteStateGuard,
    ScreenPredicate,
)


def screen_predicate(name: str, phash: str) -> ScreenPredicate:
    return ScreenPredicate(
        screen_state_id=f"screen-{name}",
        phash=phash,
        keyboard_visible=False,
        overlay_visible=False,
        foreground_app_id="com.example.app",
    )


def build_verified_knowledge(header):
    source = screen_predicate("list", "0" * 64)
    destination = screen_predicate("form", "f" * 64)
    source_guard = RouteStateGuard(
        node_id="list",
        identity_scope="exact_state",
        predicates=(source,),
    )
    destination_guard = RouteStateGuard(
        node_id="form",
        identity_scope="exact_state",
        predicates=(destination,),
    )
    app_map = seal_artifact(
        AppMap(
            header=header("app-map"),
            source_discovery_artifact_ids=("discovery",),
            nodes=(
                AppMapNode(
                    node_id="list",
                    predicates=(source,),
                    observation_ids=("historical-list",),
                ),
                AppMapNode(
                    node_id="form",
                    predicates=(destination,),
                    observation_ids=("historical-form",),
                ),
            ),
            capabilities=(
                AppCapabilityEvidence(
                    capability_id="capability-form",
                    intent="reach the form",
                    state=CapabilityState.ROUTE_VERIFIED,
                    required=True,
                    evidence_observation_ids=("historical-form",),
                ),
            ),
        )
    )
    memory = seal_artifact(
        RouteMemory(
            header=header("route-memory"),
            source_candidate_artifact_id="candidate",
            source_app_map_artifact_id=app_map.header.artifact_id,
            capability="open_form",
            semantic_instruction="Open the form",
            start=source_guard,
            segments=(
                RouteMemorySegment(
                    segment_id="segment-open-form",
                    source=source_guard,
                    action=RouteActionTemplate(
                        action_kind="tap",
                        target=ControlReacquisition(
                            control_identity="historical-control",
                            role="button",
                            text_digest=control_text_digest("Open form"),
                            source_observation_id="historical-list",
                        ),
                    ),
                    destination=destination_guard,
                    completion_predicate_id="app-map-node:form",
                    side_effect="read_only",
                    reversible=True,
                    authorization_receipt_id="authorization",
                    dispatch_receipt_id="dispatch",
                    settlement_receipt_id="settlement",
                    evidence_observation_ids=("historical-list", "historical-form"),
                ),
            ),
        )
    )
    return app_map, memory
