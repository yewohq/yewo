"""Builders for verified scenario-generation boundaries used by runner tests."""

from yewo.discovery.contracts import CapabilityState
from yewo.testing.generation_evidence import (
    GenerationCapabilityEvidence,
    GenerationRoutePrefix,
    ScenarioGenerationEvidence,
)


def scenario_generation_evidence() -> ScenarioGenerationEvidence:
    return ScenarioGenerationEvidence(
        app_map_artifact_id="app-map-test",
        app_map_integrity_sha256="a" * 64,
        capabilities=(
            GenerationCapabilityEvidence(
                capability_id="capability-test",
                intent="reach the test subject",
                state=CapabilityState.ROUTE_VERIFIED,
                required=True,
                evidence_observation_ids=("observation-test",),
            ),
        ),
        route_prefixes=(
            GenerationRoutePrefix(
                route_memory_artifact_id="route-test",
                capability="open_test_subject",
                semantic_instruction="Open the test subject.",
                source_node_id="start",
                destination_node_id="subject",
                capability_ids=("capability-test",),
                segment_count=1,
            ),
        ),
    )


def bind_scenarios_to_evidence(scenarios, evidence: ScenarioGenerationEvidence):
    capability_ids = [
        item.capability_id for item in evidence.capabilities if item.is_available
    ]
    route_memory_ids = [
        item.route_memory_artifact_id for item in evidence.route_prefixes
    ]
    return [
        scenario.model_copy(
            update={
                "app_capability_ids": scenario.app_capability_ids or capability_ids,
                "route_memory_ids": scenario.route_memory_ids or route_memory_ids,
            }
        )
        for scenario in scenarios
    ]
