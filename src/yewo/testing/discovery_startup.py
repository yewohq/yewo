"""Compile bounded discovery evidence into scenario startup knowledge."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from yewo.discovery.artifacts import DiscoveryCompatibilityKey
from yewo.discovery.contracts import DiscoveryRun, SUCCESSFUL_CAPABILITY_STATES
from yewo.discovery.coordinator import DiscoveryCoordinator
from yewo.discovery.knowledge import AppMap, RouteMemory
from yewo.discovery.map_builder import build_app_map
from yewo.discovery.route_memory import (
    RouteMemoryPromotionReceipt,
    RouteMemoryRequest,
    compile_route_memory_candidate,
    promote_route_memory,
)
from yewo.discovery.runtime import DiscoveryProposer, DiscoveryRuntime
from yewo.discovery.store import DiscoveryStore
from yewo.testing.generation_evidence import build_scenario_generation_evidence
from yewo.testing.generation_startup import ScenarioStartupKnowledge
from yewo.testing.live_route_runtime import PerceptionRouteRuntimeProvider
from yewo.testing.models import UserJourney
from yewo.testing.warm_execution import VerifiedScenarioRoutePreparer


class DiscoveryStartupKnowledgeProvider:
    """Own discovery, knowledge compilation, and execution preparation."""

    def __init__(
        self,
        store: DiscoveryStore,
        compatibility: DiscoveryCompatibilityKey,
        run_factory: Callable[[], DiscoveryRun],
        runtime: DiscoveryRuntime,
        proposer: DiscoveryProposer,
    ):
        self.store = store
        self.compatibility = compatibility
        self.run_factory = run_factory
        self.runtime = runtime
        self.proposer = proposer

    def discover(self, journey: UserJourney) -> ScenarioStartupKnowledge | None:
        coordination = DiscoveryCoordinator(self.store).ensure_knowledge(
            self.compatibility,
            self.run_factory,
            self.runtime,
            self.proposer,
        )
        if coordination.app_map is not None:
            app_map = coordination.app_map
        else:
            discovery = coordination.discovery
            if discovery is None:
                raise AssertionError("Discovery coordination lost its knowledge source.")
            app_map = build_app_map((discovery.run,))
            self.store.save_app_map(app_map)
        route_memories = self._load_or_compile_routes(
            app_map,
            tuple(journey.values.values()),
        )
        evidence = build_scenario_generation_evidence(app_map, route_memories)
        return ScenarioStartupKnowledge(
            evidence,
            VerifiedScenarioRoutePreparer(
                app_map,
                route_memories,
                self.compatibility,
                PerceptionRouteRuntimeProvider(),
            ),
        )

    def _load_or_compile_routes(
        self,
        app_map: AppMap,
        forbidden_scenario_values: Sequence[str],
    ) -> tuple[RouteMemory, ...]:
        route_memories: list[RouteMemory] = []
        for capability in app_map.capabilities:
            if capability.state not in SUCCESSFUL_CAPABILITY_STATES:
                continue
            existing = self.store.find_active_route_memories(
                self.compatibility,
                capability.intent,
            )
            if existing:
                route_memories.extend(existing)
                continue
            route_memory = self._compile_route(
                app_map,
                capability.intent,
                capability.evidence_observation_ids,
                forbidden_scenario_values,
            )
            if route_memory is not None:
                route_memories.append(route_memory)
        return tuple(
            {memory.header.artifact_id: memory for memory in route_memories}.values()
        )

    def _compile_route(
        self,
        app_map: AppMap,
        capability_intent: str,
        evidence_observation_ids: tuple[str, ...],
        forbidden_scenario_values: Sequence[str],
    ) -> RouteMemory | None:
        destination_ids = {edge.destination_node_id for edge in app_map.edges}
        source_node_ids = sorted(
            node.node_id for node in app_map.nodes if node.node_id not in destination_ids
        )
        if not source_node_ids:
            raise ValueError("The AppMap has no discovery root node.")
        source_node_id = source_node_ids[0]
        destination_node_ids = sorted(
            node.node_id
            for node in app_map.nodes
            if set(node.observation_ids).intersection(evidence_observation_ids)
        )
        if not destination_node_ids or source_node_id in destination_node_ids:
            return None
        candidate = compile_route_memory_candidate(
            app_map,
            RouteMemoryRequest(
                capability=capability_intent,
                semantic_instruction=capability_intent,
                source_node_id=source_node_id,
                destination_node_id=destination_node_ids[0],
                max_segments=max(1, len(app_map.edges)),
            ),
            forbidden_scenario_values=forbidden_scenario_values,
        )
        self.store.save_route_memory_candidate(candidate)
        receipt = RouteMemoryPromotionReceipt(
            source_candidate_artifact_id=candidate.header.artifact_id,
            verified_segment_ids=tuple(segment.segment_id for segment in candidate.segments),
            evidence_receipt_ids=tuple(
                receipt_id
                for segment in candidate.segments
                for receipt_id in (
                    segment.authorization_receipt_id,
                    segment.dispatch_receipt_id,
                    segment.settlement_receipt_id,
                )
            ),
        )
        route_memory = promote_route_memory(candidate, receipt)
        self.store.save_route_memory(route_memory)
        return route_memory
