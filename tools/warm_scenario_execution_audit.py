#!/usr/bin/env python3
"""Audit isolated scenario reuse of sealed AppMap and RouteMemory artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from yewo.agent.actuation.actions import Coordinates, TapAction
from yewo.agent.actuation.safety import ActionAuthorization
from yewo.agent.memory.control_identity import build_control_identity
from yewo.agent.run.settlement import SettlementReceipt, SettlementStatus
from yewo.discovery.artifacts import parse_artifact
from yewo.discovery.contracts import DiscoveryObservation
from yewo.discovery.knowledge import AppMap, RouteMemory, RouteStateGuard
from yewo.discovery.route_memory_execution import (
    RouteMemoryExecutionReceipt,
    RouteMemoryLiveObservation,
    RouteTargetResolution,
)
from yewo.discovery.runtime import DiscoveryDispatchReceipt
from yewo.testing.lock_file import read
from yewo.testing.warm_execution import WarmScenarioCoordinator

SCREEN_SIZE = (400, 800)


def _observation(
    guard: RouteStateGuard,
    scenario_uuid: str,
    sequence: int,
) -> RouteMemoryLiveObservation:
    predicate = guard.predicates[0]
    return RouteMemoryLiveObservation(
        observation=DiscoveryObservation(
            observation_id=f"{scenario_uuid}-observation-{sequence}",
            screen_state_id=predicate.screen_state_id,
            phash=predicate.phash,
            keyboard_visible=predicate.keyboard_visible,
            overlay_visible=predicate.overlay_visible,
            foreground_app_id=predicate.foreground_app_id,
        ),
        screen_family_id=predicate.screen_family_id,
        screen_size=SCREEN_SIZE,
    )


class _Runtime:
    def __init__(self, memory: RouteMemory, scenario_uuid: str):
        self.memory = memory
        self.scenario_uuid = scenario_uuid
        self.segment_index = 0
        self.source: RouteMemoryLiveObservation | None = None

    def reacquire(self, target, action_kind, observation):
        if action_kind != "tap":
            return None
        self.source = observation
        action = TapAction(
            coordinates=Coordinates(
                x=50 + self.segment_index,
                y=100 + self.segment_index,
            ),
            reason="Controlled fresh target reacquisition",
        )
        return RouteTargetResolution(
            source_observation_id=observation.observation.observation_id,
            historical_control_identity=target.control_identity,
            matched_role=target.role,
            matched_text_digest=target.text_digest,
            action=action,
            live_control_identity=build_control_identity(
                action,
                observation.observation.phash,
                observation.screen_size,
            ),
        )

    def execute(self, authorization: ActionAuthorization, resolution):
        del resolution
        source = self.source
        if source is None:
            raise RuntimeError("Target reacquisition must precede execution.")
        segment = self.memory.segments[self.segment_index]
        self.segment_index += 1
        destination = _observation(
            segment.destination,
            self.scenario_uuid,
            self.segment_index + 1,
        )
        return RouteMemoryExecutionReceipt(
            dispatch=DiscoveryDispatchReceipt(
                dispatch_id=f"{self.scenario_uuid}-dispatch-{self.segment_index}",
                authorization_id=authorization.authorization_id,
                success=True,
            ),
            settlement=SettlementReceipt(
                settlement_id=f"{self.scenario_uuid}-settlement-{self.segment_index}",
                control_identity=authorization.control_identity,
                source_phash=source.observation.phash,
                destination_phash=destination.observation.phash,
                status=SettlementStatus.DURABLE_PROGRESS,
                reason="The controlled destination settled.",
                observation_ids=(
                    source.observation.observation_id,
                    destination.observation.observation_id,
                ),
            ),
            destination=destination,
        )


@dataclass(frozen=True)
class _RuntimeContext:
    initial_observation: RouteMemoryLiveObservation
    runtime: _Runtime


class _RuntimeFactory:
    def __init__(self, scenario_uuid: str):
        self.scenario_uuid = scenario_uuid
        self.runtimes: list[_Runtime] = []

    def build(self, route_memory, *, context_id):
        if self.scenario_uuid not in context_id:
            raise RuntimeError("The route context does not belong to this scenario.")
        runtime = _Runtime(route_memory, self.scenario_uuid)
        self.runtimes.append(runtime)
        return _RuntimeContext(
            initial_observation=_observation(
                route_memory.start,
                self.scenario_uuid,
                1,
            ),
            runtime=runtime,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-map", type=Path, required=True)
    parser.add_argument("--route-memory", type=Path, action="append", required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    app_map = parse_artifact(args.app_map.read_bytes(), AppMap)
    memories = tuple(parse_artifact(path.read_bytes(), RouteMemory) for path in args.route_memory)
    locked = read(args.lock)
    if locked is None or len(locked.entries) < 2:
        raise RuntimeError("The audit requires at least two locked scenarios.")
    coordinator = WarmScenarioCoordinator(app_map, memories)
    rows = []
    for entry in locked.entries[:2]:
        scenario = entry.scenario()
        factory = _RuntimeFactory(scenario.scenario_uuid)
        scenario_state = {"entered_values": [], "verdicts": []}
        result = coordinator.prepare(
            scenario,
            app_map.header.compatibility,
            factory,
        )
        if not rows:
            scenario_state["entered_values"].append("scenario-1-owned-value")
            scenario_state["verdicts"].append("scenario-1-owned-verdict")
        rows.append(
            {
                "scenario_uuid": scenario.scenario_uuid,
                "cell": scenario.cell,
                "terminal": result.terminal.value,
                "route_memory_ids": list(result.route_memory_ids),
                "completed_segment_ids": [
                    segment_id
                    for route_result in result.route_results
                    for segment_id in route_result.completed_segment_ids
                ],
                "local_rediscovery": result.local_rediscovery is not None,
                "scenario_state_after_route": scenario_state,
                "route_cost": result.costs.route_reuse.model_dump(mode="json"),
            }
        )
    payload = {
        "schema": "warm-scenario-execution-audit-v1",
        "app_map_artifact_id": app_map.header.artifact_id,
        "route_memory_ids": [memory.header.artifact_id for memory in memories],
        "scenarios": rows,
        "scenario_contexts_are_isolated": len({row["scenario_uuid"] for row in rows}) == len(rows),
        "scenario_2_reused_without_rediscovery": (
            rows[1]["terminal"] == "ready" and not rows[1]["local_rediscovery"]
        ),
        "scenario_values_and_verdicts_are_isolated": (
            rows[0]["scenario_state_after_route"]
            == {
                "entered_values": ["scenario-1-owned-value"],
                "verdicts": ["scenario-1-owned-verdict"],
            }
            and rows[1]["scenario_state_after_route"] == {"entered_values": [], "verdicts": []}
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.output)
    print(
        "scenarios=2 ready="
        + str(sum(row["terminal"] == "ready" for row in rows))
        + " scenario_2_local_rediscovery="
        + str(rows[1]["local_rediscovery"]).lower()
    )
    return 0 if all(row["terminal"] == "ready" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
