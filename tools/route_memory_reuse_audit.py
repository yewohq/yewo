#!/usr/bin/env python3
"""Run isolated deterministic reuse audits over one sealed RouteMemory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from yewo.agent.actuation.actions import Coordinates, TapAction
from yewo.agent.actuation.safety import ActionAuthorization
from yewo.agent.memory.control_identity import build_control_identity
from yewo.agent.run.settlement import SettlementReceipt, SettlementStatus
from yewo.discovery.artifacts import parse_artifact
from yewo.discovery.contracts import DiscoveryObservation
from yewo.discovery.knowledge import RouteMemory, RouteStateGuard
from yewo.discovery.route_memory_execution import (
    RouteMemoryExecutionReceipt,
    RouteMemoryExecutor,
    RouteMemoryLiveObservation,
    RouteTargetResolution,
)
from yewo.discovery.runtime import DiscoveryDispatchReceipt

SCREEN_SIZE = (400, 800)


def _live_observation(
    guard: RouteStateGuard, scenario: str, sequence: int
) -> RouteMemoryLiveObservation:
    predicate = guard.predicates[0]
    return RouteMemoryLiveObservation(
        observation=DiscoveryObservation(
            observation_id=f"audit-{scenario}-observation-{sequence}",
            screen_state_id=predicate.screen_state_id,
            screen_family_candidate_id=None,
            phash=predicate.phash,
            keyboard_visible=predicate.keyboard_visible,
            overlay_visible=predicate.overlay_visible,
            foreground_app_id=predicate.foreground_app_id,
        ),
        screen_family_id=predicate.screen_family_id,
        screen_size=SCREEN_SIZE,
    )


class _ControlledRuntime:
    """Fresh scenario-local mechanics; it has no scenario values or verdict state."""

    def __init__(self, memory: RouteMemory, scenario: str):
        self.memory = memory
        self.scenario = scenario
        self.index = 0
        self.source: RouteMemoryLiveObservation | None = None

    def reacquire(self, target, action_kind, observation):
        if action_kind != "tap":
            return None
        self.source = observation
        action = TapAction(
            coordinates=Coordinates(x=50 + self.index, y=100 + self.index),
            reason="Controlled live reacquisition audit",
        )
        return RouteTargetResolution(
            source_observation_id=observation.observation.observation_id,
            historical_control_identity=target.control_identity,
            matched_role=target.role,
            matched_text_digest=target.text_digest,
            action=action,
            live_control_identity=build_control_identity(
                action, observation.observation.phash, observation.screen_size
            ),
        )

    def execute(
        self,
        authorization: ActionAuthorization,
        resolution: RouteTargetResolution,
    ) -> RouteMemoryExecutionReceipt:
        del resolution
        source = self.source
        if source is None:
            raise RuntimeError("Target reacquisition must precede execution.")
        segment = self.memory.segments[self.index]
        destination = _live_observation(segment.destination, self.scenario, self.index + 2)
        self.index += 1
        return RouteMemoryExecutionReceipt(
            dispatch=DiscoveryDispatchReceipt(
                dispatch_id=f"audit-{self.scenario}-dispatch-{self.index}",
                authorization_id=authorization.authorization_id,
                success=True,
            ),
            settlement=SettlementReceipt(
                settlement_id=f"audit-{self.scenario}-settlement-{self.index}",
                control_identity=authorization.control_identity,
                source_phash=source.observation.phash,
                destination_phash=destination.observation.phash,
                status=SettlementStatus.DURABLE_PROGRESS,
                reason="Controlled destination remained stable.",
                observation_ids=(
                    source.observation.observation_id,
                    destination.observation.observation_id,
                ),
            ),
            destination=destination,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-memory", type=Path, required=True)
    parser.add_argument("--scenario", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    memory = parse_artifact(args.route_memory.read_bytes(), RouteMemory)
    results = []
    for scenario in args.scenario:
        runtime = _ControlledRuntime(memory, scenario)
        result = RouteMemoryExecutor().execute(
            memory,
            memory.header.compatibility,
            _live_observation(memory.start, scenario, 1),
            runtime,
            context_id=f"audit-{scenario}",
        )
        results.append(
            {
                "scenario": scenario,
                "context_id": f"audit-{scenario}",
                "result": result.model_dump(mode="json"),
                "runtime_segment_count": runtime.index,
            }
        )
    payload = {
        "schema": "route-memory-reuse-audit-v1",
        "route_memory_artifact_id": memory.header.artifact_id,
        "scenario_count": len(results),
        "scenario_contexts_are_isolated": len({result["context_id"] for result in results})
        == len(results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.output)
    print(
        f"scenarios={len(results)} complete="
        f"{sum(item['result']['terminal'] == 'complete' for item in results)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
