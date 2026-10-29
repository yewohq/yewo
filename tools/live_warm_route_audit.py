#!/usr/bin/env python3
"""Execute sealed RouteMemory twice through the real device adapter."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from yewo.agent.device.launch import SetupMode
from yewo.agent.device.platform import Platform
from yewo.agent.device.session import DeviceSession
from yewo.agent.grounding import GROUNDING_MARKS
from yewo.agent.perception.blind import for_grounding
from yewo.discovery.artifacts import parse_artifact
from yewo.discovery.knowledge import AppMap, RouteMemory
from yewo.testing.lifecycle import SetupOrigin, apply_app_precondition
from yewo.testing.live_route_runtime import PerceptionRouteRuntimeProvider
from yewo.testing.warm_execution import VerifiedScenarioRoutePreparer


@dataclass
class _Scenario:
    scenario_uuid: str
    route_memory_ids: list[str]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="Pixel_8")
    parser.add_argument("--app", type=Path, required=True)
    parser.add_argument("--app-map", type=Path, required=True)
    parser.add_argument("--route-memory", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    app_map = parse_artifact(args.app_map.read_bytes(), AppMap)
    route_memories = tuple(
        parse_artifact(path.read_bytes(), RouteMemory) for path in args.route_memory
    )
    session = DeviceSession(
        device_name=args.device,
        app_path=str(args.app.resolve()),
        platform=Platform.ANDROID,
        session_dir=args.output.parent,
        keep_app=True,
    )
    rows = []
    try:
        device = session.start()
        perception = for_grounding(GROUNDING_MARKS, session)
        preparer = VerifiedScenarioRoutePreparer(
            app_map,
            route_memories,
            app_map.header.compatibility,
            PerceptionRouteRuntimeProvider(),
        )
        for sequence in (1, 2):
            apply_app_precondition(
                device,
                SetupMode.CLEAR_DATA,
                SetupOrigin.DEFAULT,
            )
            perception.reset()
            result = preparer.prepare(
                _Scenario(
                    scenario_uuid=f"live-scenario-{sequence}",
                    route_memory_ids=[memory.header.artifact_id for memory in route_memories],
                ),
                device,
                perception,
            )
            rows.append(result.model_dump(mode="json"))
            if result.terminal.value != "ready":
                break
        payload = {
            "schema": "live-warm-route-audit-v1",
            "model_calls": 0,
            "app_map_artifact_id": app_map.header.artifact_id,
            "route_memory_ids": [memory.header.artifact_id for memory in route_memories],
            "scenarios": rows,
            "scenario_2_reused_without_rediscovery": (
                len(rows) == 2
                and rows[1]["terminal"] == "ready"
                and rows[1]["local_rediscovery"] is None
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(args.output)
        print(
            f"scenarios={len(rows)} ready="
            f"{sum(row['terminal'] == 'ready' for row in rows)} model_calls=0"
        )
        return 0 if payload["scenario_2_reused_without_rediscovery"] else 1
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
