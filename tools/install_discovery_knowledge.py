#!/usr/bin/env python3
"""Install one verified AppMap and its active RouteMemory artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from yewo.discovery.artifacts import parse_artifact
from yewo.discovery.knowledge import AppMap, RouteMemory
from yewo.discovery.store import DiscoveryStore
from yewo.testing.generation_evidence import build_scenario_generation_evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-map", type=Path, required=True)
    parser.add_argument("--route-memory", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    app_map = parse_artifact(args.app_map.read_bytes(), AppMap)
    route_memories = tuple(
        parse_artifact(path.read_bytes(), RouteMemory) for path in args.route_memory
    )
    evidence = build_scenario_generation_evidence(app_map, route_memories)
    store = DiscoveryStore(args.output_root)
    app_map_path = store.save_app_map(app_map)
    route_paths = tuple(store.save_route_memory(memory) for memory in route_memories)
    print(app_map_path)
    for path in route_paths:
        print(path)
    print(
        f"app_map={evidence.app_map_artifact_id} "
        f"routes={len(evidence.route_prefixes)} evidence={evidence.evidence_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
