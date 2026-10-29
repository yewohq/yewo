#!/usr/bin/env python3
"""Compile and persist one guarded RouteMemory from a sealed AppMap."""

from __future__ import annotations

import argparse
from pathlib import Path

from yewo.discovery.artifacts import parse_artifact
from yewo.discovery.knowledge import AppMap
from yewo.discovery.route_memory import (
    RouteMemoryPromotionReceipt,
    RouteMemoryRequest,
    compile_route_memory_candidate,
    promote_route_memory,
)
from yewo.discovery.store import DiscoveryStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-map", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--capability", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--source-node", required=True)
    parser.add_argument("--destination-node", required=True)
    parser.add_argument("--max-segments", type=int, required=True)
    args = parser.parse_args()

    app_map = parse_artifact(args.app_map.read_bytes(), AppMap)
    candidate = compile_route_memory_candidate(
        app_map,
        RouteMemoryRequest(
            capability=args.capability,
            semantic_instruction=args.instruction,
            source_node_id=args.source_node,
            destination_node_id=args.destination_node,
            max_segments=args.max_segments,
        ),
    )
    memory = promote_route_memory(
        candidate,
        RouteMemoryPromotionReceipt(
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
        ),
    )
    store = DiscoveryStore(args.output_root)
    candidate_path = store.save_route_memory_candidate(candidate)
    memory_path = store.save_route_memory(memory)
    print(candidate_path)
    print(memory_path)
    print(f"segments={len(memory.segments)} capability={memory.capability}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
