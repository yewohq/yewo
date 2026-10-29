#!/usr/bin/env python3
"""Build one reproducible AppMap from sealed discovery evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from yewo.discovery.artifacts import parse_artifact, serialize_artifact
from yewo.discovery.contracts import DiscoveryRun
from yewo.discovery.map_builder import build_app_map


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = tuple(parse_artifact(path.read_bytes(), DiscoveryRun) for path in args.discovery)
    app_map = build_app_map(runs)
    content = serialize_artifact(app_map)
    if serialize_artifact(build_app_map(tuple(reversed(runs)))) != content:
        raise RuntimeError("AppMap projection changed when source order changed.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(content)
    print(args.output)
    print(f"nodes={len(app_map.nodes)} edges={len(app_map.edges)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
