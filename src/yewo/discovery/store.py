"""Verified durable storage for discovery runs and AppMaps."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from yewo.discovery.artifacts import (
    DiscoveryCompatibilityKey,
    parse_artifact,
    serialize_artifact,
)
from yewo.discovery.contracts import DiscoveryRun
from yewo.discovery.knowledge import (
    AppMap,
    RouteMemory,
    RouteMemoryCandidate,
    RouteMemoryLifecycle,
)
from yewo.discovery.map_builder import AppMapRevalidationPlan, plan_app_map_revalidation


class DiscoveryStore:
    """Expose only integrity-checked artifacts from one local store."""

    def __init__(self, root: Path):
        self.root = root

    def find_compatible_app_map(self, compatibility: DiscoveryCompatibilityKey) -> AppMap | None:
        maps = tuple(
            app_map
            for app_map in self._read_app_maps()
            if app_map.header.compatibility.is_compatible_with(compatibility)
        )
        if len(maps) > 1:
            raise ValueError("More than one compatible AppMap requires explicit selection.")
        return maps[0] if maps else None

    def save_discovery_run(self, run: DiscoveryRun) -> Path:
        if run.terminal is None:
            raise ValueError("Only terminal discovery runs may be persisted.")
        destination = self.root / "discovery_runs" / f"{run.header.artifact_id}.json"
        self._atomic_write(destination, serialize_artifact(run))
        return destination

    def save_app_map(self, app_map: AppMap) -> Path:
        destination = self.root / "app_maps" / f"{app_map.header.artifact_id}.json"
        self._atomic_write(destination, serialize_artifact(app_map))
        return destination

    def save_route_memory_candidate(self, candidate: RouteMemoryCandidate) -> Path:
        destination = self.root / "route_memory_candidates" / f"{candidate.header.artifact_id}.json"
        self._atomic_write(destination, serialize_artifact(candidate))
        return destination

    def save_route_memory(self, route_memory: RouteMemory) -> Path:
        destination = self.root / "route_memories" / f"{route_memory.header.artifact_id}.json"
        self._atomic_write(destination, serialize_artifact(route_memory))
        return destination

    def load_app_map(self, artifact_id: str) -> AppMap:
        """Load one exact sealed map; callers must not select by filename guesses."""
        path = self.root / "app_maps" / f"{artifact_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"AppMap {artifact_id} is absent from the knowledge store.")
        app_map = parse_artifact(path.read_bytes(), AppMap)
        if app_map.header.artifact_id != artifact_id:
            raise ValueError("The stored AppMap identity does not match its requested identity.")
        return app_map

    def load_route_memories(self, artifact_ids: tuple[str, ...]) -> tuple[RouteMemory, ...]:
        """Load an ordered exact set of sealed RouteMemory artifacts."""
        memories = []
        for artifact_id in artifact_ids:
            path = self.root / "route_memories" / f"{artifact_id}.json"
            if not path.is_file():
                raise FileNotFoundError(
                    f"RouteMemory {artifact_id} is absent from the knowledge store."
                )
            memory = parse_artifact(path.read_bytes(), RouteMemory)
            if memory.header.artifact_id != artifact_id:
                raise ValueError(
                    "The stored RouteMemory identity does not match its requested identity."
                )
            memories.append(memory)
        return tuple(memories)

    def find_active_route_memories(
        self,
        compatibility: DiscoveryCompatibilityKey,
        capability: str,
    ) -> tuple[RouteMemory, ...]:
        """Return only verified, compatible, active memory for one capability."""
        directory = self.root / "route_memories"
        if not directory.exists():
            return ()
        memories = (
            parse_artifact(path.read_bytes(), RouteMemory)
            for path in sorted(directory.glob("*.json"))
        )
        return tuple(
            memory
            for memory in memories
            if memory.lifecycle is RouteMemoryLifecycle.ACTIVE
            and memory.capability == capability
            and memory.header.compatibility.is_compatible_with(compatibility)
        )

    def plan_revalidation(
        self, compatibility: DiscoveryCompatibilityKey
    ) -> tuple[AppMapRevalidationPlan, ...]:
        """Retain scoped work for incompatible maps instead of discarding them."""
        return tuple(
            plan_app_map_revalidation(app_map, compatibility)
            for app_map in self._read_app_maps()
            if not app_map.header.compatibility.is_compatible_with(compatibility)
        )

    def _read_app_maps(self) -> tuple[AppMap, ...]:
        directory = self.root / "app_maps"
        if not directory.exists():
            return ()
        return tuple(
            parse_artifact(path.read_bytes(), AppMap) for path in sorted(directory.glob("*.json"))
        )

    @staticmethod
    def _atomic_write(destination: Path, content: bytes) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary.write(content)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_path = temporary.name
            os.replace(temporary_path, destination)
        finally:
            if temporary_path is not None and os.path.exists(temporary_path):
                os.unlink(temporary_path)
