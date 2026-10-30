"""Load the exact sealed knowledge referenced by one schema-2 lock."""

from pathlib import Path
from typing import Protocol

from yewo.discovery.store import DiscoveryStore
from yewo.errors import UsageError
from yewo.testing.generation_evidence import build_scenario_generation_evidence
from yewo.testing.live_route_runtime import PerceptionRouteRuntimeProvider
from yewo.testing.lock_file import read
from yewo.testing.models import UserJourney
from yewo.testing.generation_startup import ScenarioStartupKnowledge
from yewo.testing.warm_execution import VerifiedScenarioRoutePreparer


def load_locked_runner_knowledge(
    journey_file: Path | None,
    workflow: str,
    journey: UserJourney,
    knowledge_root: Path | None = None,
) -> ScenarioStartupKnowledge | None:
    """Load by lock identity before device startup. Never choose a nearby artifact."""
    if journey_file is None:
        return None
    lock_path = journey_file.with_name("yewo.lock")
    lock = read(lock_path)
    if lock is None:
        return None
    entries = tuple(lock.entries_for(workflow, journey.id).values())
    app_map_ids = {entry.app_map_artifact_id for entry in entries if entry.app_map_artifact_id}
    route_ids = tuple(
        sorted({route_id for entry in entries for route_id in entry.route_memory_ids})
    )
    if not app_map_ids and not route_ids:
        return None
    if len(app_map_ids) != 1 or not route_ids:
        raise UsageError("The lock does not identify one complete app-knowledge set.")
    root = knowledge_root or journey_file.parent / ".yewo" / "knowledge"
    store = DiscoveryStore(root)
    try:
        app_map = store.load_app_map(next(iter(app_map_ids)))
        route_memories = store.load_route_memories(route_ids)
        evidence = build_scenario_generation_evidence(app_map, route_memories)
    except (FileNotFoundError, ValueError) as exc:
        raise UsageError(
            f"The lock's verified app knowledge is unavailable at {root}: {exc}"
        ) from exc
    locked_digests = {
        entry.generation_evidence_sha256 for entry in entries if entry.generation_evidence_sha256
    }
    if locked_digests != {evidence.evidence_sha256}:
        raise UsageError("The stored app knowledge differs from the lock's evidence digest.")
    return ScenarioStartupKnowledge(
        generation_evidence=evidence,
        route_preparer=VerifiedScenarioRoutePreparer(
            app_map,
            route_memories,
            app_map.header.compatibility,
            PerceptionRouteRuntimeProvider(),
        ),
    )


class RunnerDiscoveryProvider(Protocol):
    """Produce one verified knowledge set through bounded live discovery."""

    def discover(self, journey: UserJourney) -> ScenarioStartupKnowledge | None: ...


class RunnerStartupKnowledgeSource:
    """Select lock-owned knowledge or invoke the production discovery boundary."""

    def __init__(
        self,
        journey_file: Path | None,
        workflow: str,
        knowledge_root: Path | None,
        discovery_provider: RunnerDiscoveryProvider | None = None,
        explicit_knowledge: ScenarioStartupKnowledge | None = None,
    ):
        self.journey_file = journey_file
        self.workflow = workflow
        self.knowledge_root = knowledge_root
        self.discovery_provider = discovery_provider
        self.explicit_knowledge = explicit_knowledge

    def load_compatible(self, journey: UserJourney) -> ScenarioStartupKnowledge | None:
        if self.explicit_knowledge is not None:
            return self.explicit_knowledge
        return load_locked_runner_knowledge(
            self.journey_file,
            self.workflow,
            journey,
            self.knowledge_root,
        )

    def use_discovery_provider(self, provider: RunnerDiscoveryProvider) -> None:
        """Install the boundary that needs a device, once one is running.

        Discovery observes a live screen, so its provider cannot exist when the
        runner is built. This is the one way in, rather than a later step
        reaching through the source to set the attribute itself.
        """
        self.discovery_provider = provider

    def discover(self, journey: UserJourney) -> ScenarioStartupKnowledge | None:
        if self.discovery_provider is None:
            return None
        return self.discovery_provider.discover(journey)
