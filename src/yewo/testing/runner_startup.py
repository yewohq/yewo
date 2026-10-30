"""Fail-closed startup authority for the production journey runner."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from yewo.errors import Cause
from yewo.llm.llm_client import LLMClient
from yewo.testing.generation_evidence import ScenarioGenerationEvidence
from yewo.testing.generation_startup import (
    GenerationStartupCoordinator,
    ScenarioStartupKnowledge,
)
from yewo.testing.generator import UncoveredCell
from yewo.testing.lock_file import path_beside
from yewo.testing.lock_resolution import LockResolution
from yewo.testing.models import ReportStatus, UserJourney
from yewo.testing.runner_knowledge import (
    RunnerDiscoveryProvider,
    RunnerStartupKnowledgeSource,
)
from yewo.testing.warm_execution import ScenarioRoutePreparer


class RunnerStartupAuthority:
    """Own the runner's startup decisions and the knowledge they selected.

    Generation and execution must read one evidence set. The runner holds this
    authority rather than its own copies, so no later step can execute against
    knowledge the startup decision did not select.
    """

    def __init__(
        self,
        coordinator: GenerationStartupCoordinator,
        lock_path: Path | None,
        workflow: str,
        locked: bool,
        selected_knowledge: ScenarioStartupKnowledge | None = None,
        knowledge_source: RunnerStartupKnowledgeSource | None = None,
        discovery_factory: DiscoveryProviderFactory | None = None,
    ):
        self.coordinator = coordinator
        self.lock_path = lock_path
        self.workflow = workflow
        self.locked = locked
        self.selected_knowledge = selected_knowledge
        self.knowledge_source = knowledge_source
        self.discovery_factory = discovery_factory

    @property
    def generation_evidence(self) -> ScenarioGenerationEvidence | None:
        knowledge = self.selected_knowledge
        return knowledge.generation_evidence if knowledge is not None else None

    @property
    def route_preparer(self) -> ScenarioRoutePreparer | None:
        knowledge = self.selected_knowledge
        return knowledge.route_preparer if knowledge is not None else None

    def attach_discovery(
        self,
        device_control,
        perception,
        llm_client,
        scrubber,
        artifact_writer=None,
        diagnostic_dir=None,
    ) -> None:
        """Build the discovery boundary, now that a device is running.

        Called between opening the device and resolving the suite. Discovery
        observes a live screen, so its provider cannot exist when the runner is
        built. A run that wants no discovery was given no factory, and this
        does nothing.
        """
        if self.discovery_factory is None or self.knowledge_source is None:
            return
        self.knowledge_source.use_discovery_provider(
            self.discovery_factory(
                device_control,
                perception,
                llm_client,
                scrubber,
                artifact_writer,
                diagnostic_dir,
            )
        )

    def inspect_knowledge(self, journey: UserJourney) -> None:
        """Run journey preflight and select compatible knowledge, before any device."""
        inspection = self.coordinator.inspect_readiness(journey)
        if inspection.knowledge is not None:
            self.selected_knowledge = inspection.knowledge

    def resolve_suite(self, journey: UserJourney, llm_client: LLMClient) -> LockResolution:
        """Seal the scenarios this evidence authorizes, or cover no cell at all."""
        startup = self.coordinator.resolve_suite(
            journey,
            llm_client,
            lock_path=self.lock_path,
            workflow=self.workflow,
            locked=self.locked,
        )
        if startup.suite is None or startup.knowledge is None:
            return LockResolution(
                uncovered=[
                    UncoveredCell(
                        cell,
                        ReportStatus.UNVERIFIED,
                        Cause.SCENARIO,
                        f"{startup.readiness.state.value}: {startup.readiness.reason}",
                    )
                    for cell in journey.coverage
                ]
            )
        self.selected_knowledge = startup.knowledge
        return startup.suite


DiscoveryProviderFactory = Callable[..., RunnerDiscoveryProvider]
"""Builds the discovery boundary from a started device, perception, a client, and the run's scrubber."""


def build_runner_startup_authority(
    journey_preflight: Callable[[UserJourney], None],
    journey_file: Path | None,
    workflow: str,
    locked: bool,
    knowledge_root: Path | None,
    discovery_provider: RunnerDiscoveryProvider | None,
    generation_evidence: ScenarioGenerationEvidence | None,
    route_preparer: ScenarioRoutePreparer | None,
    discovery_factory: DiscoveryProviderFactory | None = None,
) -> RunnerStartupAuthority:
    """Wire the one startup path a runner may resolve scenarios through."""
    explicit_knowledge = (
        ScenarioStartupKnowledge(generation_evidence, route_preparer)
        if generation_evidence is not None
        else None
    )
    source = RunnerStartupKnowledgeSource(
        journey_file,
        workflow,
        knowledge_root,
        discovery_provider,
        explicit_knowledge,
    )
    return RunnerStartupAuthority(
        GenerationStartupCoordinator(journey_preflight, source),
        path_beside(journey_file) if journey_file is not None else None,
        workflow,
        locked,
        explicit_knowledge,
        source,
        discovery_factory,
    )
