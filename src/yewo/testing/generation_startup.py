"""Startup order for evidence-informed generation and lock resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from yewo.llm.llm_client import LLMClient
from yewo.testing.generation_evidence import ScenarioGenerationEvidence
from yewo.testing.generation_readiness import (
    GenerationReadiness,
    GenerationReadinessState,
    assess_generation_readiness,
)
from yewo.testing.generator import ScenarioGenerator
from yewo.testing.lock_resolution import LockResolution, suite_for
from yewo.testing.models import UserJourney
from yewo.testing.warm_execution import ScenarioRoutePreparer


@dataclass(frozen=True)
class ScenarioStartupKnowledge:
    """One verified knowledge set shared by generation and execution."""

    generation_evidence: ScenarioGenerationEvidence
    route_preparer: ScenarioRoutePreparer | None = None


class GenerationKnowledgeSource(Protocol):
    """Inspect compatible knowledge and perform bounded discovery when required."""

    def load_compatible(
        self, journey: UserJourney
    ) -> ScenarioStartupKnowledge | None: ...

    def discover(self, journey: UserJourney) -> ScenarioStartupKnowledge | None: ...


@dataclass(frozen=True)
class GenerationStartupResult:
    """The startup authority decision and an optional sealed scenario suite."""

    readiness: GenerationReadiness
    suite: LockResolution | None = None
    knowledge: ScenarioStartupKnowledge | None = None

    def __post_init__(self) -> None:
        has_ready_payload = self.suite is not None and self.knowledge is not None
        if self.readiness.is_ready != has_ready_payload:
            raise ValueError("Only READY startup may carry knowledge and a scenario suite.")


@dataclass(frozen=True)
class GenerationStartupInspection:
    """The readiness decision available before live discovery starts."""

    readiness: GenerationReadiness
    knowledge: ScenarioStartupKnowledge | None = None

    def __post_init__(self) -> None:
        if self.readiness.is_ready != (self.knowledge is not None):
            raise ValueError("Only READY inspection may carry startup knowledge.")


class GenerationStartupCoordinator:
    """Enforce preflight before knowledge, generation, and lock resolution."""

    def __init__(
        self,
        journey_preflight: Callable[[UserJourney], None],
        knowledge_source: GenerationKnowledgeSource,
    ):
        self.journey_preflight = journey_preflight
        self.knowledge_source = knowledge_source

    def inspect_readiness(self, journey: UserJourney) -> GenerationStartupInspection:
        """Inspect the contract and compatible knowledge without starting discovery."""
        self.journey_preflight(journey)
        contract_readiness = assess_generation_readiness(journey, None)
        if contract_readiness.state is GenerationReadinessState.INSUFFICIENT_CONTRACT:
            return GenerationStartupInspection(contract_readiness)
        knowledge = self.knowledge_source.load_compatible(journey)
        readiness = assess_generation_readiness(
            journey,
            knowledge.generation_evidence if knowledge is not None else None,
        )
        return GenerationStartupInspection(readiness, knowledge if readiness.is_ready else None)

    def resolve_suite(
        self,
        journey: UserJourney,
        llm_client: LLMClient,
        *,
        lock_path: Path | None,
        workflow: str = "",
        locked: bool = False,
        refresh: bool = False,
    ) -> GenerationStartupResult:
        inspection = self.inspect_readiness(journey)
        readiness = inspection.readiness
        knowledge = inspection.knowledge
        if readiness.state is GenerationReadinessState.DISCOVERY_REQUIRED:
            knowledge = self.knowledge_source.discover(journey)
            readiness = assess_generation_readiness(
                journey,
                knowledge.generation_evidence if knowledge is not None else None,
                discovery_completed=True,
            )
        if not readiness.is_ready:
            return GenerationStartupResult(readiness)
        evidence = readiness.evidence
        if evidence is None:
            raise AssertionError("READY startup lost its verified app evidence.")
        generator = ScenarioGenerator(llm_client, evidence)
        return GenerationStartupResult(
            readiness,
            suite_for(
                journey,
                generator.generate_suite,
                path=lock_path,
                workflow=workflow,
                locked=locked,
                refresh=refresh,
                generation_evidence=evidence,
            ),
            knowledge,
        )
