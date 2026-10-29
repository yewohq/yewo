"""Generation startup orders preflight, app knowledge, generation, and lock."""

from pathlib import Path

import pytest

from yewo.discovery.contracts import CapabilityState
from yewo.testing.generation_evidence import (
    GenerationCapabilityEvidence,
    GenerationRoutePrefix,
    ScenarioGenerationEvidence,
)
from yewo.testing.generation_startup import (
    GenerationStartupCoordinator,
    GenerationStartupInspection,
    GenerationStartupResult,
    ScenarioStartupKnowledge,
)
from yewo.testing.generation_readiness import GenerationReadiness, GenerationReadinessState
from yewo.testing.lock_resolution import LockResolution
from yewo.testing.models import ScenarioDraft, UserJourney
from yewo.testing.steps import ActionStep, IntentStep
from yewo.testing.taxonomy import Category, CoverageCell, Priority, ScenarioType

GOAL = "A user creates a reusable schedule."


def _journey() -> UserJourney:
    return UserJourney(
        id="trainer",
        name="Trainer",
        description=GOAL,
        coverage=[
            CoverageCell(
                Category.FUNCTIONAL,
                ScenarioType.HAPPY_PATH,
                Priority.P0,
            )
        ],
    )


def _evidence() -> ScenarioGenerationEvidence:
    return ScenarioGenerationEvidence(
        app_map_artifact_id="app-map",
        app_map_integrity_sha256="a" * 64,
        capabilities=(
            GenerationCapabilityEvidence(
                capability_id="capability-form",
                intent="reach form",
                state=CapabilityState.ROUTE_VERIFIED,
                required=True,
                evidence_observation_ids=("observation-form",),
            ),
        ),
        route_prefixes=(
            GenerationRoutePrefix(
                route_memory_artifact_id="route-form",
                capability="open_form",
                semantic_instruction="Open the form",
                source_node_id="list",
                destination_node_id="form",
                capability_ids=("capability-form",),
                segment_count=2,
            ),
        ),
    )


def _knowledge() -> ScenarioStartupKnowledge:
    return ScenarioStartupKnowledge(_evidence())


class _KnowledgeSource:
    def __init__(self, events, compatible=None, discovered=None):
        self.events = events
        self.compatible = compatible
        self.discovered = discovered

    def load_compatible(self, journey):
        assert journey.id == "trainer"
        self.events.append("knowledge:inspect")
        return self.compatible

    def discover(self, journey):
        assert journey.id == "trainer"
        self.events.append("knowledge:discover")
        return self.discovered


class _Client:
    def __init__(self, events):
        self.events = events

    def generate_response(self, **request):
        del request
        self.events.append("generation")
        return ScenarioDraft(
            scenario_name="Create schedule",
            app_capability_ids=["capability-form"],
            route_memory_ids=["route-form"],
            steps=[
                ActionStep(instruction="Launch the app."),
                IntentStep(instruction=GOAL, evidence=GOAL),
            ],
            expected_outcomes=[{"statement": "The schedule is visible.", "evidence": GOAL}],
        )


def test_startup_orders_preflight_knowledge_generation_then_lock(tmp_path: Path):
    events = []
    lock_path = tmp_path / "yewo.lock"
    coordinator = GenerationStartupCoordinator(
        lambda journey: events.append(f"preflight:{journey.id}"),
        _KnowledgeSource(events, discovered=_knowledge()),
    )

    result = coordinator.resolve_suite(
        _journey(),
        _Client(events),
        lock_path=lock_path,
    )

    assert events == [
        "preflight:trainer",
        "knowledge:inspect",
        "knowledge:discover",
        "generation",
    ]
    assert result.readiness.state is GenerationReadinessState.READY
    assert result.suite is not None
    assert result.suite.generated_count == 1
    assert result.suite.lock_written
    assert lock_path.is_file()


def test_compatible_knowledge_skips_discovery(tmp_path: Path):
    events = []
    result = GenerationStartupCoordinator(
        lambda journey: events.append(f"preflight:{journey.id}"),
        _KnowledgeSource(events, compatible=_knowledge()),
    ).resolve_suite(
        _journey(),
        _Client(events),
        lock_path=tmp_path / "yewo.lock",
    )

    assert result.readiness.state is GenerationReadinessState.READY
    assert events == ["preflight:trainer", "knowledge:inspect", "generation"]


def test_missing_evidence_fails_closed_without_generation_or_lock(tmp_path: Path):
    events = []
    lock_path = tmp_path / "yewo.lock"
    result = GenerationStartupCoordinator(
        lambda journey: events.append(f"preflight:{journey.id}"),
        _KnowledgeSource(events),
    ).resolve_suite(
        _journey(),
        _Client(events),
        lock_path=lock_path,
    )

    assert result.readiness.state is GenerationReadinessState.CAPABILITY_UNOBSERVED
    assert result.suite is None
    assert events == [
        "preflight:trainer",
        "knowledge:inspect",
        "knowledge:discover",
    ]
    assert not lock_path.exists()


def test_insufficient_contract_never_inspects_knowledge(tmp_path: Path):
    events = []
    journey = _journey().model_copy(update={"description": ""})
    result = GenerationStartupCoordinator(
        lambda journey: events.append(f"preflight:{journey.id}"),
        _KnowledgeSource(events, compatible=_knowledge()),
    ).resolve_suite(
        journey,
        _Client(events),
        lock_path=tmp_path / "yewo.lock",
    )

    assert result.readiness.state is GenerationReadinessState.INSUFFICIENT_CONTRACT
    assert result.suite is None
    assert events == ["preflight:trainer"]


def test_a_startup_result_cannot_report_a_suite_it_was_not_ready_for():
    with pytest.raises(ValueError, match="Only READY startup"):
        GenerationStartupResult(
            GenerationReadiness(
                GenerationReadinessState.DISCOVERY_REQUIRED,
                "No knowledge was selected.",
            ),
            LockResolution(),
            _knowledge(),
        )


def test_a_ready_startup_result_must_carry_its_knowledge_and_suite():
    with pytest.raises(ValueError, match="Only READY startup"):
        GenerationStartupResult(
            GenerationReadiness(
                GenerationReadinessState.READY,
                "Verified evidence was selected.",
                _evidence(),
            )
        )


def test_an_inspection_cannot_carry_knowledge_it_did_not_approve():
    with pytest.raises(ValueError, match="Only READY inspection"):
        GenerationStartupInspection(
            GenerationReadiness(
                GenerationReadinessState.CAPABILITY_UNOBSERVED,
                "Discovery observed nothing the journey needs.",
            ),
            _knowledge(),
        )
