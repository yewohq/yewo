"""A lock becomes stale when its verified generation evidence changes."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from yewo.discovery.contracts import CapabilityState
from yewo.errors import UsageError
from yewo.testing.generation_evidence import (
    GenerationCapabilityEvidence,
    GenerationRoutePrefix,
    ScenarioGenerationEvidence,
)
from yewo.testing.generator import GeneratedSuite
from yewo.testing.lock_file import read
from yewo.testing.lock_file import LockEntry
from yewo.testing.lock_resolution import suite_for
from yewo.testing.models import GeneratedTestCase, ScenarioDraft, UserJourney
from yewo.testing.steps import ActionStep
from yewo.testing.taxonomy import Category, CoverageCell, Priority, ScenarioType

CELL = CoverageCell(Category.FUNCTIONAL, ScenarioType.HAPPY_PATH, Priority.P0)


def _journey() -> UserJourney:
    return UserJourney(
        id="trainer",
        name="Trainer",
        description="A user creates a reusable schedule.",
        coverage=[CELL],
    )


def _evidence(suffix: str) -> ScenarioGenerationEvidence:
    return ScenarioGenerationEvidence(
        app_map_artifact_id=f"app-map-{suffix}",
        app_map_integrity_sha256=suffix * 64,
        capabilities=(
            GenerationCapabilityEvidence(
                capability_id="capability-form",
                intent="reach the schedule form",
                state=CapabilityState.ROUTE_VERIFIED,
                required=True,
                evidence_observation_ids=("observation-form",),
            ),
        ),
        route_prefixes=(
            GenerationRoutePrefix(
                route_memory_artifact_id="route-form",
                capability="open_schedule_form",
                semantic_instruction="Open the schedule form",
                source_node_id="list",
                destination_node_id="form",
                capability_ids=("capability-form",),
                segment_count=2,
            ),
        ),
    )


def _suite(journey, cells) -> GeneratedSuite:
    del journey
    scenario = GeneratedTestCase.of(
        cells[0],
        ScenarioDraft(
            scenario_name="Create schedule",
            app_capability_ids=["capability-form"],
            route_memory_ids=["route-form"],
            steps=[ActionStep(instruction="Launch the app.")],
            expected_outcomes=["The schedule is visible."],
        ),
    )
    return GeneratedSuite(scenarios=[scenario], locked=[scenario])


def test_lock_records_knowledge_identity_and_reuses_matching_evidence(tmp_path: Path):
    path = tmp_path / "yewo.lock"
    evidence = _evidence("a")

    first = suite_for(
        _journey(),
        _suite,
        path=path,
        generation_evidence=evidence,
    )
    unused = Mock()
    second = suite_for(
        _journey(),
        unused,
        path=path,
        generation_evidence=evidence,
    )
    entry = read(path).entries[0]

    assert first.generated_count == 1
    assert second.reused_count == 1
    unused.assert_not_called()
    assert entry.app_map_artifact_id == evidence.app_map_artifact_id
    assert entry.generation_evidence_sha256 == evidence.evidence_sha256
    assert entry.app_capability_ids == ["capability-form"]
    assert entry.route_memory_ids == ["route-form"]


def test_changed_app_evidence_regenerates_or_fails_locked_mode(tmp_path: Path):
    path = tmp_path / "yewo.lock"
    suite_for(
        _journey(),
        _suite,
        path=path,
        generation_evidence=_evidence("a"),
    )
    regenerated = Mock(side_effect=_suite)

    result = suite_for(
        _journey(),
        regenerated,
        path=path,
        generation_evidence=_evidence("b"),
    )

    assert result.generated_count == 1
    regenerated.assert_called_once()
    with pytest.raises(UsageError, match="verified app evidence changed"):
        suite_for(
            _journey(),
            Mock(),
            path=path,
            locked=True,
            generation_evidence=_evidence("c"),
        )


def test_partial_generation_evidence_cannot_be_stored_in_a_lock():
    scenario = _suite(_journey(), [CELL]).locked[0]
    evidence = _evidence("a")
    payload = {
        **LockEntry.of(
            scenario,
            _journey(),
            "",
            "digest",
            evidence,
        ).model_dump(mode="json"),
        "generation_evidence_sha256": "",
    }

    with pytest.raises(ValueError, match="knowledge-bound lock entry"):
        LockEntry.model_validate(payload)
