"""A lock may only run against the exact knowledge it names, or nothing at all."""

import json

import pytest
from knowledge_fakes import build_verified_knowledge

from yewo.discovery.store import DiscoveryStore
from yewo.errors import UsageError
from yewo.testing.generation_evidence import build_scenario_generation_evidence
from yewo.testing.lock_file import Lock, LockEntry, contract_sha256, write
from yewo.testing.models import GeneratedTestCase, ScenarioDraft, UserJourney
from yewo.testing.runner_knowledge import load_locked_runner_knowledge
from yewo.testing.steps import ActionStep
from yewo.testing.taxonomy import Category, CoverageCell, Priority, ScenarioType


def _journey() -> UserJourney:
    return UserJourney(
        id="journey",
        name="Journey",
        description="Reach the form.",
        coverage=[CoverageCell(Category.FUNCTIONAL, ScenarioType.HAPPY_PATH, Priority.P0)],
    )


def _scenario(journey: UserJourney, route_memory_id: str) -> GeneratedTestCase:
    return GeneratedTestCase.of(
        journey.coverage[0],
        ScenarioDraft(
            scenario_name="Reach the form",
            app_capability_ids=["capability-form"],
            route_memory_ids=[route_memory_id],
            steps=[ActionStep(instruction="Open the form.")],
            expected_outcomes=["The form is visible."],
        ),
    )


def _write_lock(tmp_path, journey, scenario, evidence):
    journey_file = tmp_path / "yewo.yaml"
    journey_file.write_text("workflows: []\n")
    write(
        tmp_path / "yewo.lock",
        Lock(
            entries=[
                LockEntry.of(
                    scenario,
                    journey,
                    "workflow",
                    contract_sha256(journey, journey.coverage[0], evidence.evidence_sha256)
                    if evidence is not None
                    else contract_sha256(journey, journey.coverage[0]),
                    evidence,
                )
            ]
        ),
    )
    return journey_file


def test_locked_runner_loads_exact_knowledge_before_device_start(tmp_path, header):
    app_map, memory = build_verified_knowledge(header)
    evidence = build_scenario_generation_evidence(app_map, (memory,))
    journey = _journey()
    journey_file = _write_lock(
        tmp_path,
        journey,
        _scenario(journey, memory.header.artifact_id),
        evidence,
    )
    store = DiscoveryStore(tmp_path / ".yewo" / "knowledge")
    store.save_app_map(app_map)
    store.save_route_memory(memory)

    loaded = load_locked_runner_knowledge(journey_file, "workflow", journey)

    assert loaded.generation_evidence == evidence
    assert loaded.route_preparer is not None


def test_a_lock_naming_no_knowledge_selects_none_rather_than_a_nearby_artifact(
    tmp_path,
    header,
):
    app_map, memory = build_verified_knowledge(header)
    journey = _journey()
    journey_file = _write_lock(
        tmp_path,
        journey,
        GeneratedTestCase.of(
            journey.coverage[0],
            ScenarioDraft(
                scenario_name="Reach the form",
                steps=[ActionStep(instruction="Open the form.")],
                expected_outcomes=["The form is visible."],
            ),
        ),
        None,
    )
    store = DiscoveryStore(tmp_path / ".yewo" / "knowledge")
    store.save_app_map(app_map)
    store.save_route_memory(memory)

    assert load_locked_runner_knowledge(journey_file, "workflow", journey) is None


def test_knowledge_missing_from_the_store_is_refused(tmp_path, header):
    app_map, memory = build_verified_knowledge(header)
    evidence = build_scenario_generation_evidence(app_map, (memory,))
    journey = _journey()
    journey_file = _write_lock(
        tmp_path,
        journey,
        _scenario(journey, memory.header.artifact_id),
        evidence,
    )

    with pytest.raises(UsageError, match="unavailable"):
        load_locked_runner_knowledge(journey_file, "workflow", journey)


def test_stored_knowledge_that_differs_from_the_locked_digest_is_refused(tmp_path, header):
    app_map, memory = build_verified_knowledge(header)
    evidence = build_scenario_generation_evidence(app_map, (memory,))
    journey = _journey()
    journey_file = _write_lock(
        tmp_path,
        journey,
        _scenario(journey, memory.header.artifact_id),
        evidence,
    )
    store = DiscoveryStore(tmp_path / ".yewo" / "knowledge")
    store.save_app_map(app_map)
    store.save_route_memory(memory)
    lock_path = tmp_path / "yewo.lock"
    stored = json.loads(lock_path.read_text())
    stored["entries"][0]["generation_evidence_sha256"] = "d" * 64
    lock_path.write_text(json.dumps(stored))

    with pytest.raises(UsageError, match="differs from the lock"):
        load_locked_runner_knowledge(journey_file, "workflow", journey)


def test_a_lock_naming_two_app_maps_is_refused_rather_than_half_loaded(tmp_path, header):
    app_map, memory = build_verified_knowledge(header)
    evidence = build_scenario_generation_evidence(app_map, (memory,))
    other = evidence.model_copy(update={"app_map_artifact_id": "app-map-elsewhere"})
    journey = UserJourney(
        id="journey",
        name="Journey",
        description="Reach the form.",
        coverage=[
            CoverageCell(Category.FUNCTIONAL, ScenarioType.HAPPY_PATH, Priority.P0),
            CoverageCell(Category.FUNCTIONAL, ScenarioType.NEGATIVE, Priority.P1),
        ],
    )
    journey_file = tmp_path / "yewo.yaml"
    journey_file.write_text("workflows: []\n")
    write(
        tmp_path / "yewo.lock",
        Lock(
            entries=[
                LockEntry.of(
                    GeneratedTestCase.of(
                        cell,
                        ScenarioDraft(
                            scenario_name=f"Reach the form for {cell.key}",
                            app_capability_ids=["capability-form"],
                            route_memory_ids=[memory.header.artifact_id],
                            steps=[ActionStep(instruction="Open the form.")],
                            expected_outcomes=["The form is visible."],
                        ),
                    ),
                    journey,
                    "workflow",
                    contract_sha256(journey, cell, selected.evidence_sha256),
                    selected,
                )
                for cell, selected in zip(journey.coverage, (evidence, other), strict=True)
            ]
        ),
    )
    store = DiscoveryStore(tmp_path / ".yewo" / "knowledge")
    store.save_app_map(app_map)
    store.save_route_memory(memory)

    with pytest.raises(UsageError, match="one complete app-knowledge set"):
        load_locked_runner_knowledge(journey_file, "workflow", journey)
