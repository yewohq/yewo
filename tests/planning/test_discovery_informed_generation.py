"""Scenario generation uses requirements and verified app evidence together."""

from types import SimpleNamespace
from unittest.mock import Mock

from yewo.discovery.contracts import CapabilityState
from yewo.testing.generation_evidence import (
    GenerationCapabilityEvidence,
    GenerationRoutePrefix,
    ScenarioGenerationEvidence,
)
from yewo.testing.generator import ScenarioGenerator
from yewo.testing.models import ScenarioDraft, UserJourney
from yewo.testing.steps import ActionStep, IntentStep, VariationStep
from yewo.testing.taxonomy import Category, CoverageCell, Priority, ScenarioType

CAPABILITY_ID = "capability-entry-form"
ROUTE_ID = "route-entry-form"
GOAL = "A user creates and saves a reusable schedule."


def _cell(scenario_type: ScenarioType) -> CoverageCell:
    return CoverageCell(Category.FUNCTIONAL, scenario_type, Priority.P0)


def _evidence(
    state: CapabilityState = CapabilityState.ROUTE_VERIFIED,
) -> ScenarioGenerationEvidence:
    return ScenarioGenerationEvidence(
        app_map_artifact_id="app-map-1",
        app_map_integrity_sha256="a" * 64,
        capabilities=(
            GenerationCapabilityEvidence(
                capability_id=CAPABILITY_ID,
                intent="reach the schedule entry form",
                state=state,
                required=True,
                evidence_observation_ids=("observation-form",)
                if state is CapabilityState.ROUTE_VERIFIED
                else (),
            ),
        ),
        route_prefixes=(
            GenerationRoutePrefix(
                route_memory_artifact_id=ROUTE_ID,
                capability="open_schedule_entry_form",
                semantic_instruction="Open the schedule entry form",
                source_node_id="schedule-list",
                destination_node_id="schedule-form",
                capability_ids=(CAPABILITY_ID,),
                segment_count=3,
            ),
        ),
    )


def _journey(*scenario_types: ScenarioType) -> UserJourney:
    return UserJourney(
        id="trainer",
        name="Training schedules",
        description=GOAL,
        coverage=[_cell(item) for item in scenario_types],
    )


def _happy(*, capability_id: str = CAPABILITY_ID, route_id: str = ROUTE_ID):
    return ScenarioDraft(
        scenario_name="Create one schedule",
        app_capability_ids=[capability_id],
        route_memory_ids=[route_id],
        steps=[
            ActionStep(instruction="Launch the app."),
            IntentStep(instruction=GOAL, evidence=GOAL),
        ],
        expected_outcomes=[{"statement": "The saved schedule is visible.", "evidence": GOAL}],
    )


def _boundary(route_id: str = ROUTE_ID):
    return ScenarioDraft(
        scenario_name="Create a schedule with many exercises",
        app_capability_ids=[CAPABILITY_ID],
        route_memory_ids=[route_id],
        steps=[
            ActionStep(instruction="Launch the app."),
            VariationStep(
                goal=GOAL,
                objective=GOAL,
                evidence=GOAL,
                strategy=ScenarioType.BOUNDARY,
                repetitions=100,
            ),
        ],
        expected_outcomes=["The attempt has one coherent result."],
    )


def test_variants_share_verified_prefix_but_keep_distinct_user_strategy():
    client = SimpleNamespace(generate_response=Mock(side_effect=[_happy(), _boundary()]))
    generator = ScenarioGenerator(client, _evidence())

    suite = generator.generate_suite(_journey(ScenarioType.HAPPY_PATH, ScenarioType.BOUNDARY))

    assert suite.uncovered == []
    assert [item.route_memory_ids for item in suite.scenarios] == [
        [ROUTE_ID],
        [ROUTE_ID],
    ]
    assert [item.app_capability_ids for item in suite.scenarios] == [
        [CAPABILITY_ID],
        [CAPABILITY_ID],
    ]
    assert suite.scenarios[0].steps != suite.scenarios[1].steps
    assert "exactly 100 times" in suite.scenarios[1].steps_instructions[1]


def test_unknown_capability_is_rejected_instead_of_becoming_a_control_claim():
    client = SimpleNamespace(
        generate_response=Mock(return_value=_happy(capability_id="invented-control"))
    )

    suite = ScenarioGenerator(client, _evidence()).generate_suite(_journey(ScenarioType.HAPPY_PATH))

    assert suite.scenarios == []
    assert "schema never matched" in suite.uncovered[0].reason
    assert client.generate_response.call_count == 3


def test_outcome_without_requirement_provenance_is_rejected():
    draft = _happy()
    draft.expected_outcomes[0].evidence = None
    client = SimpleNamespace(generate_response=Mock(return_value=draft))

    suite = ScenarioGenerator(client, _evidence()).generate_suite(_journey(ScenarioType.HAPPY_PATH))

    assert suite.scenarios == []
    assert "schema never matched" in suite.uncovered[0].reason
    assert client.generate_response.call_count == 3


def test_variant_cannot_change_the_verified_prefix_selected_by_first_scenario():
    client = SimpleNamespace(
        generate_response=Mock(
            side_effect=[
                _happy(),
                _boundary("another-route"),
                _boundary("another-route"),
                _boundary("another-route"),
            ]
        )
    )

    suite = ScenarioGenerator(client, _evidence()).generate_suite(
        _journey(ScenarioType.HAPPY_PATH, ScenarioType.BOUNDARY)
    )

    assert [item.cell for item in suite.scenarios] == ["functional.happy_path.0"]
    assert "schema never matched" in suite.uncovered[0].reason


def test_unobserved_required_capability_is_an_explicit_gap_without_model_calls():
    client = SimpleNamespace(generate_response=Mock())

    suite = ScenarioGenerator(
        client,
        _evidence(CapabilityState.UNAVAILABLE),
    ).generate_suite(_journey(ScenarioType.HAPPY_PATH, ScenarioType.BOUNDARY))

    assert suite.scenarios == []
    assert len(suite.uncovered) == 2
    assert all(
        "reach the schedule entry form (unavailable)" in gap.reason for gap in suite.uncovered
    )
    client.generate_response.assert_not_called()


def test_prompt_exposes_ids_and_meaning_but_no_coordinates_or_scenario_outcome():
    generator = ScenarioGenerator(SimpleNamespace(), _evidence())

    prompt = generator._prompt(_journey(ScenarioType.HAPPY_PATH), _cell(ScenarioType.HAPPY_PATH))

    assert CAPABILITY_ID in prompt
    assert ROUTE_ID in prompt
    assert "Open the schedule entry form" in prompt
    assert "coordinates" not in prompt
    assert "The saved schedule is visible" not in prompt
