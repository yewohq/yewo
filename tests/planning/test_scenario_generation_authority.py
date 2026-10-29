"""Yewo generates scenarios while product meaning remains journey-owned."""

from types import SimpleNamespace
from unittest.mock import Mock
from pathlib import Path

import pytest

from yewo.testing.generator import ScenarioGenerator
from yewo.testing.models import ScenarioDraft, UserJourney
from yewo.testing.steps import ActionStep, IntentStep, VariationStep
from yewo.testing.taxonomy import Category, CoverageCell, Priority, ScenarioType
from yewo.testing.suite import load_workflows


def _cell(category: Category, scenario_type: ScenarioType) -> CoverageCell:
    return CoverageCell(category, scenario_type, Priority.P0)


def _draft() -> ScenarioDraft:
    objective = "A user creates and saves a reusable schedule."
    return ScenarioDraft(
        scenario_name="Create a schedule",
        steps=[
            ActionStep(instruction="Launch the app."),
            IntentStep(instruction=objective, evidence=objective),
        ],
        expected_outcomes=["The saved schedule is visible."],
    )


def _variation_draft(
    strategy: ScenarioType,
    *,
    goal: str = "A user creates and saves a reusable schedule.",
) -> ScenarioDraft:
    parameters: dict[str, object] = {}
    if strategy is ScenarioType.NEGATIVE:
        parameters["probe_value"] = "-100"
    elif strategy is ScenarioType.BOUNDARY:
        parameters["repetitions"] = 100
    return ScenarioDraft(
        scenario_name=f"{strategy.value} schedule user",
        steps=[
            ActionStep(instruction="Launch the app."),
            VariationStep(
                goal=goal,
                objective=goal,
                evidence=goal,
                strategy=strategy,
                **parameters,
            ),
        ],
        expected_outcomes=["The model's guessed result."],
    )


def _journey(*cells: CoverageCell, **fields) -> UserJourney:
    return UserJourney(
        id="trainer",
        name="Training schedules",
        description="A user creates and saves a reusable schedule.",
        coverage=list(cells),
        **fields,
    )


def test_happy_path_is_generated_from_the_journey() -> None:
    client = SimpleNamespace(generate_response=Mock(return_value=_draft()))
    journey = _journey(_cell(Category.FUNCTIONAL, ScenarioType.HAPPY_PATH))

    suite = ScenarioGenerator(client).generate_suite(journey)

    assert len(suite.scenarios) == 1
    assert suite.uncovered == []
    client.generate_response.assert_called_once()


@pytest.mark.parametrize(
    "category,scenario_type",
    [
        (Category.FUNCTIONAL, ScenarioType.BOUNDARY),
        (Category.INPUT, ScenarioType.EMPTY),
        (Category.FUNCTIONAL, ScenarioType.NEGATIVE),
        (Category.FUNCTIONAL, ScenarioType.EXPLORATORY),
    ],
)
def test_coverage_generates_a_user_strategy_without_declaring_product_behavior(
    category: Category, scenario_type: ScenarioType
) -> None:
    client = SimpleNamespace(
        generate_response=Mock(return_value=_variation_draft(scenario_type))
    )
    journey = _journey(_cell(category, scenario_type))

    suite = ScenarioGenerator(client).generate_suite(journey)

    assert len(suite.scenarios) == 1
    assert suite.uncovered == []
    scenario = suite.scenarios[0]
    if scenario_type is ScenarioType.BOUNDARY:
        assert "exactly 100 times" in scenario.steps_instructions[1]
    assert "completes the requested variation" in scenario.expected_outcomes[0].statement
    assert "guessed result" not in scenario.expected_outcomes[0].statement
    client.generate_response.assert_called_once()


def test_duplicate_generated_flow_is_not_a_second_scenario() -> None:
    client = SimpleNamespace(generate_response=Mock(side_effect=[_draft(), _draft()]))
    journey = _journey(
        _cell(Category.FUNCTIONAL, ScenarioType.HAPPY_PATH),
        _cell(Category.INPUT, ScenarioType.HAPPY_PATH),
    )

    suite = ScenarioGenerator(client).generate_suite(journey)

    assert len(suite.scenarios) == 1
    assert "duplicates" in suite.uncovered[0].reason


def test_trainer_journey_generates_five_distinct_user_strategies() -> None:
    workflow = next(
        item
        for item in load_workflows(Path("eval/journeys/yewo.yaml"))
        if item.name == "trainer_app"
    )
    description = workflow.journeys[0].description
    client = SimpleNamespace(
        generate_response=Mock(
            side_effect=[
                _draft(),
                _variation_draft(ScenarioType.NEGATIVE, goal=description),
                _variation_draft(ScenarioType.BOUNDARY, goal=description),
                _variation_draft(ScenarioType.EXPLORATORY, goal=description),
                _variation_draft(ScenarioType.EMPTY, goal=description),
            ]
        )
    )

    suite = ScenarioGenerator(client).generate_suite(workflow.journeys[0])

    assert [scenario.cell for scenario in suite.scenarios] == [
        "functional.happy_path.0",
        "functional.negative.0",
        "functional.boundary.0",
        "functional.exploratory.0",
        "input.empty.0",
    ]
    assert suite.uncovered == []
    assert client.generate_response.call_count == 5
