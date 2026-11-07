"""Tests for deterministic scenario generation."""

import pytest

from yewo.project_config import CoverageCategory, ScenarioType
from yewo.scenarios import (
    CategoryApplicability,
    ScenarioGenerationError,
    ScenarioExecutionChange,
    ScenarioSuite,
    ScenarioSource,
    TestDesignCategoryInterpretation as CategoryInterpretation,
    TestDesignVariation as DesignVariation,
    TranslatedTestingConcern,
    ValidatedTestDesign,
    evaluate_testing_categories,
    generate_scenarios,
)

from scenarios.test_bridge import _bridge, _journey


def _design(
    suite: ScenarioSuite,
    check: str,
    *,
    category: CoverageCategory,
    scenario_type: ScenarioType = ScenarioType.HAPPY_PATH,
    execution_change: ScenarioExecutionChange = ScenarioExecutionChange.ACCEPTED_ROUTE,
) -> ValidatedTestDesign:
    concern = TranslatedTestingConcern(
        concern_id="concern_1",
        source_check_index=0,
        source_check=check,
        category=category,
        scenario_type=scenario_type,
        execution_change=execution_change,
        expected_outcome=check,
    )
    return ValidatedTestDesign(
        design_id="design_1",
        journey_id=suite.journey_id,
        readiness_id=suite.readiness.readiness_id,
        model_request_id="request_1",
        concerns=(concern,),
        category_interpretations=(),
    )


def test_generator_groups_happy_path_lenses_into_one_execution() -> None:
    check = "The saved item must remain visible after the operation finishes."
    journey = _journey().model_copy(update={"checks": (check,)})
    suite = _bridge()
    design = _design(suite, check, category=CoverageCategory.STATE)
    registry = evaluate_testing_categories(journey, suite, concerns=design.concerns)

    result = generate_scenarios(journey, suite, design, registry)

    baseline = next(
        item for item in result.suite.scenarios if item.scenario_type is ScenarioType.HAPPY_PATH
    )
    obligations_by_category = {item.category: item for item in baseline.obligations}
    assert baseline.scenario_type is ScenarioType.HAPPY_PATH
    assert baseline.execution_change is ScenarioExecutionChange.ACCEPTED_ROUTE
    assert set(obligations_by_category) >= {
        CoverageCategory.FUNCTIONAL,
        CoverageCategory.STATE,
        CoverageCategory.INPUT,
        CoverageCategory.UI_VISUAL,
        CoverageCategory.ACCESSIBILITY,
    }
    functional = obligations_by_category[CoverageCategory.FUNCTIONAL]
    state = obligations_by_category[CoverageCategory.STATE]
    assert {item.statement for item in functional.expected_outcomes} == {
        requirement.statement for requirement in suite.readiness.requirements
    }
    assert state.source is ScenarioSource.USER_REQUIREMENT
    assert state.strategy is ScenarioType.HAPPY_PATH
    assert state.expected_outcomes[0].statement == check
    assert state.source_check_indexes == (0,)
    assert baseline.objective == journey.objective
    assert baseline.route_procedure_id == suite.readiness.procedure_id
    lifecycle = next(
        item
        for item in result.suite.scenarios
        if item.scenario_type is ScenarioType.INTERRUPTION
        and item.execution_change is ScenarioExecutionChange.LIFECYCLE_INTERRUPTION
    )
    assert any(
        obligation.category is CoverageCategory.LIFECYCLE for obligation in lifecycle.obligations
    )
    assert len(lifecycle.execution_capabilities) == 1
    lifecycle_capability = lifecycle.execution_capabilities[0]
    assert lifecycle_capability.kind.value == "app_state"
    assert lifecycle_capability.model_dump()["action"] == "background_resume"


def test_generator_adds_general_data_consistency_to_the_accepted_route() -> None:
    check = "The saved item must remain visible after the operation finishes."
    journey = _journey().model_copy(update={"checks": (check,)})
    suite = _bridge()
    related_surface_ids = tuple(item.surface_id for item in suite.surfaces[:2])
    interpretation = CategoryInterpretation(
        interpretation_id="derived_value_relationship",
        category=CoverageCategory.DATA_CONSISTENCY,
        surface_ids=related_surface_ids,
        variations=(
            DesignVariation(
                scenario_type=ScenarioType.HAPPY_PATH,
                execution_change=ScenarioExecutionChange.ACCEPTED_ROUTE,
            ),
        ),
        reason="A later observed value summarizes an earlier observed value.",
    )
    design = _design(suite, check, category=CoverageCategory.STATE).model_copy(
        update={"category_interpretations": (interpretation,)}
    )
    registry = evaluate_testing_categories(
        journey,
        suite,
        concerns=design.concerns,
        interpretations=design.category_interpretations,
    )

    result = generate_scenarios(journey, suite, design, registry)

    baseline = next(
        item
        for item in result.suite.scenarios
        if item.execution_change is ScenarioExecutionChange.ACCEPTED_ROUTE
    )
    consistency = next(
        item for item in baseline.obligations if item.category is CoverageCategory.DATA_CONSISTENCY
    )
    assert consistency.required_surface_ids == related_surface_ids
    assert consistency.expected_outcomes[0].statement == (
        "Derived values must match their observed source values."
    )
    assert tuple(
        item.owner.value for item in consistency.expected_outcomes[0].evidence_requirements
    ) == ("application_observation",)


def test_negative_qa_intent_can_reuse_the_accepted_execution() -> None:
    check = "The list stays alphabetical after the accepted denial action."
    journey = _journey().model_copy(update={"checks": (check,)})
    suite = _bridge()
    design = _design(
        suite,
        check,
        category=CoverageCategory.FUNCTIONAL,
        scenario_type=ScenarioType.NEGATIVE,
        execution_change=ScenarioExecutionChange.ACCEPTED_ROUTE,
    )
    registry = evaluate_testing_categories(journey, suite, concerns=design.concerns)

    result = generate_scenarios(journey, suite, design, registry)

    user_obligation = next(
        obligation
        for scenario in result.suite.scenarios
        for obligation in scenario.obligations
        if obligation.source_check_indexes == (0,)
    )
    owning_scenario = next(
        scenario for scenario in result.suite.scenarios if user_obligation in scenario.obligations
    )
    assert user_obligation.strategy is ScenarioType.NEGATIVE
    assert owning_scenario.execution_change is ScenarioExecutionChange.ACCEPTED_ROUTE
    assert owning_scenario.scenario_type is ScenarioType.HAPPY_PATH


def test_baseline_outcomes_keep_their_exact_discovery_surfaces() -> None:
    check = "The saved item must remain visible after the operation finishes."
    journey = _journey().model_copy(update={"checks": (check,)})
    suite = _bridge()
    design = _design(suite, check, category=CoverageCategory.STATE)
    registry = evaluate_testing_categories(journey, suite, concerns=design.concerns)

    result = generate_scenarios(journey, suite, design, registry)

    baseline = next(
        scenario
        for scenario in result.suite.scenarios
        if scenario.execution_change is ScenarioExecutionChange.ACCEPTED_ROUTE
    )
    functional = next(
        obligation
        for obligation in baseline.obligations
        if obligation.category is CoverageCategory.FUNCTIONAL
    )
    functional_surface_ids = set(functional.required_surface_ids)
    for requirement, outcome in zip(
        suite.readiness.requirements,
        functional.expected_outcomes,
        strict=True,
    ):
        expected = tuple(
            surface.surface_id
            for surface in suite.surfaces
            if surface.surface_id in functional_surface_ids
            and set(requirement.procedure_step_indexes).intersection(
                surface.provenance.procedure_step_indexes
            )
        )
        assert outcome.required_surface_ids == expected


def test_categories_with_the_same_execution_change_share_one_scenario() -> None:
    check = "The saved item must remain visible after the operation finishes."
    journey = _journey().model_copy(update={"checks": (check,)})
    suite = _bridge()
    input_surface_id = next(
        item.surface_id for item in suite.surfaces if item.kind.value == "input"
    )
    variation = DesignVariation(
        scenario_type=ScenarioType.NEGATIVE,
        execution_change=ScenarioExecutionChange.INPUT_VARIATION,
    )
    interpretations = tuple(
        CategoryInterpretation(
            interpretation_id=f"{category.value}_input",
            category=category,
            surface_ids=(input_surface_id,),
            variations=(variation,),
            reason="The same changed input can settle this lens.",
        )
        for category in (CoverageCategory.ERROR_HANDLING, CoverageCategory.SECURITY)
    )
    design = _design(suite, check, category=CoverageCategory.STATE).model_copy(
        update={"category_interpretations": interpretations}
    )
    registry = evaluate_testing_categories(
        journey,
        suite,
        concerns=design.concerns,
        interpretations=design.category_interpretations,
    )

    result = generate_scenarios(journey, suite, design, registry)

    input_scenario = next(
        item
        for item in result.suite.scenarios
        if item.execution_change is ScenarioExecutionChange.INPUT_VARIATION
        and item.scenario_type is ScenarioType.NEGATIVE
    )
    assert {item.category for item in input_scenario.obligations} == {
        CoverageCategory.ERROR_HANDLING,
        CoverageCategory.SECURITY,
    }
    assert all(
        set(obligation.required_surface_ids).issubset(
            {surface.surface_id for surface in suite.surfaces}
        )
        for scenario in result.suite.scenarios
        for obligation in scenario.obligations
    )


def test_user_scenario_uses_the_model_interpreted_surface_subset() -> None:
    check = "The saved item must remain visible after the operation finishes."
    journey = _journey().model_copy(update={"checks": (check,)})
    suite = _bridge()
    surface_id = suite.surfaces[0].surface_id
    design = _design(suite, check, category=CoverageCategory.STATE).model_copy(
        update={
            "category_interpretations": (
                CategoryInterpretation(
                    interpretation_id="state_surface",
                    category=CoverageCategory.STATE,
                    surface_ids=(surface_id,),
                    variations=(
                        DesignVariation(
                            scenario_type=ScenarioType.RECOVERY,
                            execution_change=ScenarioExecutionChange.RECOVERY_ATTEMPT,
                        ),
                    ),
                    reason="This surface produces the state under test.",
                ),
            )
        }
    )
    registry = evaluate_testing_categories(
        journey,
        suite,
        concerns=design.concerns,
        interpretations=design.category_interpretations,
    )

    result = generate_scenarios(journey, suite, design, registry)

    user_obligation = next(
        obligation
        for scenario in result.suite.scenarios
        for obligation in scenario.obligations
        if obligation.source is ScenarioSource.USER_REQUIREMENT
        and obligation.category is CoverageCategory.STATE
    )
    assert user_obligation.required_surface_ids == (surface_id,)


def test_presentation_surface_can_support_multiple_lenses_in_one_execution() -> None:
    check = "The saved item must remain visible after the operation finishes."
    journey = _journey().model_copy(update={"checks": (check,)})
    suite = _bridge()
    design = _design(suite, check, category=CoverageCategory.STATE)
    registry = evaluate_testing_categories(journey, suite, concerns=design.concerns)

    result = generate_scenarios(journey, suite, design, registry)

    baseline = next(
        item for item in result.suite.scenarios if item.scenario_type is ScenarioType.HAPPY_PATH
    )
    ui = next(item for item in baseline.obligations if item.category is CoverageCategory.UI_VISUAL)
    accessibility = next(
        item for item in baseline.obligations if item.category is CoverageCategory.ACCESSIBILITY
    )
    presentation_ids = {
        item.surface_id for item in suite.surfaces if item.kind.value == "presentation"
    }
    assert presentation_ids.intersection(ui.required_surface_ids)
    assert presentation_ids.intersection(accessibility.required_surface_ids)


def test_generator_records_unavailable_general_categories_as_gaps() -> None:
    check = "The saved item must remain visible after the operation finishes."
    journey = _journey().model_copy(update={"checks": (check,)})
    suite = _bridge()
    design = _design(suite, check, category=CoverageCategory.STATE)
    registry = evaluate_testing_categories(journey, suite, concerns=design.concerns)

    result = generate_scenarios(journey, suite, design, registry)

    permission = next(item for item in result.gaps if item.category is CoverageCategory.PERMISSION)
    assessment = next(
        item for item in registry.assessments if item.category is CoverageCategory.PERMISSION
    )
    assert assessment.applicability is CategoryApplicability.NOT_APPLICABLE
    assert permission.reason == assessment.reason


def test_generator_keeps_an_unsupported_user_check_as_an_explicit_gap() -> None:
    check = "The app must continue after native location permission is denied."
    journey = _journey().model_copy(update={"checks": (check,)})
    suite = _bridge()
    design = _design(
        suite,
        check,
        category=CoverageCategory.PERMISSION,
        scenario_type=ScenarioType.NEGATIVE,
        execution_change=ScenarioExecutionChange.PERMISSION_DECISION,
    )
    registry = evaluate_testing_categories(journey, suite, concerns=design.concerns)

    result = generate_scenarios(journey, suite, design, registry)

    gap = next(item for item in result.gaps if item.source_check_index == 0)
    assert gap.source is ScenarioSource.USER_REQUIREMENT
    assert gap.statement == check
    assert all(
        obligation.category is not CoverageCategory.PERMISSION
        for scenario in result.suite.scenarios
        for obligation in scenario.obligations
    )


def test_scenario_generation_identity_is_stable() -> None:
    check = "The saved item must remain visible after the operation finishes."
    journey = _journey().model_copy(update={"checks": (check,)})
    suite = _bridge()
    design = _design(suite, check, category=CoverageCategory.STATE)
    registry = evaluate_testing_categories(journey, suite, concerns=design.concerns)

    first = generate_scenarios(journey, suite, design, registry)
    second = generate_scenarios(journey, suite, design, registry)

    assert first == second


def test_generator_rejects_a_design_from_another_readiness_decision() -> None:
    check = "The saved item must remain visible after the operation finishes."
    journey = _journey().model_copy(update={"checks": (check,)})
    suite = _bridge()
    design = _design(suite, check, category=CoverageCategory.STATE).model_copy(
        update={"readiness_id": "other_readiness"}
    )
    registry = evaluate_testing_categories(journey, suite, concerns=design.concerns)

    with pytest.raises(ScenarioGenerationError, match="readiness decisions"):
        generate_scenarios(journey, suite, design, registry)
