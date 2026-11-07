"""Tests for automatic QA-lens selection."""

import pytest

from yewo.project_config import CapabilityKind, CoverageCategory, ScenarioType
from yewo.scenarios import (
    CategoryApplicability,
    ExpectedOutcomeAuthority,
    ScenarioExecutionChange,
    TestDesignCategoryInterpretation as CategoryInterpretation,
    TestDesignVariation as DesignVariation,
    TestingCategoryRegistryError as RegistryError,
    TestingRegistryContext as RegistryContext,
    TranslatedTestingConcern,
    evaluate_testing_categories,
)

from scenarios.test_bridge import _bridge, _journey


def _assessments(result):
    return {item.category: item for item in result.assessments}


def test_registry_applies_general_qa_knowledge_to_discovered_surfaces() -> None:
    result = evaluate_testing_categories(_journey(), _bridge())
    assessments = _assessments(result)

    assert len(assessments) == len(CoverageCategory)
    assert assessments[CoverageCategory.FUNCTIONAL].applicability is (
        CategoryApplicability.APPLICABLE
    )
    assert assessments[CoverageCategory.INPUT].applicability is CategoryApplicability.APPLICABLE
    assert assessments[CoverageCategory.STATE].applicability is CategoryApplicability.APPLICABLE
    assert assessments[CoverageCategory.UI_VISUAL].applicability is (
        CategoryApplicability.APPLICABLE
    )
    assert assessments[CoverageCategory.ACCESSIBILITY].applicability is (
        CategoryApplicability.APPLICABLE
    )
    assert assessments[CoverageCategory.LIFECYCLE].required_capabilities == (
        CapabilityKind.APP_STATE,
    )
    assert assessments[CoverageCategory.NAVIGATION].applicability is (
        CategoryApplicability.NOT_APPLICABLE
    )
    assert assessments[CoverageCategory.PERFORMANCE].applicability is (
        CategoryApplicability.INSUFFICIENT_EVIDENCE
    )
    assert assessments[CoverageCategory.DATA_CONSISTENCY].applicability is (
        CategoryApplicability.NOT_APPLICABLE
    )


def test_data_consistency_requires_an_observed_value_relationship() -> None:
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

    result = evaluate_testing_categories(
        _journey(),
        suite,
        interpretations=(interpretation,),
    )

    consistency = _assessments(result)[CoverageCategory.DATA_CONSISTENCY]
    assert consistency.applicability is CategoryApplicability.APPLICABLE
    assert consistency.surface_ids == related_surface_ids
    assert consistency.expected_outcome_authorities == (
        ExpectedOutcomeAuthority.DETERMINISTIC_INVARIANT,
    )
    assert tuple(item.value for item in consistency.evidence_owners) == ("application_observation",)


def test_user_check_remains_plain_text_until_translation() -> None:
    check = "The user's work must remain after the app restarts."
    journey = _journey().model_copy(update={"checks": (check,)})

    result = evaluate_testing_categories(journey, _bridge())

    assert result.untranslated_check_indexes == (0,)


def test_translated_concern_guides_internal_category_selection() -> None:
    check = "The user's work must remain after the app restarts."
    journey = _journey().model_copy(update={"checks": (check,)})
    concern = TranslatedTestingConcern(
        concern_id="restart_persistence",
        source_check_index=0,
        source_check=check,
        category=CoverageCategory.LIFECYCLE,
        scenario_type=ScenarioType.INTERRUPTION,
        execution_change=ScenarioExecutionChange.LIFECYCLE_INTERRUPTION,
        expected_outcome=check,
    )

    result = evaluate_testing_categories(journey, _bridge(), concerns=(concern,))
    lifecycle = _assessments(result)[CoverageCategory.LIFECYCLE]

    assert result.untranslated_check_indexes == ()
    assert lifecycle.applicability is CategoryApplicability.APPLICABLE
    assert lifecycle.concern_ids == ("restart_persistence",)
    assert ExpectedOutcomeAuthority.USER_REQUIREMENT in (lifecycle.expected_outcome_authorities)


def test_translator_cannot_change_the_user_check() -> None:
    journey = _journey().model_copy(update={"checks": ("Work must remain.",)})
    concern = TranslatedTestingConcern(
        concern_id="changed",
        source_check_index=0,
        source_check="Different requirement.",
        category=CoverageCategory.STATE,
        scenario_type=ScenarioType.HAPPY_PATH,
        execution_change=ScenarioExecutionChange.ACCEPTED_ROUTE,
        expected_outcome="Different requirement.",
    )

    with pytest.raises(RegistryError, match="changed the user's check"):
        evaluate_testing_categories(journey, _bridge(), concerns=(concern,))


def test_missing_device_capability_is_reported_as_unsupported() -> None:
    result = evaluate_testing_categories(
        _journey(),
        _bridge(),
        context=RegistryContext(available_capabilities=frozenset()),
    )

    lifecycle = _assessments(result)[CoverageCategory.LIFECYCLE]
    assert lifecycle.applicability is CategoryApplicability.UNSUPPORTED
    assert "app_state" in lifecycle.reason


def test_permission_concern_without_a_native_surface_is_insufficient() -> None:
    check = "The app must continue after location access is denied."
    journey = _journey().model_copy(update={"checks": (check,)})
    concern = TranslatedTestingConcern(
        concern_id="permission_denial",
        source_check_index=0,
        source_check=check,
        category=CoverageCategory.PERMISSION,
        scenario_type=ScenarioType.NEGATIVE,
        execution_change=ScenarioExecutionChange.PERMISSION_DECISION,
        expected_outcome=check,
    )

    result = evaluate_testing_categories(journey, _bridge(), concerns=(concern,))

    permission = _assessments(result)[CoverageCategory.PERMISSION]
    assert permission.applicability is CategoryApplicability.INSUFFICIENT_EVIDENCE


def test_permission_interpretation_requires_a_native_system_surface() -> None:
    suite = _bridge()
    interpretation = CategoryInterpretation(
        interpretation_id="app_location_choice",
        category=CoverageCategory.PERMISSION,
        surface_ids=(suite.surfaces[0].surface_id,),
        variations=(
            DesignVariation(
                scenario_type=ScenarioType.NEGATIVE,
                execution_change=ScenarioExecutionChange.PERMISSION_DECISION,
            ),
        ),
        reason="The app offers an access choice.",
    )

    with pytest.raises(RegistryError, match="native system interaction"):
        evaluate_testing_categories(
            _journey(),
            suite,
            interpretations=(interpretation,),
        )


def test_performance_needs_a_user_threshold_or_requirement() -> None:
    check = "Saving must finish within two seconds."
    journey = _journey().model_copy(update={"checks": (check,)})
    concern = TranslatedTestingConcern(
        concern_id="save_latency",
        source_check_index=0,
        source_check=check,
        category=CoverageCategory.PERFORMANCE,
        scenario_type=ScenarioType.HAPPY_PATH,
        execution_change=ScenarioExecutionChange.ACCEPTED_ROUTE,
        expected_outcome=check,
    )

    result = evaluate_testing_categories(journey, _bridge(), concerns=(concern,))

    performance = _assessments(result)[CoverageCategory.PERFORMANCE]
    assert performance.applicability is CategoryApplicability.APPLICABLE
    assert performance.expected_outcome_authorities == (ExpectedOutcomeAuthority.USER_REQUIREMENT,)


def test_compatibility_uses_the_target_matrix_instead_of_one_control() -> None:
    result = evaluate_testing_categories(
        _journey(),
        _bridge(),
        context=RegistryContext(target_count=2),
    )

    compatibility = _assessments(result)[CoverageCategory.COMPATIBILITY]
    assert compatibility.applicability is CategoryApplicability.APPLICABLE
    assert compatibility.scope.value == "target_matrix"


def test_regression_requires_an_accepted_baseline() -> None:
    without_baseline = evaluate_testing_categories(_journey(), _bridge())
    with_baseline = evaluate_testing_categories(
        _journey(),
        _bridge(),
        context=RegistryContext(has_regression_baseline=True),
    )

    assert _assessments(without_baseline)[CoverageCategory.REGRESSION].applicability is (
        CategoryApplicability.INSUFFICIENT_EVIDENCE
    )
    assert _assessments(with_baseline)[CoverageCategory.REGRESSION].applicability is (
        CategoryApplicability.APPLICABLE
    )


def test_registry_refuses_to_bypass_discovery_readiness() -> None:
    suite = _bridge()
    unready = suite.model_copy(
        update={
            "readiness": suite.readiness.model_copy(
                update={"executable": False, "independently_auditable": False}
            )
        }
    )

    with pytest.raises(RegistryError, match="ready discovery"):
        evaluate_testing_categories(_journey(), unready)
