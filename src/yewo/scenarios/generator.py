"""Compile accepted test design into evidence-grounded scenarios."""

from __future__ import annotations

from hashlib import sha256

from yewo.artifacts import RecordIdentifier
from yewo.project_config import CoverageCategory, ResolvedJourney, ScenarioType
from yewo.project_config.capabilities import AppStateCapabilitySpec, DeviceCapabilitySpec
from yewo.scenarios.designer import ValidatedTestDesign
from yewo.scenarios.execution import resolve_automatic_execution_change
from yewo.scenarios.records import (
    ScenarioCoverageObligation,
    ScenarioDefinition,
    ScenarioEvidenceRequirement,
    ScenarioExpectedOutcome,
    ScenarioGenerationGap,
    ScenarioGenerationResult,
    ScenarioSuite,
    TranslatedTestingConcern,
)
from yewo.scenarios.generator_evidence import (
    interpreted_surface_ids,
    requirement_surface_ids,
)
from yewo.scenarios.registry import (
    CategoryApplicability,
    CategoryAssessment,
    TestingCategoryRegistryResult,
)
from yewo.scenarios.taxonomy import (
    ExpectedOutcomeAuthority,
    RequiredEvidenceOwner,
    ScenarioExecutionChange,
    ScenarioSource,
)


class ScenarioGenerationError(ValueError):
    """Accepted design inputs cannot produce a truthful scenario result."""


_AUTOMATIC_STRATEGY = {
    CoverageCategory.DATA_CONSISTENCY: ScenarioType.HAPPY_PATH,
    CoverageCategory.STATE: ScenarioType.HAPPY_PATH,
    CoverageCategory.INPUT: ScenarioType.EMPTY,
    CoverageCategory.NAVIGATION: ScenarioType.HAPPY_PATH,
    CoverageCategory.ERROR_HANDLING: ScenarioType.NEGATIVE,
    CoverageCategory.UI_VISUAL: ScenarioType.EXPLORATORY,
    CoverageCategory.SECURITY: ScenarioType.NEGATIVE,
    CoverageCategory.ACCESSIBILITY: ScenarioType.HAPPY_PATH,
    CoverageCategory.PERFORMANCE: ScenarioType.HAPPY_PATH,
    CoverageCategory.NETWORK: ScenarioType.INTERRUPTION,
    CoverageCategory.LIFECYCLE: ScenarioType.INTERRUPTION,
    CoverageCategory.PERMISSION: ScenarioType.NEGATIVE,
    CoverageCategory.COMPATIBILITY: ScenarioType.HAPPY_PATH,
    CoverageCategory.INSTALLATION: ScenarioType.RECOVERY,
    CoverageCategory.LOCALIZATION: ScenarioType.BOUNDARY,
    CoverageCategory.REGRESSION: ScenarioType.HAPPY_PATH,
}
_VARIATION_BY_EXECUTION_CHANGE = {
    ScenarioExecutionChange.ACCEPTED_ROUTE: "Replay the accepted discovery route.",
    ScenarioExecutionChange.INPUT_VARIATION: "Change the selected input action or value.",
    ScenarioExecutionChange.ENVIRONMENT_VARIATION: "Change the selected device environment.",
    ScenarioExecutionChange.LIFECYCLE_INTERRUPTION: (
        "Interrupt the application lifecycle and then continue."
    ),
    ScenarioExecutionChange.PERMISSION_DECISION: "Apply the selected native permission decision.",
    ScenarioExecutionChange.RECOVERY_ATTEMPT: "Apply the selected recovery action.",
    ScenarioExecutionChange.ACTOR_VARIATION: "Use the selected alternate actor.",
    ScenarioExecutionChange.CONCURRENT_OPERATION: "Apply the selected concurrent operation.",
    ScenarioExecutionChange.INSTALLATION_CHANGE: "Apply the selected installation change.",
}
_INVARIANT_BY_CATEGORY = {
    CoverageCategory.DATA_CONSISTENCY: "Derived values must match their observed source values.",
    CoverageCategory.STATE: "The tested action must produce an observable application state.",
    CoverageCategory.NAVIGATION: "The tested navigation must produce an observable destination.",
    CoverageCategory.ERROR_HANDLING: "A rejected operation must not be reported as successful.",
    CoverageCategory.UI_VISUAL: "Required content must remain observable in the tested state.",
    CoverageCategory.SECURITY: "Sensitive content must not become newly exposed.",
    CoverageCategory.ACCESSIBILITY: (
        "Required interactive content must be available to accessibility inspection."
    ),
    CoverageCategory.NETWORK: "A controlled network change must produce an observable outcome.",
    CoverageCategory.LIFECYCLE: (
        "A controlled lifecycle change must produce an observable outcome."
    ),
    CoverageCategory.PERMISSION: (
        "A controlled permission decision must produce an observable outcome."
    ),
}
_EVIDENCE_DESCRIPTIONS = {
    RequiredEvidenceOwner.APPLICATION_OBSERVATION: (
        "A settled application observation must show the expected outcome."
    ),
    RequiredEvidenceOwner.ACTION_OUTCOME: (
        "The action outcome must show whether the intended effect occurred."
    ),
    RequiredEvidenceOwner.DEVICE_RECEIPT: (
        "A device receipt must record the controlled device-state change."
    ),
    RequiredEvidenceOwner.APPLICATION_GROUND_TRUTH: (
        "Application ground truth must show the expected value."
    ),
    RequiredEvidenceOwner.MEASUREMENT: "A recorded measurement must support the result.",
    RequiredEvidenceOwner.ACCESSIBILITY_INSPECTION: (
        "Accessibility inspection must expose the required interactive content."
    ),
}


def generate_scenarios(
    journey: ResolvedJourney,
    suite: ScenarioSuite,
    design: ValidatedTestDesign,
    registry: TestingCategoryRegistryResult,
) -> ScenarioGenerationResult:
    """Group QA obligations into the smallest truthful set of test executions."""
    _validate_generation_inputs(journey, suite, design, registry)
    assessments = {item.category: item for item in registry.assessments}
    functional = assessments[CoverageCategory.FUNCTIONAL]
    obligations_by_execution: dict[
        tuple[ScenarioExecutionChange, ScenarioType], list[ScenarioCoverageObligation]
    ] = {
        (ScenarioExecutionChange.ACCEPTED_ROUTE, ScenarioType.HAPPY_PATH): [
            _baseline_obligation(journey, suite, functional)
        ]
    }
    gaps: list[ScenarioGenerationGap] = []
    represented_categories: set[CoverageCategory] = {CoverageCategory.FUNCTIONAL}
    concerns_by_check = _concerns_by_check(design)
    for check_index, check in enumerate(journey.checks):
        for concern in concerns_by_check[check_index]:
            assessment = assessments[concern.category]
            represented_categories.add(concern.category)
            if assessment.applicability is CategoryApplicability.APPLICABLE:
                obligation = _user_check_obligation(
                    journey,
                    suite,
                    assessment,
                    concern,
                    interpreted_surface_ids(design, concern.category) or assessment.surface_ids,
                )
                obligations_by_execution.setdefault(
                    _execution_key(concern.execution_change, concern.scenario_type), []
                ).append(obligation)
            else:
                gaps.append(
                    _gap(
                        suite,
                        assessment,
                        ScenarioSource.USER_REQUIREMENT,
                        concern.scenario_type,
                        check,
                        source_check_index=check_index,
                    )
                )
    for assessment in registry.assessments:
        if assessment.category in represented_categories:
            continue
        if assessment.applicability is CategoryApplicability.APPLICABLE:
            scenario_type = _preferred_strategy(assessment)
            execution_change = resolve_automatic_execution_change(
                design, assessment.category, scenario_type
            )
            obligations_by_execution.setdefault(
                _execution_key(execution_change, scenario_type), []
            ).append(_automatic_obligation(journey, suite, assessment, scenario_type))
        else:
            gaps.append(
                _gap(
                    suite,
                    assessment,
                    ScenarioSource.AUTOMATIC_QA,
                    _preferred_strategy(assessment),
                    f"Apply the {assessment.category.value.replace('_', ' ')} testing lens.",
                )
            )
    scenarios = tuple(
        _scenario_for_execution(journey, suite, key, tuple(obligations))
        for key, obligations in obligations_by_execution.items()
    )
    generated_suite = ScenarioSuite.model_validate(
        {
            **suite.model_dump(mode="python"),
            "suite_id": _derived_identifier(
                "suite", suite.suite_id, *(item.scenario_id for item in scenarios)
            ),
            "scenarios": tuple(scenarios),
        }
    )
    return ScenarioGenerationResult(
        generation_id=_derived_identifier(
            "generation", design.design_id, registry.registry_id, generated_suite.suite_id
        ),
        design_id=design.design_id,
        registry_id=registry.registry_id,
        suite=generated_suite,
        gaps=tuple(gaps),
    )


def _validate_generation_inputs(
    journey: ResolvedJourney,
    suite: ScenarioSuite,
    design: ValidatedTestDesign,
    registry: TestingCategoryRegistryResult,
) -> None:
    if suite.scenarios:
        raise ScenarioGenerationError("Scenario generation requires an empty accepted suite.")
    if not suite.readiness.executable or not suite.readiness.independently_auditable:
        raise ScenarioGenerationError("Scenario generation requires ready discovery evidence.")
    if {journey.journey_id, design.journey_id, registry.journey_id} != {suite.journey_id}:
        raise ScenarioGenerationError("Scenario generation inputs describe different journeys.")
    if design.readiness_id != suite.readiness.readiness_id or registry.readiness_id != (
        suite.readiness.readiness_id
    ):
        raise ScenarioGenerationError(
            "Scenario generation inputs use different readiness decisions."
        )
    if registry.untranslated_check_indexes:
        raise ScenarioGenerationError("Every user check needs an accepted TestDesigner mapping.")
    concerns_by_check = _concerns_by_check(design)
    if set(concerns_by_check) != set(range(len(journey.checks))):
        raise ScenarioGenerationError("Every user check needs one accepted testing concern.")
    assessment_ids = {item.assessment_id for item in registry.assessments}
    registry_concern_ids = {
        concern_id for item in registry.assessments for concern_id in item.concern_ids
    }
    design_concern_ids = {item.concern_id for item in design.concerns}
    if registry_concern_ids != design_concern_ids:
        raise ScenarioGenerationError("The registry does not match the accepted concerns.")
    registry_interpretation_ids = {
        interpretation_id
        for item in registry.assessments
        for interpretation_id in item.interpretation_ids
    }
    design_interpretation_ids = {item.interpretation_id for item in design.category_interpretations}
    if registry_interpretation_ids != design_interpretation_ids:
        raise ScenarioGenerationError("The registry does not match the accepted interpretations.")
    if len(assessment_ids) != len(registry.assessments):
        raise ScenarioGenerationError("Registry assessment identifiers must be unique.")
    functional = next(
        (item for item in registry.assessments if item.category is CoverageCategory.FUNCTIONAL),
        None,
    )
    if functional is None or functional.applicability is not CategoryApplicability.APPLICABLE:
        raise ScenarioGenerationError("The verified journey needs an applicable functional lens.")


def _concerns_by_check(
    design: ValidatedTestDesign,
) -> dict[int, tuple[TranslatedTestingConcern, ...]]:
    return {
        index: tuple(item for item in design.concerns if item.source_check_index == index)
        for index in {item.source_check_index for item in design.concerns}
    }


def _execution_key(
    execution_change: ScenarioExecutionChange, scenario_type: ScenarioType
) -> tuple[ScenarioExecutionChange, ScenarioType]:
    if execution_change is ScenarioExecutionChange.ACCEPTED_ROUTE:
        return execution_change, ScenarioType.HAPPY_PATH
    return execution_change, scenario_type


def _baseline_obligation(
    journey: ResolvedJourney,
    suite: ScenarioSuite,
    assessment: CategoryAssessment,
) -> ScenarioCoverageObligation:
    outcomes = tuple(
        _expected_outcome(
            suite,
            assessment,
            requirement.statement,
            ExpectedOutcomeAuthority.USER_REQUIREMENT,
            f"baseline_{index}",
            requirement_surface_ids(suite, assessment.surface_ids, requirement),
        )
        for index, requirement in enumerate(suite.readiness.requirements)
    )
    return _obligation(
        journey,
        suite,
        assessment,
        ScenarioSource.USER_REQUIREMENT,
        ScenarioType.HAPPY_PATH,
        "verified_journey",
        outcomes,
    )


def _user_check_obligation(
    journey: ResolvedJourney,
    suite: ScenarioSuite,
    assessment: CategoryAssessment,
    concern: TranslatedTestingConcern,
    surface_ids: tuple[RecordIdentifier, ...],
) -> ScenarioCoverageObligation:
    if ExpectedOutcomeAuthority.USER_REQUIREMENT not in assessment.expected_outcome_authorities:
        raise ScenarioGenerationError("A user-check scenario needs user-requirement authority.")
    outcome = _expected_outcome(
        suite,
        assessment,
        concern.expected_outcome,
        ExpectedOutcomeAuthority.USER_REQUIREMENT,
        concern.concern_id,
        surface_ids,
    )
    return _obligation(
        journey,
        suite,
        assessment,
        ScenarioSource.USER_REQUIREMENT,
        concern.scenario_type,
        concern.concern_id,
        (outcome,),
        surface_ids=surface_ids,
        source_check_indexes=(concern.source_check_index,),
    )


def _automatic_obligation(
    journey: ResolvedJourney,
    suite: ScenarioSuite,
    assessment: CategoryAssessment,
    scenario_type: ScenarioType,
) -> ScenarioCoverageObligation:
    source = (
        ScenarioSource.EXPLORATORY_SUGGESTION
        if scenario_type is ScenarioType.EXPLORATORY
        else ScenarioSource.AUTOMATIC_QA
    )
    statements, authority = _automatic_expected_statements(journey, assessment)
    outcomes = tuple(
        _expected_outcome(
            suite,
            assessment,
            statement,
            authority,
            f"automatic_{index}",
            assessment.surface_ids,
        )
        for index, statement in enumerate(statements)
    )
    return _obligation(
        journey,
        suite,
        assessment,
        source,
        scenario_type,
        "automatic",
        outcomes,
        required=source is not ScenarioSource.EXPLORATORY_SUGGESTION,
    )


def _automatic_expected_statements(
    journey: ResolvedJourney,
    assessment: CategoryAssessment,
) -> tuple[tuple[str, ...], ExpectedOutcomeAuthority]:
    authorities = assessment.expected_outcome_authorities
    if (
        assessment.category is CoverageCategory.SECURITY
        and journey.security_rules
        and ExpectedOutcomeAuthority.DECLARED_RULE in authorities
    ):
        return journey.security_rules, ExpectedOutcomeAuthority.DECLARED_RULE
    if ExpectedOutcomeAuthority.DETERMINISTIC_INVARIANT in authorities:
        return (_INVARIANT_BY_CATEGORY[assessment.category],), (
            ExpectedOutcomeAuthority.DETERMINISTIC_INVARIANT
        )
    if ExpectedOutcomeAuthority.ACCEPTED_BASELINE in authorities:
        return (journey.objective,), ExpectedOutcomeAuthority.ACCEPTED_BASELINE
    if ExpectedOutcomeAuthority.USER_REQUIREMENT in authorities:
        return (journey.objective,), ExpectedOutcomeAuthority.USER_REQUIREMENT
    raise ScenarioGenerationError("An applicable category has no expected-outcome authority.")


def _obligation(
    journey: ResolvedJourney,
    suite: ScenarioSuite,
    assessment: CategoryAssessment,
    source: ScenarioSource,
    scenario_type: ScenarioType,
    semantic_key: str,
    outcomes: tuple[ScenarioExpectedOutcome, ...],
    *,
    required: bool = True,
    surface_ids: tuple[RecordIdentifier, ...] | None = None,
    source_check_indexes: tuple[int, ...] = (),
) -> ScenarioCoverageObligation:
    return ScenarioCoverageObligation(
        obligation_id=_derived_identifier(
            "obligation",
            suite.readiness.readiness_id,
            source.value,
            assessment.category.value,
            scenario_type.value,
            semantic_key,
        ),
        category_assessment_id=assessment.assessment_id,
        source=source,
        category=assessment.category,
        strategy=scenario_type,
        coverage_cell_id=_coverage_cell_id(journey, assessment.category, scenario_type),
        required_surface_ids=surface_ids or assessment.surface_ids,
        expected_outcomes=outcomes,
        source_check_indexes=source_check_indexes,
        required=required,
    )


def _scenario_for_execution(
    journey: ResolvedJourney,
    suite: ScenarioSuite,
    key: tuple[ScenarioExecutionChange, ScenarioType],
    obligations: tuple[ScenarioCoverageObligation, ...],
) -> ScenarioDefinition:
    execution_change, scenario_type = key
    procedure_id = suite.readiness.procedure_id
    if procedure_id is None:
        raise ScenarioGenerationError("An executable scenario needs an approved procedure.")
    semantic_key = execution_change.value
    execution_name = execution_change.value.replace("_", " ")
    return ScenarioDefinition(
        scenario_id=_derived_identifier(
            "scenario",
            suite.readiness.readiness_id,
            semantic_key,
            scenario_type.value,
            *(item.obligation_id for item in obligations),
        ),
        journey_id=journey.journey_id,
        readiness_id=suite.readiness.readiness_id,
        execution_change=execution_change,
        scenario_type=scenario_type,
        name=f"{journey.name}: {execution_name}"[:200],
        objective=journey.objective,
        variation=_VARIATION_BY_EXECUTION_CHANGE[execution_change],
        route_procedure_id=procedure_id,
        setup=suite.readiness.required_setup,
        execution_capabilities=_execution_capabilities(execution_change),
        declared_value_names=suite.readiness.required_declared_value_names,
        obligations=obligations,
        required=any(item.required for item in obligations),
    )


def _execution_capabilities(
    execution_change: ScenarioExecutionChange,
) -> tuple[DeviceCapabilitySpec, ...]:
    if execution_change is ScenarioExecutionChange.LIFECYCLE_INTERRUPTION:
        return (AppStateCapabilitySpec(action="background_resume"),)
    return ()


def _expected_outcome(
    suite: ScenarioSuite,
    assessment: CategoryAssessment,
    statement: str,
    authority: ExpectedOutcomeAuthority,
    semantic_key: str,
    surface_ids: tuple[RecordIdentifier, ...],
) -> ScenarioExpectedOutcome:
    outcome_id = _derived_identifier(
        "outcome", suite.readiness.readiness_id, assessment.category.value, semantic_key, statement
    )
    return ScenarioExpectedOutcome(
        outcome_id=outcome_id,
        statement=statement,
        authority=authority,
        required_surface_ids=surface_ids,
        evidence_requirements=tuple(
            ScenarioEvidenceRequirement(
                requirement_id=_derived_identifier("evidence", outcome_id, owner.value),
                owner=owner,
                description=_EVIDENCE_DESCRIPTIONS[owner],
            )
            for owner in assessment.evidence_owners
        ),
    )


def _gap(
    suite: ScenarioSuite,
    assessment: CategoryAssessment,
    source: ScenarioSource,
    scenario_type: ScenarioType,
    statement: str,
    *,
    source_check_index: int | None = None,
) -> ScenarioGenerationGap:
    return ScenarioGenerationGap(
        gap_id=_derived_identifier(
            "gap",
            suite.readiness.readiness_id,
            source.value,
            assessment.category.value,
            scenario_type.value,
            str(source_check_index),
        ),
        category_assessment_id=assessment.assessment_id,
        source=source,
        category=assessment.category,
        scenario_type=scenario_type,
        source_check_index=source_check_index,
        statement=statement,
        reason=assessment.reason,
    )


def _preferred_strategy(assessment: CategoryAssessment) -> ScenarioType:
    only_user_authority = assessment.expected_outcome_authorities == (
        ExpectedOutcomeAuthority.USER_REQUIREMENT,
    )
    if only_user_authority and ScenarioType.HAPPY_PATH in assessment.scenario_types:
        return ScenarioType.HAPPY_PATH
    preferred = _AUTOMATIC_STRATEGY.get(assessment.category, ScenarioType.HAPPY_PATH)
    return (
        preferred
        if preferred in assessment.scenario_types
        else (assessment.scenario_types[0] if assessment.scenario_types else preferred)
    )


def _coverage_cell_id(
    journey: ResolvedJourney,
    category: CoverageCategory,
    scenario_type: ScenarioType,
) -> str | None:
    return next(
        (
            item.cell_id
            for item in journey.coverage
            if item.category is category and item.scenario_type is scenario_type
        ),
        None,
    )


def _derived_identifier(prefix: str, *parts: str) -> str:
    digest = sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[: 63 - len(prefix)]}"
