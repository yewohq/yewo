"""Deterministic QA-lens selection for discovered application surfaces."""

from __future__ import annotations

from hashlib import sha256
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yewo.artifacts import RecordIdentifier
from yewo.project_config import (
    CapabilityKind,
    CoverageCategory,
    ResolvedJourney,
    ScenarioType,
)
from yewo.scenarios.records import (
    DiscoveredTestSurface,
    ScenarioSuite,
    ScenarioText,
    TestDesignCategoryInterpretation,
    TranslatedTestingConcern,
)
from yewo.scenarios.taxonomy import (
    ExpectedOutcomeAuthority,
    RequiredEvidenceOwner,
    ScenarioSurfaceKind,
)


class TestingCategoryScope(StrEnum):
    """Level at which one QA lens operates."""

    SURFACE = "surface"
    ENVIRONMENT = "environment"
    TARGET_MATRIX = "target_matrix"
    APPLICATION_LIFECYCLE = "application_lifecycle"
    HISTORICAL_COMPARISON = "historical_comparison"


class CategoryApplicability(StrEnum):
    """Whether one internal QA lens can produce scenarios."""

    APPLICABLE = "applicable"
    NOT_APPLICABLE = "not_applicable"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNSUPPORTED = "unsupported"


class TestingRegistryContext(BaseModel):
    """Run-level facts needed by non-surface testing categories."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_count: int = Field(default=1, ge=1, le=1_000)
    available_capabilities: frozenset[CapabilityKind] | None = None
    has_regression_baseline: bool = False
    supports_installation_update: bool = False


class CategoryAssessment(BaseModel):
    """Deterministic applicability result for one internal QA lens."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    assessment_id: RecordIdentifier
    category: CoverageCategory
    scope: TestingCategoryScope
    applicability: CategoryApplicability
    surface_ids: tuple[RecordIdentifier, ...] = ()
    scenario_types: tuple[ScenarioType, ...] = ()
    required_capabilities: tuple[CapabilityKind, ...] = ()
    expected_outcome_authorities: tuple[ExpectedOutcomeAuthority, ...] = ()
    evidence_owners: tuple[RequiredEvidenceOwner, ...] = ()
    concern_ids: tuple[RecordIdentifier, ...] = ()
    interpretation_ids: tuple[RecordIdentifier, ...] = ()
    reason: ScenarioText

    @model_validator(mode="after")
    def _validate_applicability(self) -> Self:
        linked = (
            self.surface_ids,
            self.scenario_types,
            self.expected_outcome_authorities,
            self.evidence_owners,
        )
        if self.applicability is CategoryApplicability.APPLICABLE and any(
            not values for values in linked
        ):
            raise ValueError("An applicable category needs surfaces, strategies, and evidence.")
        for values in (
            *linked,
            self.required_capabilities,
            self.concern_ids,
            self.interpretation_ids,
        ):
            if len(values) != len(set(values)):
                raise ValueError("Category assessment references must be unique.")
        return self


class TestingCategoryRegistryResult(BaseModel):
    """Complete internal QA-lens decision for one ready journey."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    registry_id: RecordIdentifier
    journey_id: RecordIdentifier
    readiness_id: RecordIdentifier
    assessments: tuple[CategoryAssessment, ...] = Field(min_length=1)
    untranslated_check_indexes: tuple[int, ...] = ()

    @model_validator(mode="after")
    def _validate_categories(self) -> Self:
        categories = tuple(item.category for item in self.assessments)
        if len(categories) != len(set(categories)):
            raise ValueError("The registry must assess each testing category once.")
        if len(self.untranslated_check_indexes) != len(set(self.untranslated_check_indexes)):
            raise ValueError("Untranslated check indexes must be unique.")
        return self


class TestingCategoryRegistryError(ValueError):
    """Testing categories cannot be evaluated from the supplied evidence."""


_ALL_SURFACES = frozenset(ScenarioSurfaceKind)
_OPERATIONAL_SURFACES = frozenset(
    {
        ScenarioSurfaceKind.OPERATION,
        ScenarioSurfaceKind.INPUT,
        ScenarioSurfaceKind.SELECTION,
        ScenarioSurfaceKind.NAVIGATION,
        ScenarioSurfaceKind.STATE_CHANGE,
        ScenarioSurfaceKind.PERSISTENCE_BOUNDARY,
        ScenarioSurfaceKind.RECOVERY_PATH,
        ScenarioSurfaceKind.SYSTEM_INTERACTION,
    }
)
_SURFACES_BY_CATEGORY = {
    CoverageCategory.FUNCTIONAL: _OPERATIONAL_SURFACES,
    CoverageCategory.DATA_CONSISTENCY: frozenset(),
    CoverageCategory.STATE: frozenset(
        {
            ScenarioSurfaceKind.INPUT,
            ScenarioSurfaceKind.SELECTION,
            ScenarioSurfaceKind.STATE_CHANGE,
            ScenarioSurfaceKind.PERSISTENCE_BOUNDARY,
        }
    ),
    CoverageCategory.INPUT: frozenset({ScenarioSurfaceKind.INPUT}),
    CoverageCategory.NAVIGATION: frozenset({ScenarioSurfaceKind.NAVIGATION}),
    CoverageCategory.ERROR_HANDLING: frozenset(
        {ScenarioSurfaceKind.ERROR_STATE, ScenarioSurfaceKind.RECOVERY_PATH}
    ),
    CoverageCategory.UI_VISUAL: _ALL_SURFACES,
    CoverageCategory.SECURITY: frozenset(
        {ScenarioSurfaceKind.INPUT, ScenarioSurfaceKind.PRESENTATION}
    ),
    CoverageCategory.ACCESSIBILITY: _ALL_SURFACES,
    CoverageCategory.PERFORMANCE: _OPERATIONAL_SURFACES,
    CoverageCategory.NETWORK: frozenset({ScenarioSurfaceKind.SYSTEM_INTERACTION}),
    CoverageCategory.LIFECYCLE: frozenset(
        {
            ScenarioSurfaceKind.OPERATION,
            ScenarioSurfaceKind.INPUT,
            ScenarioSurfaceKind.STATE_CHANGE,
            ScenarioSurfaceKind.PERSISTENCE_BOUNDARY,
            ScenarioSurfaceKind.RECOVERY_PATH,
        }
    ),
    CoverageCategory.PERMISSION: frozenset({ScenarioSurfaceKind.SYSTEM_INTERACTION}),
    CoverageCategory.COMPATIBILITY: _ALL_SURFACES,
    CoverageCategory.INSTALLATION: _ALL_SURFACES,
    CoverageCategory.LOCALIZATION: frozenset(
        {
            ScenarioSurfaceKind.INPUT,
            ScenarioSurfaceKind.SELECTION,
            ScenarioSurfaceKind.PRESENTATION,
        }
    ),
    CoverageCategory.REGRESSION: _ALL_SURFACES,
}
_SCOPE_BY_CATEGORY = {
    CoverageCategory.NETWORK: TestingCategoryScope.ENVIRONMENT,
    CoverageCategory.LIFECYCLE: TestingCategoryScope.ENVIRONMENT,
    CoverageCategory.PERMISSION: TestingCategoryScope.ENVIRONMENT,
    CoverageCategory.LOCALIZATION: TestingCategoryScope.ENVIRONMENT,
    CoverageCategory.COMPATIBILITY: TestingCategoryScope.TARGET_MATRIX,
    CoverageCategory.INSTALLATION: TestingCategoryScope.APPLICATION_LIFECYCLE,
    CoverageCategory.REGRESSION: TestingCategoryScope.HISTORICAL_COMPARISON,
}
_STRATEGIES_BY_CATEGORY = {
    CoverageCategory.FUNCTIONAL: (ScenarioType.HAPPY_PATH,),
    CoverageCategory.DATA_CONSISTENCY: (ScenarioType.HAPPY_PATH,),
    CoverageCategory.STATE: (
        ScenarioType.HAPPY_PATH,
        ScenarioType.INTERRUPTION,
        ScenarioType.RECOVERY,
    ),
    CoverageCategory.INPUT: (ScenarioType.HAPPY_PATH,),
    CoverageCategory.NAVIGATION: (ScenarioType.HAPPY_PATH, ScenarioType.RECOVERY),
    CoverageCategory.ERROR_HANDLING: (ScenarioType.NEGATIVE, ScenarioType.RECOVERY),
    CoverageCategory.UI_VISUAL: (ScenarioType.HAPPY_PATH,),
    CoverageCategory.SECURITY: (ScenarioType.NEGATIVE,),
    CoverageCategory.ACCESSIBILITY: (ScenarioType.HAPPY_PATH,),
    CoverageCategory.PERFORMANCE: (ScenarioType.HAPPY_PATH,),
    CoverageCategory.NETWORK: (ScenarioType.INTERRUPTION, ScenarioType.RECOVERY),
    CoverageCategory.LIFECYCLE: (ScenarioType.INTERRUPTION, ScenarioType.RECOVERY),
    CoverageCategory.PERMISSION: (ScenarioType.NEGATIVE, ScenarioType.RECOVERY),
    CoverageCategory.COMPATIBILITY: (ScenarioType.HAPPY_PATH,),
    CoverageCategory.INSTALLATION: (ScenarioType.HAPPY_PATH, ScenarioType.RECOVERY),
    CoverageCategory.LOCALIZATION: (ScenarioType.HAPPY_PATH, ScenarioType.BOUNDARY),
    CoverageCategory.REGRESSION: (ScenarioType.HAPPY_PATH,),
}
_CAPABILITIES_BY_CATEGORY = {
    CoverageCategory.NETWORK: (CapabilityKind.NETWORK,),
    CoverageCategory.LIFECYCLE: (CapabilityKind.APP_STATE,),
    CoverageCategory.PERMISSION: (CapabilityKind.PERMISSION,),
    CoverageCategory.LOCALIZATION: (CapabilityKind.LOCALE,),
    CoverageCategory.INSTALLATION: (CapabilityKind.APP_STATE,),
}
_EVIDENCE_BY_CATEGORY = {
    CoverageCategory.DATA_CONSISTENCY: (RequiredEvidenceOwner.APPLICATION_OBSERVATION,),
    CoverageCategory.ACCESSIBILITY: (RequiredEvidenceOwner.ACCESSIBILITY_INSPECTION,),
    CoverageCategory.PERFORMANCE: (RequiredEvidenceOwner.MEASUREMENT,),
    CoverageCategory.NETWORK: (
        RequiredEvidenceOwner.DEVICE_RECEIPT,
        RequiredEvidenceOwner.APPLICATION_OBSERVATION,
    ),
    CoverageCategory.LIFECYCLE: (
        RequiredEvidenceOwner.DEVICE_RECEIPT,
        RequiredEvidenceOwner.APPLICATION_OBSERVATION,
    ),
    CoverageCategory.PERMISSION: (
        RequiredEvidenceOwner.DEVICE_RECEIPT,
        RequiredEvidenceOwner.APPLICATION_OBSERVATION,
    ),
    CoverageCategory.COMPATIBILITY: (RequiredEvidenceOwner.APPLICATION_OBSERVATION,),
    CoverageCategory.INSTALLATION: (
        RequiredEvidenceOwner.DEVICE_RECEIPT,
        RequiredEvidenceOwner.APPLICATION_OBSERVATION,
    ),
    CoverageCategory.REGRESSION: (RequiredEvidenceOwner.APPLICATION_OBSERVATION,),
}


def evaluate_testing_categories(
    journey: ResolvedJourney,
    suite: ScenarioSuite,
    *,
    concerns: tuple[TranslatedTestingConcern, ...] = (),
    interpretations: tuple[TestDesignCategoryInterpretation, ...] = (),
    context: TestingRegistryContext | None = None,
) -> TestingCategoryRegistryResult:
    """Apply internal QA knowledge without exposing technical choices to the user."""
    if suite.journey_id != journey.journey_id:
        raise TestingCategoryRegistryError("The journey and discovery suite do not match.")
    if not suite.readiness.executable or not suite.readiness.independently_auditable:
        raise TestingCategoryRegistryError("Testing categories require ready discovery evidence.")
    _validate_concerns(journey, concerns)
    _validate_interpretations(suite, interpretations)
    selected_context = context or TestingRegistryContext()
    by_category = {
        category: tuple(item for item in concerns if item.category is category)
        for category in CoverageCategory
    }
    interpretations_by_category = {
        category: tuple(item for item in interpretations if item.category is category)
        for category in CoverageCategory
    }
    assessments = tuple(
        _assess_category(
            journey,
            suite,
            category,
            by_category[category],
            interpretations_by_category[category],
            selected_context,
        )
        for category in CoverageCategory
    )
    translated_indexes = {item.source_check_index for item in concerns}
    untranslated = tuple(
        index for index in range(len(journey.checks)) if index not in translated_indexes
    )
    return TestingCategoryRegistryResult(
        registry_id=_derived_identifier(
            "registry",
            suite.readiness.readiness_id,
            *(item.concern_id for item in concerns),
            *(item.interpretation_id for item in interpretations),
        ),
        journey_id=journey.journey_id,
        readiness_id=suite.readiness.readiness_id,
        assessments=assessments,
        untranslated_check_indexes=untranslated,
    )


def _validate_concerns(
    journey: ResolvedJourney,
    concerns: tuple[TranslatedTestingConcern, ...],
) -> None:
    concern_ids = tuple(item.concern_id for item in concerns)
    if len(concern_ids) != len(set(concern_ids)):
        raise TestingCategoryRegistryError("Translated concern identifiers must be unique.")
    for concern in concerns:
        if concern.source_check_index >= len(journey.checks):
            raise TestingCategoryRegistryError("A translated concern references an unknown check.")
        if journey.checks[concern.source_check_index] != concern.source_check:
            raise TestingCategoryRegistryError("A translated concern changed the user's check.")


def _validate_interpretations(
    suite: ScenarioSuite,
    interpretations: tuple[TestDesignCategoryInterpretation, ...],
) -> None:
    interpretation_ids = tuple(item.interpretation_id for item in interpretations)
    if len(interpretation_ids) != len(set(interpretation_ids)):
        raise TestingCategoryRegistryError("Category interpretation identifiers must be unique.")
    known_surfaces = {item.surface_id for item in suite.surfaces}
    if any(
        not set(interpretation.surface_ids).issubset(known_surfaces)
        for interpretation in interpretations
    ):
        raise TestingCategoryRegistryError("A category interpretation invented a test surface.")
    system_surface_ids = {
        item.surface_id
        for item in suite.surfaces
        if item.kind is ScenarioSurfaceKind.SYSTEM_INTERACTION
    }
    if any(
        interpretation.category is CoverageCategory.PERMISSION
        and not set(interpretation.surface_ids).issubset(system_surface_ids)
        for interpretation in interpretations
    ):
        raise TestingCategoryRegistryError(
            "Permission testing requires a native system interaction."
        )


def _assess_category(
    journey: ResolvedJourney,
    suite: ScenarioSuite,
    category: CoverageCategory,
    concerns: tuple[TranslatedTestingConcern, ...],
    interpretations: tuple[TestDesignCategoryInterpretation, ...],
    context: TestingRegistryContext,
) -> CategoryAssessment:
    eligible = _eligible_surfaces(
        journey,
        suite,
        category,
        bool(concerns),
        interpretations,
    )
    status, reason = _applicability(
        category,
        bool(eligible),
        bool(concerns),
        bool(interpretations),
        context,
    )
    capabilities = _CAPABILITIES_BY_CATEGORY.get(category, ())
    unavailable = (
        tuple(item for item in capabilities if item not in context.available_capabilities)
        if context.available_capabilities is not None
        else ()
    )
    if status is CategoryApplicability.APPLICABLE and unavailable:
        status = CategoryApplicability.UNSUPPORTED
        reason = "The selected target does not support: " + ", ".join(
            item.value for item in unavailable
        )
    strategies = tuple(
        dict.fromkeys(
            (
                *_STRATEGIES_BY_CATEGORY[category],
                *(item.scenario_type for item in concerns),
                *(strategy for item in interpretations for strategy in item.scenario_types),
            )
        )
    )
    authorities = _outcome_authorities(journey, category, concerns, context)
    evidence_owners = _EVIDENCE_BY_CATEGORY.get(
        category,
        (
            RequiredEvidenceOwner.ACTION_OUTCOME,
            RequiredEvidenceOwner.APPLICATION_OBSERVATION,
        ),
    )
    if status is CategoryApplicability.APPLICABLE and not authorities:
        status = CategoryApplicability.INSUFFICIENT_EVIDENCE
        reason = "This testing lens has no trustworthy expected-result authority."
    if status is not CategoryApplicability.APPLICABLE:
        strategies = ()
        authorities = ()
        evidence_owners = ()
    return CategoryAssessment(
        assessment_id=_derived_identifier(
            "assessment",
            suite.readiness.readiness_id,
            category.value,
            *(item.concern_id for item in concerns),
        ),
        category=category,
        scope=_SCOPE_BY_CATEGORY.get(category, TestingCategoryScope.SURFACE),
        applicability=status,
        surface_ids=tuple(item.surface_id for item in eligible),
        scenario_types=strategies,
        required_capabilities=capabilities,
        expected_outcome_authorities=authorities,
        evidence_owners=evidence_owners,
        concern_ids=tuple(item.concern_id for item in concerns),
        interpretation_ids=tuple(item.interpretation_id for item in interpretations),
        reason=reason,
    )


def _eligible_surfaces(
    journey: ResolvedJourney,
    suite: ScenarioSuite,
    category: CoverageCategory,
    has_concern: bool,
    interpretations: tuple[TestDesignCategoryInterpretation, ...],
) -> tuple[DiscoveredTestSurface, ...]:
    kinds = _SURFACES_BY_CATEGORY[category]
    if (has_concern or interpretations) and category in {
        CoverageCategory.ERROR_HANDLING,
        CoverageCategory.NETWORK,
        CoverageCategory.INSTALLATION,
    }:
        kinds = _OPERATIONAL_SURFACES
    interpreted_surface_ids = {
        surface_id for item in interpretations for surface_id in item.surface_ids
    }
    surfaces = tuple(
        item
        for item in suite.surfaces
        if item.kind in kinds or item.surface_id in interpreted_surface_ids
    )
    if category is CoverageCategory.SECURITY and not has_concern and not interpretations:
        secret_names = {name for name, value in journey.declared_values.items() if value.secret}
        surfaces = tuple(
            item for item in surfaces if secret_names.intersection(item.declared_value_names)
        )
        if journey.security_rules and not surfaces:
            surfaces = tuple(item for item in suite.surfaces if item.kind in kinds)
    return surfaces


def _applicability(
    category: CoverageCategory,
    has_surfaces: bool,
    has_concern: bool,
    has_interpretation: bool,
    context: TestingRegistryContext,
) -> tuple[CategoryApplicability, str]:
    has_semantic_request = has_concern or has_interpretation
    if category is CoverageCategory.COMPATIBILITY and context.target_count < 2:
        return CategoryApplicability.NOT_APPLICABLE, "Compatibility needs multiple targets."
    if category is CoverageCategory.REGRESSION and not context.has_regression_baseline:
        return (
            CategoryApplicability.INSUFFICIENT_EVIDENCE,
            "Regression testing needs an accepted historical baseline.",
        )
    if category is CoverageCategory.INSTALLATION and not context.supports_installation_update:
        return (
            CategoryApplicability.UNSUPPORTED
            if has_semantic_request
            else CategoryApplicability.NOT_APPLICABLE,
            "Installation and update execution is not available yet.",
        )
    if category is CoverageCategory.PERFORMANCE and not has_semantic_request:
        return (
            CategoryApplicability.INSUFFICIENT_EVIDENCE,
            "Performance grading needs a user requirement or accepted threshold.",
        )
    if not has_surfaces:
        return (
            CategoryApplicability.INSUFFICIENT_EVIDENCE
            if has_semantic_request
            else CategoryApplicability.NOT_APPLICABLE,
            "Discovery did not expose a relevant test surface.",
        )
    return CategoryApplicability.APPLICABLE, (
        "A user-written check and discovery evidence support this testing lens."
        if has_concern
        else "TestDesigner and discovery evidence support this testing lens."
        if has_interpretation
        else "Automatic QA policy and discovery evidence support this testing lens."
    )


def _outcome_authorities(
    journey: ResolvedJourney,
    category: CoverageCategory,
    concerns: tuple[TranslatedTestingConcern, ...],
    context: TestingRegistryContext,
) -> tuple[ExpectedOutcomeAuthority, ...]:
    authorities: list[ExpectedOutcomeAuthority] = []
    if (
        category
        in {
            CoverageCategory.FUNCTIONAL,
            CoverageCategory.STATE,
            CoverageCategory.INPUT,
            CoverageCategory.NAVIGATION,
            CoverageCategory.COMPATIBILITY,
        }
        or concerns
    ):
        authorities.append(ExpectedOutcomeAuthority.USER_REQUIREMENT)
    if category is CoverageCategory.SECURITY and journey.security_rules:
        authorities.append(ExpectedOutcomeAuthority.DECLARED_RULE)
    if category in {
        CoverageCategory.DATA_CONSISTENCY,
        CoverageCategory.STATE,
        CoverageCategory.NAVIGATION,
        CoverageCategory.ERROR_HANDLING,
        CoverageCategory.UI_VISUAL,
        CoverageCategory.SECURITY,
        CoverageCategory.ACCESSIBILITY,
        CoverageCategory.NETWORK,
        CoverageCategory.LIFECYCLE,
        CoverageCategory.PERMISSION,
    }:
        authorities.append(ExpectedOutcomeAuthority.DETERMINISTIC_INVARIANT)
    if category is CoverageCategory.REGRESSION and context.has_regression_baseline:
        authorities.append(ExpectedOutcomeAuthority.ACCEPTED_BASELINE)
    return tuple(dict.fromkeys(authorities))


def _derived_identifier(prefix: str, *parts: str) -> str:
    digest = sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[: 63 - len(prefix)]}"
