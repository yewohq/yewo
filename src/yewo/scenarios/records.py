"""Evidence-grounded scenario-generation contracts."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yewo.artifacts import EvidenceApplicationIdentity, RecordIdentifier
from yewo.project_config.capabilities import DeviceCapabilitySpec, ScenarioPreconditionsSpec
from yewo.project_config.records import CoverageCategory, ScenarioType
from yewo.scenarios.taxonomy import (
    ExpectedOutcomeAuthority,
    RequiredEvidenceOwner,
    RequirementEvidenceState,
    ScenarioExecutionChange,
    ScenarioSource,
    ScenarioSurfaceKind,
)


ScenarioText = Annotated[str, Field(min_length=1, max_length=4_000)]


class TranslatedTestingConcern(BaseModel):
    """Internal interpretation of one user-written check."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    concern_id: RecordIdentifier
    source_check_index: int = Field(ge=0)
    source_check: ScenarioText
    category: CoverageCategory
    scenario_type: ScenarioType
    execution_change: ScenarioExecutionChange
    expected_outcome: ScenarioText


class TestDesignVariation(BaseModel):
    """One QA strategy and the concrete execution change it requires."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_type: ScenarioType
    execution_change: ScenarioExecutionChange


class TestDesignCategoryInterpretation(BaseModel):
    """Runtime-owned semantic interpretation of discovery evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    interpretation_id: RecordIdentifier
    category: CoverageCategory
    surface_ids: tuple[RecordIdentifier, ...] = Field(min_length=1, max_length=100)
    variations: tuple[TestDesignVariation, ...] = Field(min_length=1, max_length=20)
    reason: ScenarioText

    @model_validator(mode="after")
    def _validate_references(self) -> Self:
        if len(self.surface_ids) != len(set(self.surface_ids)):
            raise ValueError("A category interpretation cannot repeat a surface.")
        scenario_types = tuple(item.scenario_type for item in self.variations)
        if len(scenario_types) != len(set(scenario_types)):
            raise ValueError("A category interpretation cannot repeat a strategy.")
        return self

    @property
    def scenario_types(self) -> tuple[ScenarioType, ...]:
        """Return strategies for deterministic category assessment."""
        return tuple(item.scenario_type for item in self.variations)


class DiscoveryProvenance(BaseModel):
    """Accepted discovery evidence that supports a testing claim."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    discovery_run_id: RecordIdentifier
    discovery_objective_id: RecordIdentifier
    procedure_id: RecordIdentifier
    verification_id: RecordIdentifier
    observation_ids: tuple[RecordIdentifier, ...] = Field(min_length=1, max_length=500)
    procedure_step_indexes: tuple[int, ...] = Field(default=(), max_length=500)

    @model_validator(mode="after")
    def _validate_references(self) -> Self:
        _require_unique(self.observation_ids, "Discovery observations")
        _require_unique(self.procedure_step_indexes, "Discovery procedure steps")
        if any(index < 0 for index in self.procedure_step_indexes):
            raise ValueError("Discovery procedure step indexes cannot be negative.")
        return self


class DiscoveredTestSurface(BaseModel):
    """One evidence-backed application characteristic that can be tested."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    surface_id: RecordIdentifier
    journey_id: RecordIdentifier
    kind: ScenarioSurfaceKind
    summary: ScenarioText
    observable_outcomes: tuple[ScenarioText, ...] = Field(min_length=1, max_length=100)
    control_ids: tuple[RecordIdentifier, ...] = Field(default=(), max_length=100)
    state_ids: tuple[RecordIdentifier, ...] = Field(default=(), max_length=100)
    declared_value_names: tuple[RecordIdentifier, ...] = Field(default=(), max_length=100)
    provenance: DiscoveryProvenance

    @model_validator(mode="after")
    def _validate_references(self) -> Self:
        _require_unique(self.observable_outcomes, "Surface outcomes")
        _require_unique(self.control_ids, "Surface controls")
        _require_unique(self.state_ids, "Surface states")
        _require_unique(self.declared_value_names, "Surface declared values")
        return self


class JourneyRequirementEvidence(BaseModel):
    """Discovery support for the objective or one completion condition."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    requirement_id: RecordIdentifier
    kind: Literal["objective", "completion_condition"]
    statement: ScenarioText
    state: RequirementEvidenceState
    observation_ids: tuple[RecordIdentifier, ...] = Field(default=(), max_length=500)
    procedure_step_indexes: tuple[int, ...] = Field(default=(), max_length=500)
    reason: ScenarioText

    @model_validator(mode="after")
    def _validate_state(self) -> Self:
        _require_unique(self.observation_ids, "Requirement observations")
        _require_unique(self.procedure_step_indexes, "Requirement procedure steps")
        if any(index < 0 for index in self.procedure_step_indexes):
            raise ValueError("Requirement procedure step indexes cannot be negative.")
        has_evidence = bool(self.observation_ids and self.procedure_step_indexes)
        if self.state is RequirementEvidenceState.SUPPORTED and not has_evidence:
            raise ValueError("A supported requirement needs observations and procedure steps.")
        if self.state is RequirementEvidenceState.UNRESOLVED and (
            self.observation_ids or self.procedure_step_indexes
        ):
            raise ValueError("An unresolved requirement cannot claim accepted evidence.")
        return self


class ScenarioEvidenceReadiness(BaseModel):
    """Deterministic decision that discovery can support scenario generation."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    readiness_id: RecordIdentifier
    journey_id: RecordIdentifier
    application: EvidenceApplicationIdentity
    discovery_run_id: RecordIdentifier
    procedure_id: RecordIdentifier | None = None
    verification_id: RecordIdentifier | None = None
    requirements: tuple[JourneyRequirementEvidence, ...] = Field(min_length=1, max_length=100)
    required_declared_value_names: tuple[RecordIdentifier, ...] = Field(default=(), max_length=100)
    required_setup: ScenarioPreconditionsSpec = ScenarioPreconditionsSpec()
    blocking_action_risk_ids: tuple[RecordIdentifier, ...] = Field(default=(), max_length=100)
    readiness_blockers: tuple[ScenarioText, ...] = Field(default=(), max_length=100)
    executable: bool
    independently_auditable: bool
    reason: ScenarioText

    @model_validator(mode="after")
    def _validate_decision(self) -> Self:
        _require_unique(
            tuple(requirement.requirement_id for requirement in self.requirements),
            "Journey requirements",
        )
        _require_unique(self.required_declared_value_names, "Required declared values")
        _require_unique(self.blocking_action_risk_ids, "Blocking action risks")
        _require_unique(self.readiness_blockers, "Readiness blockers")
        all_supported = all(
            requirement.state is RequirementEvidenceState.SUPPORTED
            for requirement in self.requirements
        )
        has_verified_procedure = self.procedure_id is not None and self.verification_id is not None
        can_execute = (
            all_supported
            and has_verified_procedure
            and not self.blocking_action_risk_ids
            and not self.readiness_blockers
        )
        if self.executable != can_execute:
            raise ValueError(
                "Executable readiness requires supported requirements, a verified procedure, "
                "and no blocking action risks."
            )
        if self.independently_auditable and not self.executable:
            raise ValueError("Only an executable journey can be independently auditable.")
        return self


class ScenarioEvidenceRequirement(BaseModel):
    """Evidence that must exist before an expected outcome can be settled."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    requirement_id: RecordIdentifier
    owner: RequiredEvidenceOwner
    description: ScenarioText
    minimum_count: int = Field(default=1, ge=1, le=100)


class ScenarioExpectedOutcome(BaseModel):
    """Expected behavior and the authority allowed to define it."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    outcome_id: RecordIdentifier
    statement: ScenarioText
    authority: ExpectedOutcomeAuthority
    required_surface_ids: tuple[RecordIdentifier, ...] = Field(min_length=1, max_length=100)
    evidence_requirements: tuple[ScenarioEvidenceRequirement, ...] = Field(
        min_length=1, max_length=20
    )

    @model_validator(mode="after")
    def _validate_evidence_requirements(self) -> Self:
        _require_unique(self.required_surface_ids, "Expected-outcome test surfaces")
        _require_unique(
            tuple(requirement.requirement_id for requirement in self.evidence_requirements),
            "Expected-outcome evidence requirements",
        )
        return self


class ScenarioCoverageObligation(BaseModel):
    """One QA lens that must be settled during a scenario execution."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    obligation_id: RecordIdentifier
    category_assessment_id: RecordIdentifier
    source: ScenarioSource
    category: CoverageCategory
    strategy: ScenarioType
    coverage_cell_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{0,127}$")
    required_surface_ids: tuple[RecordIdentifier, ...] = Field(min_length=1, max_length=100)
    expected_outcomes: tuple[ScenarioExpectedOutcome, ...] = Field(min_length=1, max_length=100)
    source_check_indexes: tuple[int, ...] = Field(default=(), max_length=100)
    required: bool = True

    @model_validator(mode="after")
    def _validate_obligation(self) -> Self:
        _require_unique(self.required_surface_ids, "Coverage-obligation test surfaces")
        _require_unique(
            tuple(outcome.outcome_id for outcome in self.expected_outcomes),
            "Coverage-obligation expected outcomes",
        )
        if any(
            not set(outcome.required_surface_ids).issubset(self.required_surface_ids)
            for outcome in self.expected_outcomes
        ):
            raise ValueError(
                "Expected-outcome test surfaces must belong to their coverage obligation."
            )
        _require_unique(self.source_check_indexes, "Coverage-obligation source checks")
        if any(index < 0 for index in self.source_check_indexes):
            raise ValueError("Coverage-obligation source check indexes cannot be negative.")
        if self.source is ScenarioSource.EXPLORATORY_SUGGESTION and self.required:
            raise ValueError("An exploratory suggestion cannot be a required obligation.")
        return self


class ScenarioDefinition(BaseModel):
    """One executable test variation with one or more QA obligations."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    scenario_id: RecordIdentifier
    journey_id: RecordIdentifier
    readiness_id: RecordIdentifier
    execution_change: ScenarioExecutionChange
    scenario_type: ScenarioType
    name: str = Field(min_length=1, max_length=200)
    objective: ScenarioText
    variation: ScenarioText
    route_procedure_id: RecordIdentifier
    setup: ScenarioPreconditionsSpec = ScenarioPreconditionsSpec()
    execution_capabilities: tuple[DeviceCapabilitySpec, ...] = Field(default=(), max_length=20)
    declared_value_names: tuple[RecordIdentifier, ...] = Field(default=(), max_length=100)
    obligations: tuple[ScenarioCoverageObligation, ...] = Field(min_length=1, max_length=100)
    required: bool = True

    @model_validator(mode="after")
    def _validate_scenario(self) -> Self:
        _require_unique(self.declared_value_names, "Scenario declared values")
        _require_unique(
            tuple(obligation.obligation_id for obligation in self.obligations),
            "Scenario coverage obligations",
        )
        if self.required != any(obligation.required for obligation in self.obligations):
            raise ValueError("Scenario required state must match its coverage obligations.")
        if (
            self.execution_change is ScenarioExecutionChange.ACCEPTED_ROUTE
            and self.scenario_type is not ScenarioType.HAPPY_PATH
        ):
            raise ValueError("An accepted-route execution must use the happy-path scenario type.")
        return self


class ScenarioSuite(BaseModel):
    """One journey's accepted scenario-generation output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    suite_id: RecordIdentifier
    journey_id: RecordIdentifier
    application: EvidenceApplicationIdentity
    readiness: ScenarioEvidenceReadiness
    surfaces: tuple[DiscoveredTestSurface, ...] = ()
    scenarios: tuple[ScenarioDefinition, ...] = ()

    @model_validator(mode="after")
    def _validate_links(self) -> Self:
        if self.readiness.journey_id != self.journey_id:
            raise ValueError("Suite readiness must belong to the suite journey.")
        if self.readiness.application != self.application:
            raise ValueError("Suite readiness must describe the suite application build.")
        _require_unique(tuple(surface.surface_id for surface in self.surfaces), "Test surfaces")
        _require_unique(tuple(scenario.scenario_id for scenario in self.scenarios), "Scenarios")
        if self.scenarios and not (
            self.readiness.executable and self.readiness.independently_auditable
        ):
            raise ValueError("A suite can contain scenarios only when readiness permits them.")
        surface_ids = {surface.surface_id for surface in self.surfaces}
        for surface in self.surfaces:
            if surface.journey_id != self.journey_id:
                raise ValueError("Every test surface must belong to the suite journey.")
        for scenario in self.scenarios:
            self._validate_scenario_links(scenario, surface_ids)
        return self

    def _validate_scenario_links(
        self, scenario: ScenarioDefinition, surface_ids: set[RecordIdentifier]
    ) -> None:
        if scenario.journey_id != self.journey_id:
            raise ValueError("Every scenario must belong to the suite journey.")
        if scenario.readiness_id != self.readiness.readiness_id:
            raise ValueError("Every scenario must reference the suite readiness decision.")
        if scenario.route_procedure_id != self.readiness.procedure_id:
            raise ValueError("Every scenario must use the readiness-approved procedure.")
        for obligation in scenario.obligations:
            if not set(obligation.required_surface_ids).issubset(surface_ids):
                raise ValueError(
                    "A scenario obligation references a test surface outside the suite."
                )
        if not set(scenario.declared_value_names).issubset(
            self.readiness.required_declared_value_names
        ):
            raise ValueError("A scenario references a declared value outside readiness.")


class ScenarioGenerationGap(BaseModel):
    """One requested or automatic testing cell that cannot produce an obligation."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    gap_id: RecordIdentifier
    category_assessment_id: RecordIdentifier
    source: ScenarioSource
    category: CoverageCategory
    scenario_type: ScenarioType
    source_check_index: int | None = Field(default=None, ge=0)
    statement: ScenarioText
    reason: ScenarioText


class ScenarioGenerationResult(BaseModel):
    """Scenarios and explicit gaps produced from one accepted test design."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    generation_id: RecordIdentifier
    design_id: RecordIdentifier
    registry_id: RecordIdentifier
    suite: ScenarioSuite
    gaps: tuple[ScenarioGenerationGap, ...] = ()

    @model_validator(mode="after")
    def _validate_links(self) -> Self:
        _require_unique(tuple(item.gap_id for item in self.gaps), "Scenario-generation gaps")
        return self


def _require_unique(values: tuple[object, ...], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique.")
