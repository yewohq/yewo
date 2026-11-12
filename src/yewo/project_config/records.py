"""Strict public records for a Yewo YAML project file."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yewo.project_config.capabilities import ScenarioPreconditionsSpec


Identifier = Annotated[
    str, Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
]
ValueName = Annotated[str, Field(min_length=1, max_length=100, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]
UserCheck = Annotated[str, Field(min_length=1, max_length=4_000)]


class StrictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class DeviceSpec(StrictRecord):
    name: str | None = Field(default=None, min_length=1)
    platform: Literal["android", "ios"] | None = None
    headless: bool | None = None


class TargetSpec(StrictRecord):
    id: Identifier
    platform: Literal["android", "ios"]
    device: str = Field(min_length=1)
    app: str | None = Field(default=None, min_length=1)
    headless: bool | None = None


ScalarValue = str | int | float | bool


class DeclaredValueSpec(StrictRecord):
    value: ScalarValue = Field(repr=False)
    secret: bool = False
    refused: bool = False


EnvironmentValue = DeclaredValueSpec | ScalarValue


class PolicySpec(StrictRecord):
    waive: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_waivers(self) -> Self:
        if any(
            "." not in check_id or not reason.strip() for check_id, reason in self.waive.items()
        ):
            raise ValueError("Every policy waiver needs a semantic check ID and a reason.")
        return self


class CoverageCategory(StrEnum):
    """A general QA lens applied to a discovered application surface."""

    FUNCTIONAL = "functional"
    DATA_CONSISTENCY = "data_consistency"
    STATE = "state"
    INPUT = "input"
    NAVIGATION = "navigation"
    ERROR_HANDLING = "error_handling"
    UI_VISUAL = "ui_visual"
    SECURITY = "security"
    ACCESSIBILITY = "accessibility"
    PERFORMANCE = "performance"
    NETWORK = "network"
    LIFECYCLE = "lifecycle"
    PERMISSION = "permission"
    COMPATIBILITY = "compatibility"
    INSTALLATION = "installation"
    LOCALIZATION = "localization"
    REGRESSION = "regression"


class ScenarioType(StrEnum):
    """The variation strategy used within one testing category."""

    HAPPY_PATH = "happy_path"
    NEGATIVE = "negative"
    BOUNDARY = "boundary"
    EMPTY = "empty"
    CONCURRENCY = "concurrency"
    INTERRUPTION = "interruption"
    RECOVERY = "recovery"
    ROLE = "role"
    EXPLORATORY = "exploratory"


class CoveragePriority(StrEnum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"


class CoverageSpec(StrictRecord):
    category: CoverageCategory
    types: tuple[ScenarioType, ...] = Field(min_length=1)
    priority: CoveragePriority = CoveragePriority.P1


class GroundingProtocol(StrEnum):
    SOURCE_PIXELS = "source_pixels"
    NORMALIZED = "normalized"
    NORMALIZED_1000 = "normalized_1000"


class ReasoningSetting(StrEnum):
    """How much private reasoning one role may spend before it answers.

    An effort keeps reasoning and bounds it. Reasoning and the answer share one completion
    budget, so a role left to the provider can spend the whole budget thinking and return
    nothing.
    """

    DISABLED = "disabled"
    PROVIDER_DEFAULT = "provider_default"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class GrounderSelectionSpec(StrictRecord):
    model: str = Field(min_length=1)
    protocol: GroundingProtocol
    reasoning: ReasoningSetting = ReasoningSetting.PROVIDER_DEFAULT


class ModelSelectionSpec(StrictRecord):
    model: str = Field(min_length=1)
    reasoning: ReasoningSetting = ReasoningSetting.PROVIDER_DEFAULT


ModelSelection = str | ModelSelectionSpec


class ModelOverridesSpec(StrictRecord):
    planner: ModelSelection | None = None
    auditor: ModelSelection | None = None
    recovery: ModelSelection | None = None
    proposal_verifier: ModelSelection | None = None
    outcome_verifier: ModelSelection | None = None
    screen_analyzer: ModelSelection | None = None
    replay_preparer: ModelSelection | None = None
    test_designer: ModelSelection | None = None


class SettingsSpec(StrictRecord):
    device: DeviceSpec | str | None = None
    platform: Literal["android", "ios"] | None = None
    headless: bool | None = None
    app: str | None = Field(default=None, min_length=1)
    model: str | None = Field(default=None, min_length=1)
    models: ModelOverridesSpec | None = None
    grounder: GrounderSelectionSpec | None = None
    max_steps: int | None = Field(default=None, ge=1, le=500)
    max_duration_seconds: float | None = Field(default=None, gt=0, le=14_400)
    settle_time: float | None = Field(default=None, ge=0, le=5)
    output_dir: str | None = Field(default=None, min_length=1)
    memory_root: str | None = Field(default=None, min_length=1)
    record: bool | None = None
    durable_replay: bool | None = None


class JourneySpec(StrictRecord):
    id: Identifier | None = None
    name: str | None = Field(default=None, min_length=1)
    journey: str = Field(min_length=1)
    security_rules: tuple[str, ...] = ()
    checks: tuple[UserCheck, ...] = ()
    env: dict[ValueName, EnvironmentValue] = Field(default_factory=dict)
    preconditions: ScenarioPreconditionsSpec | None = None


class WorkflowSpec(SettingsSpec):
    env: dict[ValueName, EnvironmentValue] = Field(default_factory=dict)
    secrets: tuple[ValueName, ...] | ValueName = ()
    journeys: tuple[JourneySpec, ...] | None = None
    journey: str | None = Field(default=None, min_length=1)
    id: Identifier | None = None
    name: str | None = Field(default=None, min_length=1)
    security_rules: tuple[str, ...] = ()
    checks: tuple[UserCheck, ...] = ()
    coverage: tuple[CoverageSpec, ...] | None = None
    actors: dict[ValueName, dict[ValueName, EnvironmentValue]] | None = None
    policy: PolicySpec | None = None
    preconditions: ScenarioPreconditionsSpec | None = None
    targets: tuple[TargetSpec, ...] | None = None

    @model_validator(mode="after")
    def _validate_journeys(self) -> Self:
        has_list = self.journeys is not None
        has_inline = self.journey is not None
        if has_list == has_inline:
            raise ValueError("Use journeys or one inline journey.")
        if self.journeys is not None and not self.journeys:
            raise ValueError("Journeys must not be empty.")
        _validate_unique_target_ids(self.targets)
        return self


class YewoFileSpec(SettingsSpec):
    env: dict[ValueName, EnvironmentValue] = Field(default_factory=dict)
    secrets: tuple[ValueName, ...] | ValueName = ()
    workflows: dict[Identifier, WorkflowSpec] | None = None
    journeys: tuple[JourneySpec, ...] | None = None
    journey: str | None = Field(default=None, min_length=1)
    id: Identifier | None = None
    name: str | None = Field(default=None, min_length=1)
    security_rules: tuple[str, ...] = ()
    checks: tuple[UserCheck, ...] = ()
    coverage: tuple[CoverageSpec, ...] | None = None
    actors: dict[ValueName, dict[ValueName, EnvironmentValue]] | None = None
    policy: PolicySpec | None = None
    preconditions: ScenarioPreconditionsSpec | None = None
    targets: tuple[TargetSpec, ...] | None = None

    @model_validator(mode="after")
    def _validate_structure(self) -> Self:
        has_workflows = self.workflows is not None
        has_journeys = self.journeys is not None
        has_inline = self.journey is not None
        if sum((has_workflows, has_journeys, has_inline)) > 1:
            raise ValueError("Use workflows, journeys, or one inline journey.")
        if self.workflows is not None and not self.workflows:
            raise ValueError("Workflows must not be empty.")
        if self.journeys is not None and not self.journeys:
            raise ValueError("Journeys must not be empty.")
        _validate_unique_target_ids(self.targets)
        return self


def _validate_unique_target_ids(targets: tuple[TargetSpec, ...] | None) -> None:
    if targets is None:
        return
    target_ids = [target.id for target in targets]
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("Target identifiers must be unique.")


class ResolvedValue(StrictRecord):
    name: ValueName
    value: str = Field(repr=False)
    secret: bool = False
    refused: bool = False


class CoverageCell(StrictRecord):
    category: CoverageCategory
    scenario_type: ScenarioType
    priority: CoveragePriority
    index: int = Field(ge=0)

    @property
    def cell_id(self) -> str:
        return f"{self.category.value}.{self.scenario_type.value}.{self.index}"


class ResolvedSettings(StrictRecord):
    platform: Literal["android", "ios"] | None = None
    device_name: str | None = None
    headless: bool | None = None
    app_path: Path | None = None
    model: str | None = None
    models: ModelOverridesSpec | None = None
    grounder: GrounderSelectionSpec | None = None
    max_steps: int | None = Field(default=None, ge=1, le=500)
    max_duration_seconds: float | None = Field(default=None, gt=0, le=14_400)
    settle_time: float | None = Field(default=None, ge=0, le=5)
    output_dir: Path | None = None
    memory_root: Path | None = None
    record: bool | None = None
    durable_replay: bool | None = None


class ResolvedJourney(StrictRecord):
    journey_id: Identifier
    name: str
    objective: str = Field(min_length=1)
    security_rules: tuple[str, ...] = ()
    checks: tuple[UserCheck, ...] = ()
    declared_values: dict[ValueName, ResolvedValue] = Field(default_factory=dict)
    actors: dict[ValueName, dict[ValueName, ResolvedValue]] = Field(default_factory=dict)
    coverage: tuple[CoverageCell, ...]
    preconditions: ScenarioPreconditionsSpec


class ResolvedTarget(StrictRecord):
    target_id: Identifier
    settings: ResolvedSettings


class ResolvedWorkflow(StrictRecord):
    name: str
    settings: ResolvedSettings
    journeys: tuple[ResolvedJourney, ...]
    policy_waivers: dict[str, str] = Field(default_factory=dict)
    targets: tuple[ResolvedTarget, ...] = ()


class LoadedYewoProject(StrictRecord):
    source_path: Path
    settings: ResolvedSettings
    workflows: tuple[ResolvedWorkflow, ...]
