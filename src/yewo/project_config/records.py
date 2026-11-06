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
    app_path: str | None = Field(default=None, min_length=1)
    headless: bool | None = None

    @model_validator(mode="after")
    def _validate_app_alias(self) -> Self:
        if self.app is not None and self.app_path is not None:
            raise ValueError("Use app or app_path, not both.")
        return self


class ValueRedaction(StrEnum):
    HIDE = "hide"
    LAST4 = "last4"
    SHOW = "show"


class ValueVerificationMode(StrEnum):
    AUTO = "auto"
    EXACT = "exact"
    NORMALIZED = "normalized"
    MASKED = "masked"


class ValueNormalizer(StrEnum):
    WHITESPACE = "whitespace"
    HYPHENS = "hyphens"
    CASE = "case"
    CURRENCY = "currency"
    GROUPING = "grouping"


ScalarValue = str | int | float | bool


class DeclaredValueSpec(StrictRecord):
    value: ScalarValue = Field(repr=False)
    secret: bool = False
    refused: bool = False
    redaction: ValueRedaction = ValueRedaction.HIDE
    verify_as: ValueVerificationMode = ValueVerificationMode.AUTO
    normalizers: tuple[ValueNormalizer, ...] = ()
    visible_suffix: int = Field(default=0, ge=0)
    masked_length: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _validate_verification_options(self) -> Self:
        if self.verify_as is ValueVerificationMode.EXACT and (
            self.normalizers or self.visible_suffix or self.masked_length is not None
        ):
            raise ValueError("Exact verification does not accept comparison options.")
        if self.verify_as is ValueVerificationMode.NORMALIZED and not self.normalizers:
            raise ValueError("Normalized verification requires at least one normalizer.")
        if self.verify_as not in {ValueVerificationMode.AUTO, ValueVerificationMode.MASKED} and (
            self.visible_suffix or self.masked_length is not None
        ):
            raise ValueError("Masking options require auto or masked verification.")
        if not self.secret and self.redaction is not ValueRedaction.HIDE:
            raise ValueError("Redaction options require a secret value.")
        return self


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
    FUNCTIONAL = "functional"
    INPUT = "input"
    SECURITY = "security"
    ACCESSIBILITY = "accessibility"
    PERFORMANCE = "performance"
    NETWORK = "network"
    LIFECYCLE = "lifecycle"
    PERMISSION = "permission"


class ScenarioType(StrEnum):
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


class GrounderSelectionSpec(StrictRecord):
    model: str = Field(min_length=1)
    protocol: GroundingProtocol


class ModelOverridesSpec(StrictRecord):
    planner: str | None = Field(default=None, min_length=1)
    auditor: str | None = Field(default=None, min_length=1)
    recovery: str | None = Field(default=None, min_length=1)
    action_verifier: str | None = Field(default=None, min_length=1)
    screen_analyzer: str | None = Field(default=None, min_length=1)
    replay_preparer: str | None = Field(default=None, min_length=1)


class SettingsSpec(StrictRecord):
    device: DeviceSpec | str | None = None
    emulator: str | None = Field(default=None, min_length=1)
    platform: Literal["android", "ios"] | None = None
    headless: bool | None = None
    app: str | None = Field(default=None, min_length=1)
    app_path: str | None = Field(default=None, min_length=1)
    model: str | None = Field(default=None, min_length=1)
    models: ModelOverridesSpec | None = None
    grounder: GrounderSelectionSpec | None = None
    max_steps: int | None = Field(default=None, ge=1, le=500)
    settle_time: float | None = Field(default=None, ge=0, le=5)
    output_dir: str | None = Field(default=None, min_length=1)
    memory_root: str | None = Field(default=None, min_length=1)
    record: bool | None = None
    record_session: bool | None = None
    durable_replay: bool | None = None
    allow_secret_exposure: bool | None = None

    @model_validator(mode="after")
    def _validate_aliases(self) -> Self:
        for primary, alias in (
            ("device", "emulator"),
            ("app", "app_path"),
            ("record", "record_session"),
        ):
            if getattr(self, primary) is not None and getattr(self, alias) is not None:
                raise ValueError(f"Use {primary} or {alias}, not both.")
        return self


class JourneySpec(StrictRecord):
    id: Identifier | None = None
    name: str | None = Field(default=None, min_length=1)
    journey: str | None = Field(default=None, min_length=1)
    description: str | None = Field(default=None, min_length=1)
    security_rules: tuple[str, ...] = ()
    env: dict[ValueName, EnvironmentValue] = Field(default_factory=dict)
    preconditions: ScenarioPreconditionsSpec | None = None

    @model_validator(mode="after")
    def _validate_description_alias(self) -> Self:
        if (self.journey is None) == (self.description is None):
            raise ValueError("Use exactly one of journey or description.")
        return self


class WorkflowSpec(SettingsSpec):
    env: dict[ValueName, EnvironmentValue] = Field(default_factory=dict)
    secrets: tuple[ValueName, ...] | ValueName = ()
    journeys: tuple[JourneySpec, ...] | None = None
    journey: str | None = Field(default=None, min_length=1)
    id: Identifier | None = None
    name: str | None = Field(default=None, min_length=1)
    description: str | None = Field(default=None, min_length=1)
    security_rules: tuple[str, ...] = ()
    coverage: tuple[CoverageSpec, ...] | None = None
    actors: dict[ValueName, dict[ValueName, EnvironmentValue]] | None = None
    policy: PolicySpec | None = None
    preconditions: ScenarioPreconditionsSpec | None = None
    targets: tuple[TargetSpec, ...] | None = None

    @model_validator(mode="after")
    def _validate_journeys(self) -> Self:
        if self.journey is not None and self.description is not None:
            raise ValueError("Use exactly one of journey or description.")
        has_list = self.journeys is not None
        has_inline = self.journey is not None or self.description is not None
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
    description: str | None = Field(default=None, min_length=1)
    security_rules: tuple[str, ...] = ()
    coverage: tuple[CoverageSpec, ...] | None = None
    actors: dict[ValueName, dict[ValueName, EnvironmentValue]] | None = None
    policy: PolicySpec | None = None
    preconditions: ScenarioPreconditionsSpec | None = None
    targets: tuple[TargetSpec, ...] | None = None

    @model_validator(mode="after")
    def _validate_structure(self) -> Self:
        if self.journey is not None and self.description is not None:
            raise ValueError("Use exactly one of journey or description.")
        has_workflows = self.workflows is not None
        has_journeys = self.journeys is not None
        has_inline = self.journey is not None or self.description is not None
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
    redaction: ValueRedaction = ValueRedaction.HIDE
    verify_as: ValueVerificationMode = ValueVerificationMode.AUTO
    normalizers: tuple[ValueNormalizer, ...] = ()
    visible_suffix: int = 0
    masked_length: int | None = None


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
    settle_time: float | None = Field(default=None, ge=0, le=5)
    output_dir: Path | None = None
    memory_root: Path | None = None
    record: bool | None = None
    durable_replay: bool | None = None
    allow_secret_exposure: bool | None = None


class ResolvedJourney(StrictRecord):
    journey_id: Identifier
    name: str
    objective: str = Field(min_length=1)
    security_rules: tuple[str, ...] = ()
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
