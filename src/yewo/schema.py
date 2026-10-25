"""The shape of a yewo.yaml.

One file holds the settings and the tests. A typo can mean either a setting
that does nothing or a test that never runs. Every key is declared and every
value is constrained. Anything else stops the run and names the key.

These models describe the file. `config.py` resolves the settings from them
and `suite.py` turns the rest into workflows, so one declaration answers for
both. The JSON Schema editors read is
generated from these models by `tools/emit_schema.py`, so the two cannot
disagree.
"""

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from yewo.testing.taxonomy import Category, Priority, ScenarioType
from yewo.testing.lifecycle import ScenarioPreconditions

PLATFORMS = ("android", "ios")
GROUNDINGS = ("marks", "describe")

Platform = Literal["android", "ios"]
Grounding = Literal["marks", "describe"]


class StrictModel(BaseModel):
    """A mapping that accepts only the keys it declares."""

    model_config = ConfigDict(extra="forbid")


class DeviceSpec(StrictModel):
    """The target a run drives."""

    name: Optional[str] = Field(default=None, description="Emulator, simulator or device name.")
    platform: Optional[Platform] = Field(default=None, description="Which platform to run against.")
    headless: Optional[bool] = Field(default=None, description="Run the emulator without a window.")


class TargetSpec(StrictModel):
    """One explicitly named member of a sequential target matrix."""

    id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
        description="Stable target name used in reports and artifact paths.",
    )
    platform: Platform
    device: str = Field(min_length=1, description="Emulator, simulator, or device name.")
    app: Optional[str] = Field(
        default=None,
        description="Target-specific app binary. Falls back to the workflow app.",
    )
    app_path: Optional[str] = Field(
        default=None, description="Another spelling of the target-specific app."
    )
    headless: Optional[bool] = Field(
        default=None, description="Run this target's emulator without a window."
    )

    @field_validator("device")
    @classmethod
    def _validate_device(cls, value: str) -> str:
        device = value.strip()
        if not device:
            raise ValueError("device must not be blank")
        return device


def _validate_target_ids(targets: Optional[list[TargetSpec]]) -> Optional[list[TargetSpec]]:
    """Refuse two matrix entries that would own one report path."""
    if targets is None:
        return None
    target_ids = [target.id for target in targets]
    duplicates = sorted({target_id for target_id in target_ids if target_ids.count(target_id) > 1})
    if duplicates:
        raise ValueError(f"target IDs must be unique: {', '.join(duplicates)}")
    return targets


class DeclaredValue(StrictModel):
    """A declared value, whether it is sensitive, and what the app owes it.

    A value marked secret is replaced by its name in everything Yewo writes
    down. Nothing else is masked, so a credential written straight into the
    journey text is not hidden.
    """

    value: str
    secret: bool = False
    redaction: Literal["hide", "last4", "show"] = Field(
        default="hide",
        description="How much of this value an external model may receive.",
    )
    verify_as: Literal["auto", "exact", "normalized", "masked"] = Field(
        default="auto",
        description=(
            "How local post-action checks may recognize this value. Auto accepts "
            "an exact value or an observed masked representation."
        ),
    )
    normalizers: list[
        Literal["whitespace", "hyphens", "case", "currency", "grouping"]
    ] = Field(
        default_factory=list,
        description="Transformations permitted before local value comparison.",
    )
    visible_suffix: int = Field(
        default=0,
        ge=0,
        description="Characters that may remain visible at the end of a masked value.",
    )
    masked_length: Optional[int] = Field(
        default=None,
        ge=1,
        description="Required displayed length for a masked value. Defaults to input length.",
    )
    refused: bool = Field(
        default=False,
        description=(
            "True where this app refuses the value, which is a fact about the "
            "app that no reading of the value could find. A scenario about a "
            "refusal names one of these, and the declaration answers for it."
        ),
    )


EnvValue = Union[DeclaredValue, str, int, float, bool]


class Settings(StrictModel):
    """How a run behaves. Every one of these may be set per file or per workflow."""

    device: Optional[Union[DeviceSpec, str]] = Field(
        default=None, description="The target, as a name or a block."
    )
    emulator: Optional[str] = Field(default=None, description="Another spelling of 'device'.")
    platform: Optional[Platform] = None
    headless: Optional[bool] = None
    app: Optional[str] = Field(default=None, description="App binary, relative to this file.")
    app_path: Optional[str] = Field(default=None, description="Another spelling of 'app'.")
    model: Optional[str] = None
    fast_model: Optional[str] = Field(
        default=None, description="Cheaper model for per-step planning."
    )
    grounding: Optional[Grounding] = Field(
        default=None,
        description=(
            "How the planner names a target. Marks selects a detected element. "
            "describe places a target from its visible description."
        ),
    )
    max_steps: Optional[int] = Field(default=None, ge=1)
    settle_time: Optional[float] = Field(default=None, ge=0)
    output_dir: Optional[str] = None
    record: Optional[bool] = Field(default=None, description="Record the screen during the run.")
    record_session: Optional[bool] = Field(
        default=None, description="Another spelling of 'record'."
    )
    durable_replay: Optional[bool] = Field(
        default=None,
        description=(
            "Re-walk a path a passing run saved, instead of planning it "
            "again. Unlike 'replay', which reuses a path inside one run, "
            "this one outlives the run. On by default. Each run audits its "
            "own screens. Drift returns only the affected action to the planner."
        ),
    )
    allow_secret_exposure: Optional[bool] = Field(
        default=None,
        description=(
            "Allow declared secrets in model input and screen recordings. "
            "Off by default. Stored reports and artifacts stay protected."
        ),
    )


class PolicySpec(StrictModel):
    """Static package-policy exceptions, keyed by stable check ID."""

    waive: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Policy check IDs accepted for this workflow, with the reason each "
            "exception is necessary."
        ),
    )

    @field_validator("waive")
    @classmethod
    def _validate_waivers(cls, values: dict[str, str]) -> dict[str, str]:
        cleaned = {}
        for raw_id, raw_reason in values.items():
            check_id = raw_id.strip()
            reason = raw_reason.strip()
            if not check_id or "." not in check_id:
                raise ValueError("waiver keys must be semantic check IDs, such as android.debuggable")
            if not reason:
                raise ValueError(f"waiver '{check_id}' must explain why it is accepted")
            cleaned[check_id] = reason
        return cleaned


class CoverageSpec(StrictModel):
    """One category, and which shapes of test to write for it.

    Eight categories and nine types give up to seventy-two pairs. One model
    call per pair is only affordable where a file says which pairs matter. A
    pair nothing declares is never generated, and the report names it as
    uncovered.
    """

    category: Category = Field(description="What these scenarios look for.")
    types: list[ScenarioType] = Field(
        min_length=1, description="Which shapes of test to write for this category."
    )
    priority: Priority = Field(
        default=Priority.P1,
        description="Which cells a run has to finish first, if it cannot finish all.",
    )


class JourneySpec(StrictModel):
    """One test: what a user does, and what must be true afterwards."""

    id: Optional[str] = None
    name: Optional[str] = None
    journey: Optional[str] = Field(default=None, description="The journey, in prose.")
    description: Optional[str] = Field(default=None, description="Another spelling of 'journey'.")
    security_rules: list[str] = Field(
        default_factory=list, description="Rules every scenario must respect."
    )
    env: dict[str, EnvValue] = Field(
        default_factory=dict, description="Values for this test, over the workflow's."
    )
    preconditions: Optional[ScenarioPreconditions] = Field(
        default=None,
        description="Optional app and device state applied before visible UI execution.",
    )


class WorkflowSpec(Settings):
    """One environment, its target or targets, and the tests to run there."""

    env: dict[str, EnvValue] = Field(default_factory=dict)
    secrets: Union[list[str], str] = Field(
        default_factory=list, description="Names in 'env' to treat as sensitive."
    )
    journeys: Optional[list[JourneySpec]] = None
    journey: Optional[str] = None
    id: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    security_rules: list[str] = Field(default_factory=list)
    coverage: Optional[list[CoverageSpec]] = None
    actors: Optional[dict[str, dict[str, EnvValue]]] = Field(
        default=None,
        description=(
            "Named personas or role contexts for role scenarios. Each holds "
            "app-specific values under names you choose."
        ),
    )
    policy: Optional[PolicySpec] = None
    preconditions: Optional[ScenarioPreconditions] = None
    targets: Optional[list[TargetSpec]] = Field(
        default=None,
        min_length=1,
        description="Sequential targets that each receive the same locked journeys.",
    )

    _unique_target_ids = field_validator("targets")(_validate_target_ids)


class YewoFile(Settings):
    """A yewo.yaml: settings for every workflow, and the workflows."""

    model_config = ConfigDict(extra="forbid", title="Yewo configuration")

    env: dict[str, EnvValue] = Field(default_factory=dict)
    secrets: Union[list[str], str] = Field(default_factory=list)
    workflows: Optional[dict[str, WorkflowSpec]] = Field(
        default=None, description="Named workflows. Each owns an app and its tests."
    )
    # A file written before workflows existed is one unnamed workflow.
    journeys: Optional[list[JourneySpec]] = None
    journey: Optional[str] = None
    id: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    security_rules: list[str] = Field(default_factory=list)
    coverage: Optional[list[CoverageSpec]] = None
    actors: Optional[dict[str, dict[str, EnvValue]]] = Field(
        default=None,
        description=(
            "Named personas or role contexts for role scenarios. Each holds "
            "app-specific values under names you choose."
        ),
    )
    policy: Optional[PolicySpec] = None
    preconditions: Optional[ScenarioPreconditions] = None
    targets: Optional[list[TargetSpec]] = Field(
        default=None,
        min_length=1,
        description="Sequential targets inherited by each workflow unless overridden.",
    )

    _unique_target_ids = field_validator("targets")(_validate_target_ids)


def _format_error_location(location: tuple[Any, ...]) -> str:
    """Names the place in the file a problem is, the way the file reads."""
    return ".".join(str(part) for part in location) or "the file"


def format_validation_errors(error: Any) -> list[str]:
    """Turns a validation failure into lines that name the key and the fix."""
    lines = []
    for detail in error.errors():
        location = _format_error_location(detail["loc"])
        kind = detail.get("type", "")
        if kind == "extra_forbidden":
            lines.append(f"{location}: unknown key")
        elif kind == "literal_error":
            allowed = detail.get("ctx", {}).get("expected", "")
            lines.append(f"{location}: {detail.get('input')!r} is not one of {allowed}")
        else:
            lines.append(f"{location}: {detail.get('msg', 'is not valid')}")
    return lines
