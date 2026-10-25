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

from pydantic import BaseModel, ConfigDict, Field

from yewo.testing.taxonomy import Category, Priority, ScenarioType

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


class WorkflowSpec(Settings):
    """One app, one device, one environment, and the tests to run against them."""

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
