

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator

from yewo.testing import steps
from yewo.identity import derived_uuid, new_uuid
from yewo.testing.steps import Step
from yewo.testing.taxonomy import (
    DEFAULT_COVERAGE,
    Category,
    CoverageCell,
    Priority,
    ScenarioType,
)
from typing import Dict, FrozenSet, List, Optional

# Trail models live in agent/trail.py — re-exported here for backward compatibility.
from yewo.agent.actuation.actions import Capability, CapabilityKind
from yewo.metrics import Measurement, Metric  # noqa: F401
from yewo.policy import PolicyCheckReport, PolicyCheckStatus  # noqa: F401
from yewo.agent.run.postconditions import ValueVerificationPolicy
from yewo.agent.device.launch import SetupMode  # noqa: F401
from yewo.agent.memory.trail import SessionTrail, TrailElement, ScreenEntry  # noqa: F401
from yewo.testing.lifecycle import ScenarioPreconditions, SetupOrigin

# Re-exported so a caller reads one module. These live in evidence.py, which
# holds what a run produces, and the rules it answers.
from yewo.testing.evidence import (  # noqa: F401
    AssertionScope,
    BoundingBox,
    ExpectedOutcome,
    IssueKind,
    JourneyReport,
    JourneyStatus,
    QAIssue,
    REPORT_STATUS_ALIASES,
    ReceiptRecord,
    Rendering,
    ReportStatus,
    RuleVerification,
    RunEnvironment,
    STATUS_SPELLINGS,
    ScenarioReport,
    ScenarioStatus,
    SensitiveFinding,
    VerdictInputs,
)


class Actor(BaseModel):
    """A named persona or role context used by a scenario.

    A role scenario compares what different actors may see, reach, or do. The
    values inside are the app's, not Yewo's. They can establish any context
    the declared journey needs.
    """

    values: Dict[str, str] = Field(
        description="App-specific values that establish this actor's context, by name."
    )


class UserJourney(BaseModel):
    """Represents a high-level natural language user journey to test."""
    id: str
    name: str
    description: str
    security_rules: List[str] = Field(default_factory=list)
    secrets: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Name-to-value pairs the journey declared sensitive. Masked by name "
            "in artifacts and replaced with placeholders before model calls."
        ),
    )
    secret_displays: Dict[str, str] = Field(
        default_factory=dict,
        description="How much of each declared value an external model may receive.",
    )
    value_verifications: Dict[str, ValueVerificationPolicy] = Field(
        default_factory=dict,
        description="How each declared value may be represented for local verification.",
    )
    values: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Every name the journey's 'env' declared, resolved. These are the "
            "values that are meant to work. A scenario asks for one by name, "
            "rather than copying a string out of the description."
        ),
    )
    refused: FrozenSet[str] = Field(
        default_factory=frozenset,
        description=(
            "The names in 'values' the file said this app refuses. A fact "
            "about the app that no reading of the value could find."
        ),
    )
    actors: Dict[str, "Actor"] = Field(
        default_factory=dict,
        description="Named personas or role contexts available to role scenarios.",
    )
    coverage: List[CoverageCell] = Field(
        default_factory=lambda: list(DEFAULT_COVERAGE),
        description="The cells a run covers for this journey, in declared order.",
    )
    preconditions: ScenarioPreconditions = Field(default_factory=ScenarioPreconditions)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ScenarioDraft(BaseModel):
    """What a model writes for one cell: the walk, and what must hold after it.

    The identity is absent on purpose. Yewo knows which cell it asked
    about. A model asked to name the category, the type and the id again
    could only disagree with the question. `GeneratedTestCase` adds it.
    """

    scenario_name: str
    steps: List[Step] = Field(
        min_length=1,
        description=(
            "What the scenario does, in order. 'enter' names a declared value "
            "and the field it goes in. 'capability' carries a device state. "
            "'action' is one action on one control, in prose."
        ),
    )
    expected_outcomes: List[ExpectedOutcome] = Field(
        ...,
        min_length=1,
        description=(
            "What must be true for this scenario to pass. Give at least one. "
            "A separate component judges these, so a scenario that declares "
            "none cannot be judged at all. Yewo adds an outcome of its own "
            "for every value the file says this app refuses."
        ),
    )
    @model_validator(mode="after")
    def _validate_entered_values_declared(self, info: ValidationInfo) -> "ScenarioDraft":
        """Holds an entered value to a name the file declared.

        The names reach this through the validation context. An unknown one
        is then refused on the same path as any other mismatch, and the model
        is asked again. Read without the context, this passes: a caller that
        knows no names cannot judge one.

        Raises:
            ValueError: If a step enters a name the file did not declare.
        """
        known = (info.context or {}).get("declared")
        if known is None:
            return self
        unknown = steps.undeclared(self.steps, known)
        if unknown:
            raise ValueError(
                f"No value named {unknown!r} is declared. Use one of: "
                f"{', '.join(sorted(known)) or '(none declared)'}."
            )
        return self

    @model_validator(mode="after")
    def _validate_refused_value_present(
        self, info: ValidationInfo
    ) -> "ScenarioDraft":
        """Holds a cell about a refusal to entering one the file marked refused.

        Raises:
            ValueError: If this cell needs a refused value and no step enters one.
        """
        refused = (info.context or {}).get("refused")
        if not refused:
            return self
        if not any(
            step.value in refused for step in steps.entered_values(self.steps)
        ):
            raise ValueError(
                f"This cell enters a value this app refuses. One 'enter' step "
                f"has to name one of: {', '.join(sorted(refused))}."
            )
        return self

    @model_validator(mode="after")
    def _validate_capability_cell_steps(
        self, info: ValidationInfo
    ) -> "ScenarioDraft":
        """Holds a cell about a device state to having asked for one.

        Raises:
            ValueError: If this cell needs a capability and no step carries it.
        """
        required_capability = (info.context or {}).get("capability")
        if required_capability is None:
            return self
        if required_capability not in steps.capability_kinds_in(self.steps):
            raise ValueError(
                f"This cell is about the device being {required_capability.value}. One "
                f"'capability' step has to carry a {required_capability.value} capability."
            )
        return self


class GeneratedTestCase(ScenarioDraft):
    """One scenario, as a run holds it: a draft plus the cell it answers."""

    scenario_uuid: str = Field(
        default_factory=new_uuid,
        description=(
            "What this scenario is called wherever it is recorded. Drawn once, "
            "then kept by the lock, so two runs of one test agree on which "
            "test it was. `test_id` and `cell` are labels for a reader."
        ),
    )
    test_id: str
    cell: str
    """The key this scenario keeps across runs, such as `security.negative.0`."""

    category: Category
    scenario_type: ScenarioType
    priority: Priority
    setup: SetupMode = SetupMode.CLEAR_DATA
    setup_origin: SetupOrigin = SetupOrigin.DEFAULT
    initial_capabilities: List[Capability] = Field(default_factory=list)
    steps_instructions: List[str] = Field(
        default_factory=list,
        description=(
            "The steps as the sentences the planner reads, rendered by Yewo. "
            "Written from the steps so neither can drift from the other."
        ),
    )
    required_capabilities: List[CapabilityKind] = Field(
        default_factory=list,
        description=(
            "The device states this scenario asks for, read off its own steps. "
            "A scenario that declared these beside its steps had two answers."
        ),
    )
    required_metrics: List[Metric] = Field(
        default_factory=list,
        description=(
            "The measurements this scenario requires, derived from its measured "
            "outcomes so the list cannot disagree with them."
        ),
    )

    @model_validator(mode="after")
    def _derive_instructions_and_capabilities(self) -> "GeneratedTestCase":
        """Fills the rendered sentences and the capabilities from the steps.

        Derived rather than passed in. A scenario built any other way then
        cannot end up with an empty walk. Nor with a capability list that
        disagrees with its own steps.
        """
        if not self.steps_instructions:
            self.steps_instructions = steps.rendered(self.steps)
        if not self.required_capabilities:
            kinds = [capability.kind for capability in self.initial_capabilities]
            kinds.extend(steps.capability_kinds_in(self.steps))
            self.required_capabilities = list(dict.fromkeys(kinds))
        measured_metrics = [
            outcome.metric
            for outcome in self.expected_outcomes
            if outcome.metric is not None
        ]
        if not self.required_metrics:
            self.required_metrics = list(dict.fromkeys(measured_metrics))
        elif self.required_metrics != list(dict.fromkeys(measured_metrics)):
            raise ValueError("required_metrics must match the measured outcomes.")
        return self

    def outcome_uuid(self, index: int) -> str:
        """What the outcome at this position is called.

        Derived from the scenario and the position, which is what the outcome
        is. The lock pins the list, so the position holds across runs of one
        locked test.
        """
        return derived_uuid(self.scenario_uuid, index)

    @classmethod
    def of(
        cls,
        cell: CoverageCell,
        draft: ScenarioDraft,
        scenario_uuid: Optional[str] = None,
        setup: SetupMode = SetupMode.CLEAR_DATA,
        setup_origin: SetupOrigin = SetupOrigin.DEFAULT,
        initial_capabilities: Optional[List[Capability]] = None,
    ) -> "GeneratedTestCase":
        """Stamps a draft with the identity of the cell that asked for it.

        The identifier is passed in where a lock already holds one, so a
        locked scenario keeps the name it was reviewed under.
        """
        return cls(
            **({"scenario_uuid": scenario_uuid} if scenario_uuid else {}),
            test_id=cell.test_id,
            cell=cell.key,
            category=cell.category,
            scenario_type=cell.scenario_type,
            priority=cell.priority,
            setup=setup,
            setup_origin=setup_origin,
            initial_capabilities=list(initial_capabilities or []),
            **draft.model_dump(),
        )
