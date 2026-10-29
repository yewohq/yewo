"""What a scenario does, as a typed list rather than a list of sentences.

A step says what it is, so nothing has to be read out of a sentence.

Two things follow. A value entered and the sentence the planner reads come
from one object, so they cannot name different values. The outcome is written
from that same object. A device state carries the typed capability, so what a
scenario needs is read off its own steps. Declared beside them, the two could
disagree.

Only an action on a control stays prose. The planner finds that control on
whatever screen the app shows, so a fixed target would fail the first time a
layout moved. A prose step is held to opening with an action and to carrying
no claim, which is as much as an opening word can decide.
"""

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

from yewo.agent.actuation.actions import Capability, CapabilityKind
from yewo.testing.step_language import ACTION_VERBS, leaked_assertion, unknown_action
from yewo.testing.taxonomy import ScenarioType


class EnterStep(BaseModel):
    """A value entered into a field, named rather than written out.

    The value is always a name the file declared. A scenario that wrote one
    out would be guessing what this app does with it. The declaration is the
    only place that answer exists.
    """

    kind: Literal["enter"] = "enter"
    field: str = Field(
        description="The visible label of the field, as a user reads it."
    )
    value: str = Field(
        description="The name of a declared value, without braces, such as 'email'."
    )

    def rendered(self) -> str:
        """The sentence the planner reads, written here and not by a model."""
        return f"Enter '{{{{{self.value}}}}}' in the {self.field} field."


class ActionStep(BaseModel):
    """One action on a control, as prose the planner adapts to the screen.

    Prose on purpose. The planner finds the control on whatever screen the app
    actually shows, and a fixed target would fail the first time a layout
    moved. So this one is held to opening with an action and to carrying no
    claim, which is all a sentence can be held to.
    """

    kind: Literal["action"] = "action"
    instruction: str = Field(
        description="One action on one control, such as 'Tap the Save button.'"
    )
    evidence: Optional[str] = Field(
        default=None,
        description=(
            "The same instruction copied verbatim from the journey. Omit only for "
            "the synthetic app-launch step."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_action(cls, data):
        if isinstance(data, dict):
            val = dict(data)
            if "type" in val and "kind" not in val:
                val["kind"] = val.pop("type")
            if "action" in val and "instruction" not in val:
                action_val = val.pop("action")
                if isinstance(action_val, dict):
                    action_val = action_val.get("instruction") or action_val.get("action")
                target = val.pop("target", None)
                if isinstance(action_val, str) and target and target not in action_val:
                    val["instruction"] = f"{action_val} {target}"
                elif isinstance(action_val, str):
                    val["instruction"] = action_val
            if "instruction" in val and isinstance(val["instruction"], dict):
                inst_dict = val["instruction"]
                nested_instruction = inst_dict.get("instruction") or inst_dict.get("action")
                if isinstance(nested_instruction, str):
                    val["instruction"] = nested_instruction
            return val
        return data

    @model_validator(mode="after")
    def _validate_instruction_action(self) -> "ActionStep":
        """Raises:
            ValueError: If the instruction names no action, or makes a claim.
        """
        unknown = unknown_action(self.instruction)
        if unknown is not None:
            raise ValueError(
                f"{self.instruction!r} opens with {unknown!r}, which is not an "
                f"action. Open it with one of: {', '.join(sorted(ACTION_VERBS))}."
            )
        leaked = leaked_assertion(self.instruction)
        if leaked:
            raise ValueError(
                f"{self.instruction!r} goes on to '{leaked}'. A step carries "
                f"one action. Put what must be true in expected_outcomes."
            )
        return self

    def rendered(self) -> str:
        return self.instruction


class IntentStep(BaseModel):
    """A user objective whose mechanics must come from the observed application."""

    kind: Literal["intent"] = "intent"
    instruction: str = Field(
        min_length=1,
        description=(
            "One outcome-oriented user objective copied from the journey."
        ),
    )
    evidence: str = Field(
        min_length=1,
        description="The same objective copied verbatim from the journey.",
    )

    def rendered(self) -> str:
        return self.instruction


VARIATION_TYPES = frozenset(
    {
        ScenarioType.NEGATIVE,
        ScenarioType.BOUNDARY,
        ScenarioType.EMPTY,
        ScenarioType.EXPLORATORY,
    }
)


class VariationStep(BaseModel):
    """A different user strategy for pursuing the same complete journey.

    The strategy may choose a probe. It does not declare how the product must
    handle that probe. The runtime planner grounds the mechanics against the
    observed application, while the unchanged goal prevents a coverage cell
    from becoming an isolated control check.
    """

    kind: Literal["variation"] = "variation"
    goal: str = Field(
        min_length=1,
        description="The complete journey copied verbatim from the app context.",
    )
    objective: str = Field(
        min_length=1,
        description="The journey objective to vary, copied verbatim from the context.",
    )
    evidence: str = Field(
        min_length=1,
        description="The same objective copied verbatim from the app context.",
    )
    strategy: ScenarioType = Field(
        description="The requested negative, boundary, empty, or exploratory strategy."
    )
    probe_value: Optional[str] = Field(
        default=None,
        description=(
            "A generated test input used as a probe, not a claimed product limit or "
            "expected result."
        ),
    )
    repetitions: Optional[int] = Field(
        default=None,
        ge=2,
        le=1000,
        description="A bounded number of attempts used to probe repeated or high-volume use.",
    )

    @model_validator(mode="after")
    def _validate_strategy_parameters(self) -> "VariationStep":
        if self.strategy not in VARIATION_TYPES:
            raise ValueError(
                "A variation strategy must be negative, boundary, empty, or exploratory."
            )
        if (
            self.strategy is ScenarioType.BOUNDARY
            and self.probe_value is None
            and self.repetitions is None
        ):
            raise ValueError("A boundary variation needs a probe_value or repetitions.")
        if self.strategy in {ScenarioType.EMPTY, ScenarioType.EXPLORATORY} and (
            self.probe_value is not None or self.repetitions is not None
        ):
            raise ValueError(
                f"A {self.strategy.value} variation does not accept probe parameters."
            )
        return self

    def rendered(self) -> str:
        goal_instruction = f"Complete this journey: {self.goal}"
        if self.strategy is ScenarioType.NEGATIVE and self.probe_value is not None:
            variation = (
                f"use probe value {self.probe_value!r} while attempting this objective: "
                f"{self.objective}"
            )
        elif self.strategy is ScenarioType.NEGATIVE:
            variation = (
                "use the separately declared refused value while attempting this "
                f"objective: {self.objective}"
            )
        elif self.strategy is ScenarioType.BOUNDARY and self.repetitions is not None:
            variation = (
                f"attempt this objective exactly {self.repetitions} times: {self.objective}"
            )
        elif self.strategy is ScenarioType.BOUNDARY:
            variation = (
                f"use probe value {self.probe_value!r} while attempting this objective: "
                f"{self.objective}"
            )
        elif self.strategy is ScenarioType.EMPTY:
            variation = (
                "omit the relevant value or item while attempting this objective: "
                f"{self.objective}"
            )
        else:
            variation = (
                "use an alternate route or action order that the observed app supports "
                f"while attempting this objective: {self.objective}"
            )
        return f"{goal_instruction} While doing so, {variation}."


class CapabilityStep(BaseModel):
    """A device state the scenario needs, carrying the capability itself.

    Typed rather than described. A cell about the network is then held to
    having asked for the network. The preflight reads what a scenario needs
    off its own steps.
    """

    kind: Literal["capability"] = "capability"
    capability: Capability = Field(
        description="The device state to apply before the steps after this one."
    )

    def rendered(self) -> str:
        """The sentence the planner reads, from the capability's own fields.

        A hint rather than the mechanism. The planner asks for the capability
        it recognises, and the typed capability beside this says which one.
        """
        capability_name = self.capability.kind.value.replace("_", " ")
        params = {
            name: value
            for name, value in self.capability.model_dump(exclude={"kind"}).items()
            if value is not None
        }
        if not params:
            return f"Set the device {capability_name}."
        if len(params) == 1:
            return f"Set the device {capability_name} to {next(iter(params.values()))}."
        formatted_params = ", ".join(f"{name} {value}" for name, value in params.items())
        return f"Set the device {capability_name}: {formatted_params}."


Step = Annotated[
    Union[EnterStep, ActionStep, IntentStep, VariationStep, CapabilityStep],
    Field(discriminator="kind"),
]
"""One thing a scenario does. The kind says which, so nothing is parsed."""


def rendered(steps: list) -> list[str]:
    """Every step as the sentence the planner reads, in order."""
    return [step.rendered() for step in steps]


def capability_kinds_in(steps: list) -> list[CapabilityKind]:
    """Which device states these steps ask for, in the order they ask.

    Read off the steps rather than declared beside them. A scenario that
    declared one list and acted on another would have two answers.
    """
    found = []
    for step in steps:
        if isinstance(step, CapabilityStep) and step.capability.kind not in found:
            found.append(step.capability.kind)
    return found


def capabilities_in(steps: list) -> list[Capability]:
    """Every capability these steps ask for, in the order they ask.

    The capability survives rather than its kind, so a caller can read the
    variant. Support can differ between two variants of one kind, and a kind
    on its own cannot say which one a scenario wants. `capability_kinds_in`
    answers the narrower question.
    """
    return [step.capability for step in steps if isinstance(step, CapabilityStep)]


def entered_values(steps: list) -> list[EnterStep]:
    """Every value these steps enter, in order."""
    return [step for step in steps if isinstance(step, EnterStep)]


def variations_in(steps: list) -> list[VariationStep]:
    """Every user-strategy variation in scenario order."""
    return [step for step in steps if isinstance(step, VariationStep)]


APP_LAUNCH_INSTRUCTIONS = frozenset({"launch the app", "open the app"})


def _provenance_text(value: str) -> str:
    """Normalize presentation-only whitespace without translating text."""
    return " ".join(value.split()).casefold()


def unsupported_ui_claim(step: ActionStep, source: str) -> Optional[str]:
    """Return why an executable action is not a verbatim journey instruction.

    This deliberately does not classify words such as ``button`` or ``tap``.
    A vocabulary-based gate would be language-specific and would always miss
    some future control type. Verbatim provenance is deterministic in every
    language; everything else remains an intent for the runtime planner to
    ground against the observed application.
    """
    if _provenance_text(step.instruction).rstrip(".!?…") in APP_LAUNCH_INSTRUCTIONS:
        return None
    evidence = (step.evidence or "").strip()
    if not evidence:
        return "the executable action provides no verbatim source instruction"
    normalized_evidence = _provenance_text(evidence)
    if normalized_evidence not in _provenance_text(source):
        return "the action evidence is not an exact excerpt from the journey"
    if normalized_evidence != _provenance_text(step.instruction):
        return "the executable action is not copied verbatim from its evidence"
    return None


def unsupported_intent_claim(step: IntentStep, source: str) -> Optional[str]:
    """Return why an intent is not a verbatim journey objective."""
    normalized_evidence = _provenance_text(step.evidence)
    if normalized_evidence not in _provenance_text(source):
        return "the intent evidence is not an exact excerpt from the journey"
    if normalized_evidence != _provenance_text(step.instruction):
        return "the intent is not copied verbatim from its evidence"
    return None


def unsupported_variation_claim(step: VariationStep, source: str) -> Optional[str]:
    """Return why a varied journey or objective lacks verbatim provenance."""
    normalized_source = _provenance_text(source)
    if _provenance_text(step.goal) != normalized_source:
        return "the variation does not preserve the complete journey verbatim"
    normalized_evidence = _provenance_text(step.evidence)
    if normalized_evidence not in normalized_source:
        return "the variation evidence is not an exact excerpt from the journey"
    if _provenance_text(step.objective) != normalized_evidence:
        return "the varied objective is not copied verbatim from its evidence"
    return None


def refusal_outcome(field: str) -> str:
    """The outcome Yewo writes where a step enters a value the app refuses.

    Written from the step rather than by a model. The file said the app
    refuses the value, so the words carrying that must not be free prose
    beside it.
    """
    return (
        f"The app refuses the value entered in the {field} field, and the "
        f"screen shows that it was refused."
    )


def undeclared(steps: list, declared: frozenset) -> Optional[str]:
    """The first value a step enters that the file never declared.

    Raises nothing. The caller decides whether an unknown name is worth one
    more turn or costs the cell.
    """
    for step in entered_values(steps):
        if step.value not in declared:
            return step.value
    return None
