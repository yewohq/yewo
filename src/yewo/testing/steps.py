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
        description="One action on one control, such as 'Tap the Login button.'"
    )

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
    Union[EnterStep, ActionStep, CapabilityStep], Field(discriminator="kind")
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
