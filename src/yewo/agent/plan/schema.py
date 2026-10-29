"""The shape of one planning answer, apart from the prompt that asks for it."""

from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field

from yewo.agent.actuation.actions import DescribeAction, MarksAction
from yewo.agent.actuation.coordinate_actions import CoordinateAction


class DoneAction(BaseModel):
    """Signals that the goal is complete or that execution cannot continue."""

    type: Literal["done"] = "done"
    reason: str = Field(..., description="Why the goal is complete, or why execution must stop.")


DeviceAction = Annotated[CoordinateAction | DoneAction, Field(discriminator="type")]
MarksDeviceAction = Annotated[MarksAction | DoneAction, Field(discriminator="type")]
DescribeDeviceAction = Annotated[DescribeAction | DoneAction, Field(discriminator="type")]


class DeviceStepResponse(BaseModel):
    """Single planning step response from the device agent."""

    current_screen: str = Field(
        ...,
        description="Descriptive name of the current screen or page, based on visible content.",
    )
    evaluation_previous_goal: str = Field(
        default="",
        description=(
            "Brief honest assessment of whether the previous action succeeded, failed, or produced "
            "an uncertain result. Empty string on the very first step. "
            "Examples: 'Success — the requested screen opened', "
            "'Failed — button did not respond', "
            "'Uncertain — field unchanged but masked content cannot be confirmed visually'."
        ),
    )
    observation: str = Field(
        ...,
        description="Description of what is currently visible on the screen.",
    )
    reasoning: str = Field(
        ...,
        description="Explain why the chosen action is the best next step.",
    )
    goal_satisfied: bool = Field(
        default=False,
        description=(
            "True when the goal's stated outcome is observable on the current screen, before "
            "any further action. Judge only the stated outcome. Follow-on UI does not undo "
            "evidence that remains visible."
        ),
    )
    goal_evidence: str = Field(
        default="",
        description=(
            "What on the current screen shows the goal is met. Name the stated outcome you "
            "can see. A changed screen and a pressed control are not the outcome. Empty "
            "unless goal_satisfied."
        ),
    )
    action: DeviceAction = Field(
        ...,
        description="The single action to execute next. Use done() when the goal is complete.",
    )
    external_navigation: bool = Field(
        default=False,
        description=(
            "Set True if this action is expected to temporarily navigate outside the SUT app "
            "(for example, opening a browser link, document picker, or system dialog)."
        ),
    )
    active_instruction_index: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "Diagnostic estimate of the active scenario instruction's one-based "
            "position. None outside scenario execution. The instruction controller "
            "does not accept this field as progress evidence."
        ),
    )


class MarksStepResponse(DeviceStepResponse):
    """The step response used in marks mode, where targets have no coordinates."""

    action: MarksDeviceAction = Field(
        ..., description="The single next action, naming its target by number."
    )


class DescribeStepResponse(DeviceStepResponse):
    """The step response used in describe mode, where targets are named in words."""

    action: DescribeDeviceAction = Field(
        ..., description="The single next action, naming its target in words."
    )
