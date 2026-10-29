"""Planner request for a target intentionally absent from the current target list."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class TargetNotListedAction(BaseModel):
    """Report an app target absent from the current target list."""

    type: Literal["target_not_listed"] = "target_not_listed"
    description: str = Field(..., description="What the target looks like and roughly where it is.")
    region: Literal[
        "upper_left",
        "upper_center",
        "upper_right",
        "middle_left",
        "middle_center",
        "middle_right",
        "lower_left",
        "lower_center",
        "lower_right",
    ] = Field(..., description="The coarse screen region to inspect more closely.")
    intent: Literal["enter_text", "activate"] = Field(
        "activate",
        description="Whether you need to put text into this target or press it.",
    )
    anchor_above: Optional[int] = Field(
        None,
        description="Number of the visible control immediately above the missing target.",
    )
    anchor_below: Optional[int] = Field(
        None,
        description="Number of the visible control immediately below the missing target.",
    )
    """What the target is for, which decides how it may be searched for.

    A field can be probed: a tap that only moves the cursor can be read back
    and undone by tapping elsewhere. Pressing an unknown control cannot be
    undone, so a target to activate is never probed blind. Defaults to the
    answer that forbids probing, so an older plan without this field loses
    nothing but the search.
    """

    reason: str = Field(..., description="What you were trying to achieve.")
