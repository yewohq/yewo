"""Planner request for a target intentionally absent from the current target list."""

from typing import Literal

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
    reason: str = Field(..., description="What you were trying to achieve.")
