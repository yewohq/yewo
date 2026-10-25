"""Planner actions that pair an exact point with a semantic target."""

from typing import Annotated

from pydantic import BaseModel, Field, create_model

from yewo.agent.actuation.actions import (
    CapabilityAction,
    ClearAction,
    DoubleTapAction,
    DragAndDropAction,
    KeyEventAction,
    LongPressAction,
    SwipeAction,
    TapAction,
    TargetNotListedAction,
    TypeAction,
    WaitAction,
)


def _build_coordinate_action(model: type[BaseModel], name: str) -> type[BaseModel]:
    """Require the target description at the planner boundary."""
    return create_model(
        name,
        __base__=model,
        target=(
            str,
            Field(
                ...,
                description=(
                    "What the coordinate points to, described by its own visible "
                    "text, symbol, shape, or colour."
                ),
            ),
        ),
    )


TypeCoordinateAction = _build_coordinate_action(TypeAction, "TypeCoordinateAction")
TapCoordinateAction = _build_coordinate_action(TapAction, "TapCoordinateAction")
ClearCoordinateAction = _build_coordinate_action(ClearAction, "ClearCoordinateAction")
DoubleTapCoordinateAction = _build_coordinate_action(DoubleTapAction, "DoubleTapCoordinateAction")
LongPressCoordinateAction = _build_coordinate_action(LongPressAction, "LongPressCoordinateAction")


CoordinateAction = Annotated[
    TypeCoordinateAction
    | TapCoordinateAction
    | DoubleTapCoordinateAction
    | LongPressCoordinateAction
    | ClearCoordinateAction
    | DragAndDropAction
    | KeyEventAction
    | SwipeAction
    | CapabilityAction
    | WaitAction
    | TargetNotListedAction,
    Field(discriminator="type"),
]
