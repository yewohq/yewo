"""Language- and resolution-independent identity for one device action."""

from dataclasses import dataclass
from typing import Literal, Optional

from yewo.agent.actuation.actions import (
    Action,
    ActionType,
    CapabilityAction,
    ClearAction,
    DoubleTapAction,
    DragAndDropAction,
    KeyEventAction,
    LongPressAction,
    PlannerTapAction,
    ShowKeyboardTargetsAction,
    SwipeAction,
    TapAction,
    TypeAction,
    WaitAction,
)
from yewo.agent.actuation.target_not_listed import TargetNotListedAction
from yewo.agent.memory.screen_identity import is_same_screen
from yewo.identity import derived_uuid

RELATIVE_POSITION_TOLERANCE = 0.02
PIXEL_POSITION_TOLERANCE = 40.0


@dataclass(frozen=True)
class ControlIdentity:
    """The action semantics and screen-relative points that identify a control."""

    action_type: str
    screen_phash: Optional[str]
    coordinate_space: Literal["relative", "pixels", "none"]
    points: tuple[tuple[float, float], ...] = ()
    qualifier: str = ""

    def matches(self, other: "ControlIdentity") -> bool:
        """Whether two observations describe the same action on the same control."""
        if (
            self.action_type != other.action_type
            or self.qualifier != other.qualifier
            or self.coordinate_space != other.coordinate_space
            or len(self.points) != len(other.points)
            or not is_same_screen(self.screen_phash, other.screen_phash)
        ):
            return False
        tolerance = (
            RELATIVE_POSITION_TOLERANCE
            if self.coordinate_space == "relative"
            else PIXEL_POSITION_TOLERANCE
        )
        return all(
            abs(left_x - right_x) <= tolerance
            and abs(left_y - right_y) <= tolerance
            for (left_x, left_y), (right_x, right_y) in zip(
                self.points, other.points, strict=True
            )
        )


def control_identity_id(identity: ControlIdentity) -> str:
    """Stable identifier for one exact control identity contract."""
    return derived_uuid(
        "control-identity-v1",
        identity.action_type,
        identity.screen_phash,
        identity.coordinate_space,
        identity.points,
        identity.qualifier,
    )


def _points(
    coordinates: tuple[tuple[int, int], ...],
    screen_size: Optional[tuple[int, int]],
) -> tuple[Literal["relative", "pixels"], tuple[tuple[float, float], ...]]:
    if screen_size is not None and screen_size[0] > 0 and screen_size[1] > 0:
        width, height = screen_size
        return "relative", tuple((x / width, y / height) for x, y in coordinates)
    return "pixels", tuple((float(x), float(y)) for x, y in coordinates)


def build_control_identity(
    action: Action | PlannerTapAction,
    screen_phash: Optional[str],
    screen_size: Optional[tuple[int, int]],
) -> ControlIdentity:
    """Build a complete identity for every executable action type."""
    action_type = (
        action.type.value if isinstance(action.type, ActionType) else str(action.type)
    )
    coordinates: tuple[tuple[int, int], ...] = ()
    qualifier = ""

    if isinstance(action, TypeAction | ClearAction | DoubleTapAction):
        coordinates = ((action.coordinates.x, action.coordinates.y),)
    elif isinstance(action, TapAction | PlannerTapAction):
        coordinates = ((action.coordinates.x, action.coordinates.y),)
        qualifier = f"{action.count}:{action.interval_ms}"
    elif isinstance(action, LongPressAction):
        coordinates = ((action.coordinates.x, action.coordinates.y),)
        qualifier = str(action.duration)
    elif isinstance(action, DragAndDropAction):
        coordinates = (
            (action.source_x, action.source_y),
            (action.destination_x, action.destination_y),
        )
    elif isinstance(action, SwipeAction):
        coordinates = ((action.start_x, action.start_y), (action.end_x, action.end_y))
    elif isinstance(action, KeyEventAction):
        qualifier = str(action.key)
    elif isinstance(action, CapabilityAction):
        qualifier = action.capability.model_dump_json()
    elif isinstance(action, WaitAction):
        qualifier = str(action.duration)
    elif isinstance(action, TargetNotListedAction):
        qualifier = action.region
    elif not isinstance(action, ShowKeyboardTargetsAction):
        raise TypeError(f"Action identity is not defined for {type(action).__name__}")

    if not coordinates:
        return ControlIdentity(action_type, screen_phash, "none", qualifier=qualifier)
    coordinate_space, normalized = _points(coordinates, screen_size)
    return ControlIdentity(
        action_type,
        screen_phash,
        coordinate_space,
        normalized,
        qualifier,
    )
