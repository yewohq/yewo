"""Canonical action, outcome-signal, and per-action runtime contracts."""

from dataclasses import dataclass
from enum import StrEnum

from yewo.grounding import GroundingTargetOperation


class ActionKind(StrEnum):
    """Actions understood by the planner-facing runtime contract."""

    TAP = "tap"
    FOCUS = "focus"
    TYPE = "type"
    CLEAR = "clear"
    DOUBLE_TAP = "double_tap"
    LONG_PRESS = "long_press"
    DRAG = "drag"
    SCROLL = "scroll"
    BACK = "back"
    DISMISS_KEYBOARD = "dismiss_keyboard"
    WAIT = "wait"
    SET_DEVICE_CAPABILITY = "set_device_capability"
    DONE = "done"
    CANNOT_PROGRESS = "cannot_progress"


class SideEffectClass(StrEnum):
    """Conservative side effects used by mode policy."""

    NONE = "none"
    NAVIGATION = "navigation"
    APPLICATION_MUTATION = "application_mutation"
    UNCLASSIFIED_INTERACTION = "unclassified_interaction"
    DEVICE_CONFIGURATION = "device_configuration"


class ActionHandling(StrEnum):
    """The runtime boundary that owns an action."""

    EXECUTE = "execute"
    TERMINAL = "terminal"
    CAPABILITY = "capability"


class ActionOutcomeSignal(StrEnum):
    """One observed change associated with a dispatched action."""

    PIXELS_CHANGED = "pixels_changed"
    CONTEXT_CHANGED = "context_changed"
    KEYBOARD_OPENED = "keyboard_opened"
    KEYBOARD_CLOSED = "keyboard_closed"
    FOCUS_CHANGED = "focus_changed"
    TEXT_LANDED = "text_landed"
    TEXT_NOT_LANDED = "text_not_landed"
    APP_BACKGROUNDED = "app_backgrounded"
    SYSTEM_SURFACE_OPENED = "system_surface_opened"
    APP_EXITED = "app_exited"

    @classmethod
    def _missing_(cls, value: object) -> "ActionOutcomeSignal | None":
        if value == "screen_changed":
            return cls.PIXELS_CHANGED
        return None


@dataclass(frozen=True, kw_only=True)
class ActionDefinition:
    """Classification, grounding, and proposal fields for one action."""

    handling: ActionHandling
    side_effect: SideEffectClass
    required_fields: frozenset[str]
    optional_fields: frozenset[str] = frozenset()
    target_operation: GroundingTargetOperation | None = None
    destination_operation: GroundingTargetOperation | None = None
    blocked_when_keyboard_visible: bool = False
    requires_keyboard_visible: bool = False
    incidental_replay_signals: frozenset[ActionOutcomeSignal] = frozenset()


_COMMON = frozenset({"expected_outcome", "reason"})
_TARGET = frozenset({"target_expression", "target_element_ref"})
ACTION_DEFINITIONS = {
    ActionKind.TAP: ActionDefinition(
        handling=ActionHandling.EXECUTE,
        side_effect=SideEffectClass.UNCLASSIFIED_INTERACTION,
        required_fields=_COMMON,
        optional_fields=_TARGET,
        target_operation=GroundingTargetOperation.TAP,
        blocked_when_keyboard_visible=True,
    ),
    ActionKind.FOCUS: ActionDefinition(
        handling=ActionHandling.EXECUTE,
        side_effect=SideEffectClass.UNCLASSIFIED_INTERACTION,
        required_fields=_COMMON,
        optional_fields=_TARGET,
        target_operation=GroundingTargetOperation.TEXT_FIELD,
    ),
    ActionKind.TYPE: ActionDefinition(
        handling=ActionHandling.EXECUTE,
        side_effect=SideEffectClass.APPLICATION_MUTATION,
        required_fields=_COMMON | {"text"},
        optional_fields=_TARGET,
        target_operation=GroundingTargetOperation.TEXT_FIELD,
        requires_keyboard_visible=True,
        incidental_replay_signals=frozenset({ActionOutcomeSignal.FOCUS_CHANGED}),
    ),
    ActionKind.CLEAR: ActionDefinition(
        handling=ActionHandling.EXECUTE,
        side_effect=SideEffectClass.APPLICATION_MUTATION,
        required_fields=_COMMON,
        requires_keyboard_visible=True,
        incidental_replay_signals=frozenset({ActionOutcomeSignal.FOCUS_CHANGED}),
    ),
    ActionKind.DOUBLE_TAP: ActionDefinition(
        handling=ActionHandling.EXECUTE,
        side_effect=SideEffectClass.UNCLASSIFIED_INTERACTION,
        required_fields=_COMMON,
        optional_fields=_TARGET,
        target_operation=GroundingTargetOperation.TAP,
        blocked_when_keyboard_visible=True,
    ),
    ActionKind.LONG_PRESS: ActionDefinition(
        handling=ActionHandling.EXECUTE,
        side_effect=SideEffectClass.UNCLASSIFIED_INTERACTION,
        required_fields=_COMMON | {"duration_seconds"},
        optional_fields=_TARGET,
        target_operation=GroundingTargetOperation.TAP,
        blocked_when_keyboard_visible=True,
    ),
    ActionKind.DRAG: ActionDefinition(
        handling=ActionHandling.EXECUTE,
        side_effect=SideEffectClass.UNCLASSIFIED_INTERACTION,
        required_fields=_COMMON | {"duration_seconds"},
        optional_fields=_TARGET | {"destination_expression", "destination_element_ref"},
        target_operation=GroundingTargetOperation.DRAG_SOURCE,
        destination_operation=GroundingTargetOperation.DRAG_DESTINATION,
        blocked_when_keyboard_visible=True,
    ),
    ActionKind.SCROLL: ActionDefinition(
        handling=ActionHandling.EXECUTE,
        side_effect=SideEffectClass.NAVIGATION,
        required_fields=_COMMON | {"direction", "distance"},
        optional_fields=_TARGET,
        target_operation=GroundingTargetOperation.SCROLL_REGION,
        blocked_when_keyboard_visible=True,
    ),
    ActionKind.BACK: ActionDefinition(
        handling=ActionHandling.EXECUTE,
        side_effect=SideEffectClass.NAVIGATION,
        required_fields=_COMMON,
        blocked_when_keyboard_visible=True,
    ),
    ActionKind.DISMISS_KEYBOARD: ActionDefinition(
        handling=ActionHandling.EXECUTE,
        side_effect=SideEffectClass.NAVIGATION,
        required_fields=_COMMON,
    ),
    ActionKind.WAIT: ActionDefinition(
        handling=ActionHandling.EXECUTE,
        side_effect=SideEffectClass.NONE,
        required_fields=_COMMON | {"duration_seconds"},
    ),
    ActionKind.SET_DEVICE_CAPABILITY: ActionDefinition(
        handling=ActionHandling.CAPABILITY,
        side_effect=SideEffectClass.DEVICE_CONFIGURATION,
        required_fields=frozenset({"reason"}),
    ),
    ActionKind.DONE: ActionDefinition(
        handling=ActionHandling.TERMINAL,
        side_effect=SideEffectClass.NONE,
        required_fields=frozenset({"reason"}),
    ),
    ActionKind.CANNOT_PROGRESS: ActionDefinition(
        handling=ActionHandling.TERMINAL,
        side_effect=SideEffectClass.NONE,
        required_fields=frozenset({"reason"}),
    ),
}
