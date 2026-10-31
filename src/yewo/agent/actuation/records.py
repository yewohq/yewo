"""Typed proposals, authorizations, refusals, and dispatch receipts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yewo.artifacts import ArtifactRetention, ArtifactSensitivity, RecordIdentifier
from yewo.devices.interaction import (
    DEFAULT_MAXIMUM_WAIT_SECONDS,
    MAXIMUM_GESTURE_SECONDS,
    ScrollDirection,
)
from yewo.grounding import GroundingTargetOperation, PixelPoint
from yewo.observations import PixelRegion


class ActionKind(StrEnum):
    """Actions understood by the planner-facing runtime contract."""

    TAP = "tap"
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


@dataclass(frozen=True)
class ActionDefinition:
    """Stable classification and grounding rules for one action."""

    handling: ActionHandling
    side_effect: SideEffectClass
    target_operation: GroundingTargetOperation | None = None
    destination_operation: GroundingTargetOperation | None = None
    blocked_when_keyboard_visible: bool = False


ACTION_DEFINITIONS = {
    ActionKind.TAP: ActionDefinition(
        handling=ActionHandling.EXECUTE,
        side_effect=SideEffectClass.UNCLASSIFIED_INTERACTION,
        target_operation=GroundingTargetOperation.TAP,
        blocked_when_keyboard_visible=True,
    ),
    ActionKind.TYPE: ActionDefinition(
        ActionHandling.EXECUTE,
        SideEffectClass.APPLICATION_MUTATION,
        GroundingTargetOperation.TEXT_FIELD,
    ),
    ActionKind.CLEAR: ActionDefinition(
        ActionHandling.EXECUTE,
        SideEffectClass.APPLICATION_MUTATION,
        GroundingTargetOperation.TEXT_FIELD,
    ),
    ActionKind.DOUBLE_TAP: ActionDefinition(
        handling=ActionHandling.EXECUTE,
        side_effect=SideEffectClass.UNCLASSIFIED_INTERACTION,
        target_operation=GroundingTargetOperation.TAP,
        blocked_when_keyboard_visible=True,
    ),
    ActionKind.LONG_PRESS: ActionDefinition(
        handling=ActionHandling.EXECUTE,
        side_effect=SideEffectClass.UNCLASSIFIED_INTERACTION,
        target_operation=GroundingTargetOperation.TAP,
        blocked_when_keyboard_visible=True,
    ),
    ActionKind.DRAG: ActionDefinition(
        handling=ActionHandling.EXECUTE,
        side_effect=SideEffectClass.UNCLASSIFIED_INTERACTION,
        target_operation=GroundingTargetOperation.DRAG_SOURCE,
        destination_operation=GroundingTargetOperation.DRAG_DESTINATION,
        blocked_when_keyboard_visible=True,
    ),
    ActionKind.SCROLL: ActionDefinition(
        handling=ActionHandling.EXECUTE,
        side_effect=SideEffectClass.NAVIGATION,
        target_operation=GroundingTargetOperation.SCROLL_REGION,
        blocked_when_keyboard_visible=True,
    ),
    ActionKind.BACK: ActionDefinition(
        ActionHandling.EXECUTE,
        SideEffectClass.NAVIGATION,
        blocked_when_keyboard_visible=True,
    ),
    ActionKind.DISMISS_KEYBOARD: ActionDefinition(
        ActionHandling.EXECUTE,
        SideEffectClass.NAVIGATION,
    ),
    ActionKind.WAIT: ActionDefinition(ActionHandling.EXECUTE, SideEffectClass.NONE),
    ActionKind.SET_DEVICE_CAPABILITY: ActionDefinition(
        ActionHandling.CAPABILITY,
        SideEffectClass.DEVICE_CONFIGURATION,
    ),
    ActionKind.DONE: ActionDefinition(ActionHandling.TERMINAL, SideEffectClass.NONE),
    ActionKind.CANNOT_PROGRESS: ActionDefinition(
        ActionHandling.TERMINAL,
        SideEffectClass.NONE,
    ),
}


class ActionSource(StrEnum):
    """Trusted runtime source of one authorization request."""

    PLANNER = "planner"
    DETERMINISTIC_TEST = "deterministic_test"


class ScrollDistance(StrEnum):
    """Bounded semantic scroll distances."""

    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class LiteralText(BaseModel):
    """Non-secret text that can remain in run-scoped records."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: Literal["literal"] = "literal"
    value: str = Field(min_length=1, max_length=10_000)
    sensitivity: Literal[ArtifactSensitivity.DEVICE_CONTENT] = ArtifactSensitivity.DEVICE_CONTENT
    retention: Literal[ArtifactRetention.RUN] = ArtifactRetention.RUN


class DeclaredValueReference(BaseModel):
    """A secret-safe reference resolved only at the device boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    source: Literal["declared_value"] = "declared_value"
    value_name: RecordIdentifier
    sensitivity: Literal[ArtifactSensitivity.DECLARED_SENSITIVE] = (
        ArtifactSensitivity.DECLARED_SENSITIVE
    )
    retention: Literal[ArtifactRetention.RUN] = ArtifactRetention.RUN


TextInput = Annotated[
    LiteralText | DeclaredValueReference,
    Field(discriminator="source"),
]


class ActionProposal(BaseModel):
    """A planner proposal tied to its source observation."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    schema_version: Literal[1] = 1
    proposal_id: RecordIdentifier
    source_observation_id: RecordIdentifier
    action: ActionKind
    target_expression: str | None = Field(default=None, min_length=1, max_length=1_000)
    destination_expression: str | None = Field(default=None, min_length=1, max_length=1_000)
    text: TextInput | None = None
    duration_seconds: float | None = Field(default=None, ge=0)
    direction: ScrollDirection | None = None
    distance: ScrollDistance | None = None
    expected_outcome: str | None = Field(default=None, min_length=1, max_length=2_000)
    reason: str | None = Field(default=None, min_length=1, max_length=2_000)
    sensitivity: Literal[ArtifactSensitivity.DEVICE_CONTENT] = ArtifactSensitivity.DEVICE_CONTENT
    retention: Literal[ArtifactRetention.RUN] = ArtifactRetention.RUN

    @model_validator(mode="after")
    def _validate_action_fields(self) -> Self:
        required: dict[ActionKind, frozenset[str]] = {
            ActionKind.TAP: frozenset({"target_expression", "expected_outcome"}),
            ActionKind.TYPE: frozenset({"target_expression", "text", "expected_outcome"}),
            ActionKind.CLEAR: frozenset({"target_expression", "expected_outcome"}),
            ActionKind.DOUBLE_TAP: frozenset({"target_expression", "expected_outcome"}),
            ActionKind.LONG_PRESS: frozenset(
                {"target_expression", "duration_seconds", "expected_outcome"}
            ),
            ActionKind.DRAG: frozenset(
                {
                    "target_expression",
                    "destination_expression",
                    "duration_seconds",
                    "expected_outcome",
                }
            ),
            ActionKind.SCROLL: frozenset({"direction", "distance", "expected_outcome"}),
            ActionKind.BACK: frozenset({"expected_outcome"}),
            ActionKind.DISMISS_KEYBOARD: frozenset({"expected_outcome"}),
            ActionKind.WAIT: frozenset({"duration_seconds", "expected_outcome"}),
            ActionKind.SET_DEVICE_CAPABILITY: frozenset({"reason"}),
            ActionKind.DONE: frozenset({"reason"}),
            ActionKind.CANNOT_PROGRESS: frozenset({"reason"}),
        }
        optional: dict[ActionKind, frozenset[str]] = {
            ActionKind.SCROLL: frozenset({"target_expression"}),
        }
        action_fields = {
            "target_expression": self.target_expression,
            "destination_expression": self.destination_expression,
            "text": self.text,
            "duration_seconds": self.duration_seconds,
            "direction": self.direction,
            "distance": self.distance,
            "expected_outcome": self.expected_outcome,
            "reason": self.reason,
        }
        present = {name for name, value in action_fields.items() if value is not None}
        expected = required[self.action]
        if not expected.issubset(present):
            missing = ", ".join(sorted(expected - present))
            raise ValueError(f"{self.action.value} requires: {missing}.")
        unexpected = present - expected - optional.get(self.action, frozenset())
        if unexpected:
            names = ", ".join(sorted(unexpected))
            raise ValueError(f"{self.action.value} does not accept: {names}.")
        return self


class TapRepetition(BaseModel):
    """A bounded repeat count for one deterministic test action."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    count: int = Field(ge=2, le=10)
    interval_seconds: float = Field(default=0, ge=0, le=1)


class ActionBudget(BaseModel):
    """Runtime-owned usage counters supplied to authorization."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dispatched_actions: int = Field(ge=0)


class ActionPolicyLimits(BaseModel):
    """Mode-specific limits consumed by the policy boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    allowed_actions: frozenset[ActionKind]
    allowed_side_effects: frozenset[SideEffectClass]
    supported_actions: frozenset[ActionKind]
    maximum_actions: int = Field(gt=0)
    maximum_wait_seconds: float = Field(
        default=DEFAULT_MAXIMUM_WAIT_SECONDS,
        gt=0,
        le=DEFAULT_MAXIMUM_WAIT_SECONDS,
    )
    maximum_gesture_seconds: float = Field(
        default=MAXIMUM_GESTURE_SECONDS,
        gt=0,
        le=MAXIMUM_GESTURE_SECONDS,
    )
    allow_exploratory_back: bool = False

    @model_validator(mode="after")
    def _validate_keyboard_exit(self) -> Self:
        available_actions = self.allowed_actions & self.supported_actions
        has_keyboard_blocked_action = any(
            ACTION_DEFINITIONS[action].blocked_when_keyboard_visible for action in available_actions
        )
        if has_keyboard_blocked_action and ActionKind.DISMISS_KEYBOARD not in available_actions:
            raise ValueError("A mode with keyboard-blocked actions must allow keyboard dismissal.")
        return self


class ActionRefusalCode(StrEnum):
    """Stable reasons why a proposal was not authorized."""

    STALE_OBSERVATION = "stale_observation"
    STALE_GROUNDING = "stale_grounding"
    INVALID_GROUNDING = "invalid_grounding"
    ACTION_NOT_ALLOWED = "action_not_allowed"
    SIDE_EFFECT_NOT_ALLOWED = "side_effect_not_allowed"
    UNSUPPORTED_ACTION = "unsupported_action"
    INVALID_CONSTRAINT = "invalid_constraint"
    KEYBOARD_RULE = "keyboard_rule"
    APPLICATION_SCOPE = "application_scope"
    ACTION_BUDGET_EXHAUSTED = "action_budget_exhausted"
    NON_DISPATCH_ACTION = "non_dispatch_action"


class ActionRefusal(BaseModel):
    """A fail-closed policy decision with a stable reason code."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    proposal_id: RecordIdentifier
    code: ActionRefusalCode
    reason: str = Field(min_length=1)
    sensitivity: ArtifactSensitivity = ArtifactSensitivity.DEVICE_CONTENT
    retention: Literal[ArtifactRetention.RUN] = ArtifactRetention.RUN


class AuthorizedAction(BaseModel):
    """An immutable authorization bound to exact observation evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    authorization_id: RecordIdentifier
    proposal_id: RecordIdentifier
    source_observation_id: RecordIdentifier
    source_screenshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    application_id: str = Field(min_length=1)
    action: ActionKind
    source: ActionSource
    side_effect: SideEffectClass
    expected_outcome: str = Field(min_length=1, max_length=2_000)
    target: PixelPoint | None = None
    destination: PixelPoint | None = None
    region: PixelRegion | None = None
    text: TextInput | None = None
    duration_seconds: float | None = None
    direction: ScrollDirection | None = None
    distance: ScrollDistance | None = None
    tap_count: int = Field(default=1, ge=1, le=10)
    tap_interval_seconds: float = Field(default=0, ge=0, le=1)
    expected_back_destination_observation_id: RecordIdentifier | None = None
    sensitivity: ArtifactSensitivity
    retention: Literal[ArtifactRetention.RUN]

    @model_validator(mode="after")
    def _validate_action_fields(self) -> Self:
        if self.side_effect is not ACTION_DEFINITIONS[self.action].side_effect:
            raise ValueError("The side-effect class does not match the action.")
        if self.text is not None and self.sensitivity is not self.text.sensitivity:
            raise ValueError("The authorization sensitivity does not match its text input.")
        required: dict[ActionKind, frozenset[str]] = {
            ActionKind.TAP: frozenset({"target"}),
            ActionKind.TYPE: frozenset({"target", "text"}),
            ActionKind.CLEAR: frozenset({"target"}),
            ActionKind.DOUBLE_TAP: frozenset({"target"}),
            ActionKind.LONG_PRESS: frozenset({"target", "duration_seconds"}),
            ActionKind.DRAG: frozenset({"target", "destination", "duration_seconds"}),
            ActionKind.SCROLL: frozenset({"direction", "distance"}),
            ActionKind.BACK: frozenset(),
            ActionKind.DISMISS_KEYBOARD: frozenset(),
            ActionKind.WAIT: frozenset({"duration_seconds"}),
        }
        optional: dict[ActionKind, frozenset[str]] = {
            ActionKind.SCROLL: frozenset({"region"}),
        }
        action_fields = {
            "target": self.target,
            "destination": self.destination,
            "region": self.region,
            "text": self.text,
            "duration_seconds": self.duration_seconds,
            "direction": self.direction,
            "distance": self.distance,
        }
        present = {name for name, value in action_fields.items() if value is not None}
        expected = required.get(self.action)
        if expected is None:
            raise ValueError(f"{self.action.value} is not dispatchable.")
        if not expected.issubset(present):
            missing = ", ".join(sorted(expected - present))
            raise ValueError(f"{self.action.value} authorization requires: {missing}.")
        unexpected = present - expected - optional.get(self.action, frozenset())
        if unexpected:
            names = ", ".join(sorted(unexpected))
            raise ValueError(f"{self.action.value} authorization does not accept: {names}.")
        if self.action is not ActionKind.TAP and (
            self.tap_count != 1 or self.tap_interval_seconds != 0
        ):
            raise ValueError("Only a tap authorization can contain repetition.")
        if self.tap_count == 1 and self.tap_interval_seconds != 0:
            raise ValueError("A single tap cannot contain a repeat interval.")
        if self.tap_count > 1 and self.source is not ActionSource.DETERMINISTIC_TEST:
            raise ValueError("Only a deterministic test can repeat a tap.")
        return self


class DispatchDisposition(StrEnum):
    """Whether the device boundary accepted the dispatch."""

    DISPATCHED = "dispatched"
    DEVICE_REFUSED = "device_refused"


class DispatchReceipt(BaseModel):
    """A dispatch receipt that contains no action-outcome claim."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    receipt_id: RecordIdentifier
    authorization_id: RecordIdentifier
    proposal_id: RecordIdentifier
    action: ActionKind
    disposition: DispatchDisposition
    reason: str | None = Field(default=None, min_length=1)
    sensitivity: ArtifactSensitivity
    retention: Literal[ArtifactRetention.RUN]

    @model_validator(mode="after")
    def _validate_disposition(self) -> Self:
        if self.disposition is DispatchDisposition.DEVICE_REFUSED and self.reason is None:
            raise ValueError("A refused dispatch must contain a reason.")
        if self.disposition is DispatchDisposition.DISPATCHED and self.reason is not None:
            raise ValueError("A dispatched receipt cannot contain a refusal reason.")
        return self
