"""Typed script and evidence records for the device-foundation workflow."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yewo.artifacts import RecordIdentifier
from yewo.devices import DevicePlatform
from yewo.devices.interaction import AndroidKey, ScrollDirection
from yewo.observations import RedactionRegion


class DeviceFoundationActionKind(StrEnum):
    """A deterministic operation implemented by the device interaction layer."""

    TAP = "tap"
    DOUBLE_TAP = "double_tap"
    LONG_PRESS = "long_press"
    DRAG = "drag"
    SWIPE = "swipe"
    SCROLL = "scroll"
    TYPE_TEXT = "type_text"
    CLEAR_TEXT = "clear_text"
    PRESS_KEY = "press_key"
    BACK = "back"
    SUBMIT = "submit"
    DISMISS_KEYBOARD = "dismiss_keyboard"
    WAIT = "wait"
    ACTIVATE = "activate"
    TERMINATE = "terminate"
    CLEAR_APPLICATION_DATA = "clear_application_data"
    RELAUNCH = "relaunch"
    BACKGROUND_AND_RESUME = "background_and_resume"
    OPEN_DEEP_LINK = "open_deep_link"
    RECOVER_TO_APPLICATION = "recover_to_application"


class _Action(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action_id: RecordIdentifier


class TapAction(_Action):
    kind: Literal[DeviceFoundationActionKind.TAP] = DeviceFoundationActionKind.TAP
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    count: int = Field(default=1, ge=1, le=10)
    interval_seconds: float = Field(default=0.1, ge=0, le=1)


class DoubleTapAction(_Action):
    kind: Literal[DeviceFoundationActionKind.DOUBLE_TAP] = DeviceFoundationActionKind.DOUBLE_TAP
    x: int = Field(ge=0)
    y: int = Field(ge=0)


class LongPressAction(_Action):
    kind: Literal[DeviceFoundationActionKind.LONG_PRESS] = DeviceFoundationActionKind.LONG_PRESS
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    duration_seconds: float = Field(ge=0.1, le=10)


class PointerMotionAction(_Action):
    kind: Literal[
        DeviceFoundationActionKind.DRAG,
        DeviceFoundationActionKind.SWIPE,
    ]
    start_x: int = Field(ge=0)
    start_y: int = Field(ge=0)
    end_x: int = Field(ge=0)
    end_y: int = Field(ge=0)
    duration_seconds: float = Field(default=1, ge=0.1, le=10)


class ScrollAction(_Action):
    kind: Literal[DeviceFoundationActionKind.SCROLL] = DeviceFoundationActionKind.SCROLL
    direction: ScrollDirection
    left: int = Field(default=0, ge=0)
    top: int = Field(default=0, ge=0)
    right: int | None = Field(default=None, ge=0)
    bottom: int | None = Field(default=None, ge=0)


class TypeTextAction(_Action):
    kind: Literal[DeviceFoundationActionKind.TYPE_TEXT] = DeviceFoundationActionKind.TYPE_TEXT
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    text: str = Field(min_length=1)


class ClearTextAction(_Action):
    kind: Literal[DeviceFoundationActionKind.CLEAR_TEXT] = DeviceFoundationActionKind.CLEAR_TEXT
    x: int = Field(ge=0)
    y: int = Field(ge=0)


class PressKeyAction(_Action):
    kind: Literal[DeviceFoundationActionKind.PRESS_KEY] = DeviceFoundationActionKind.PRESS_KEY
    key: str = Field(min_length=1)


class NoArgumentAction(_Action):
    kind: Literal[
        DeviceFoundationActionKind.BACK,
        DeviceFoundationActionKind.SUBMIT,
        DeviceFoundationActionKind.DISMISS_KEYBOARD,
        DeviceFoundationActionKind.ACTIVATE,
        DeviceFoundationActionKind.TERMINATE,
        DeviceFoundationActionKind.CLEAR_APPLICATION_DATA,
        DeviceFoundationActionKind.RELAUNCH,
        DeviceFoundationActionKind.RECOVER_TO_APPLICATION,
    ]


class WaitAction(_Action):
    kind: Literal[DeviceFoundationActionKind.WAIT] = DeviceFoundationActionKind.WAIT
    duration_seconds: float = Field(ge=0, le=30)


class BackgroundAction(_Action):
    kind: Literal[DeviceFoundationActionKind.BACKGROUND_AND_RESUME] = (
        DeviceFoundationActionKind.BACKGROUND_AND_RESUME
    )
    duration_seconds: int = Field(ge=1, le=30)


class DeepLinkAction(_Action):
    kind: Literal[DeviceFoundationActionKind.OPEN_DEEP_LINK] = (
        DeviceFoundationActionKind.OPEN_DEEP_LINK
    )
    url: str = Field(min_length=1, max_length=2048)


DeviceFoundationAction = Annotated[
    TapAction
    | DoubleTapAction
    | LongPressAction
    | PointerMotionAction
    | ScrollAction
    | TypeTextAction
    | ClearTextAction
    | PressKeyAction
    | NoArgumentAction
    | WaitAction
    | BackgroundAction
    | DeepLinkAction,
    Field(discriminator="kind"),
]


class DeviceFoundationScript(BaseModel):
    """One bounded, predeclared sequence for a controlled application."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    script_id: RecordIdentifier
    platform: DevicePlatform
    actions: tuple[DeviceFoundationAction, ...] = Field(min_length=1, max_length=100)
    redaction_regions: tuple[RedactionRegion, ...] = Field(default=(), max_length=100)
    device_log_sensitive_values: tuple[str, ...] = Field(
        default=(),
        max_length=100,
        repr=False,
    )

    @model_validator(mode="after")
    def _validate_action_ids(self) -> Self:
        action_ids = tuple(action.action_id for action in self.actions)
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("A device-foundation script cannot repeat an action ID.")
        if any(not value for value in self.device_log_sensitive_values):
            raise ValueError("A declared device-log sensitive value cannot be empty.")
        if len(self.device_log_sensitive_values) != len(set(self.device_log_sensitive_values)):
            raise ValueError("A device-foundation script cannot repeat a sensitive value.")
        if self.platform is DevicePlatform.IOS and any(
            isinstance(action, NoArgumentAction) and action.kind is DeviceFoundationActionKind.BACK
            for action in self.actions
        ):
            raise ValueError("An iOS device-foundation script cannot request Back navigation.")
        if self.platform is DevicePlatform.IOS and any(
            isinstance(action, PointerMotionAction) and action.duration_seconds < 0.5
            for action in self.actions
        ):
            raise ValueError("An iOS drag or swipe must last at least 0.5 seconds.")
        if self.platform is DevicePlatform.ANDROID:
            for action in self.actions:
                if isinstance(action, PressKeyAction):
                    AndroidKey.parse_name(action.key)
        return self


class DeviceFoundationActionStatus(StrEnum):
    """Whether one declared interaction completed without rejection."""

    COMPLETED = "completed"
    REJECTED = "rejected"


class DeviceFoundationActionReceipt(BaseModel):
    """Evidence linking one action to its immediate observations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    script_id: RecordIdentifier
    action_id: RecordIdentifier
    action_index: int = Field(ge=0)
    kind: DeviceFoundationActionKind
    status: DeviceFoundationActionStatus
    before_observation_id: RecordIdentifier
    after_observation_id: RecordIdentifier
    started_at: datetime
    completed_at: datetime

    @model_validator(mode="after")
    def _validate_times(self) -> Self:
        if self.started_at.utcoffset() is None or self.completed_at.utcoffset() is None:
            raise ValueError("Action receipt times must include a timezone.")
        if self.completed_at < self.started_at:
            raise ValueError("An action cannot complete before it starts.")
        return self


class DeviceFoundationResultState(StrEnum):
    """Whether the complete declared script ran."""

    COMPLETED = "completed"
    INCOMPLETE = "incomplete"


class DeviceFoundationResult(BaseModel):
    """Payload-free summary of one deterministic device script."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    script_id: RecordIdentifier
    platform: DevicePlatform
    state: DeviceFoundationResultState
    initial_observation_id: RecordIdentifier
    final_observation_id: RecordIdentifier
    declared_action_count: int = Field(ge=1)
    completed_action_ids: tuple[RecordIdentifier, ...]
    rejected_action_id: RecordIdentifier | None = None

    @model_validator(mode="after")
    def _validate_result(self) -> Self:
        if len(self.completed_action_ids) != len(set(self.completed_action_ids)):
            raise ValueError("A result cannot repeat a completed action ID.")
        if self.rejected_action_id in self.completed_action_ids:
            raise ValueError("A rejected action cannot also be completed.")
        if len(self.completed_action_ids) > self.declared_action_count:
            raise ValueError("A result cannot complete more actions than the script declared.")
        if self.state is DeviceFoundationResultState.COMPLETED:
            if (
                len(self.completed_action_ids) != self.declared_action_count
                or self.rejected_action_id is not None
            ):
                raise ValueError("A completed result must account for every declared action.")
        elif self.rejected_action_id is None:
            raise ValueError("An incomplete result must identify its rejected action.")
        return self
