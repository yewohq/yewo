"""Typed proposals, authorizations, refusals, and dispatch receipts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator

from yewo.artifacts import ArtifactRetention, ArtifactSensitivity, RecordIdentifier
from yewo.devices.interaction import (
    DEFAULT_MAXIMUM_WAIT_SECONDS,
    MAXIMUM_GESTURE_SECONDS,
    ScrollDirection,
)
from yewo.grounding import GroundingTargetOperation, PixelPoint
from yewo.observations import PixelRegion
from yewo.agent.actuation.action_catalog import (
    ACTION_DEFINITIONS,
    ActionKind,
    SideEffectClass,
)


MAXIMUM_ACTION_TARGET_EXPRESSION_LENGTH = 1_000


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


class ActionProposalDecision(BaseModel):
    """Model-authored action content without runtime-owned identity."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    action: ActionKind
    grounding_correction_for_proposal_id: RecordIdentifier | None = None
    interaction_refinement_for_proposal_id: RecordIdentifier | None = None
    target_element_ref: RecordIdentifier | None = None
    destination_element_ref: RecordIdentifier | None = None
    target_expression: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAXIMUM_ACTION_TARGET_EXPRESSION_LENGTH,
    )
    destination_expression: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAXIMUM_ACTION_TARGET_EXPRESSION_LENGTH,
    )
    text: TextInput | None = None
    duration_seconds: float | None = Field(default=None, ge=0)
    direction: ScrollDirection | None = None
    distance: ScrollDistance | None = None
    expected_outcome: str | None = Field(
        default=None,
        min_length=1,
        max_length=2_000,
        description=(
            "The planner's unverified prediction for this action. This value is not "
            "observed outcome evidence."
        ),
    )
    reason: str | None = Field(default=None, min_length=1, max_length=2_000)
    sensitivity: Literal[ArtifactSensitivity.DEVICE_CONTENT] = ArtifactSensitivity.DEVICE_CONTENT
    retention: Literal[ArtifactRetention.RUN] = ArtifactRetention.RUN

    @model_validator(mode="after")
    def _validate_action_fields(self) -> Self:
        definition = ACTION_DEFINITIONS[self.action]
        if self.grounding_correction_for_proposal_id is not None and (
            definition.target_operation is None
            or definition.target_operation is GroundingTargetOperation.SCROLL_REGION
            or definition.destination_operation is not None
        ):
            raise ValueError("A grounding correction requires one visual point target.")
        if self.interaction_refinement_for_proposal_id is not None and (
            self.target_expression is None
            or definition.target_operation is None
            or definition.target_operation is GroundingTargetOperation.SCROLL_REGION
            or definition.destination_operation is not None
        ):
            raise ValueError("An interaction refinement requires one visual point target.")
        if (
            self.grounding_correction_for_proposal_id is not None
            and self.interaction_refinement_for_proposal_id is not None
        ):
            raise ValueError(
                "A proposal cannot be both a grounding correction and an interaction refinement."
            )
        action_fields = {
            "target_element_ref": self.target_element_ref,
            "destination_element_ref": self.destination_element_ref,
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
        expected = definition.required_fields
        if not expected.issubset(present):
            missing = ", ".join(sorted(expected - present))
            raise ValueError(f"{self.action.value} requires: {missing}.")
        unexpected = present - expected - definition.optional_fields
        if unexpected:
            names = ", ".join(sorted(unexpected))
            raise ValueError(f"{self.action.value} does not accept: {names}.")
        target_fields = {"target_expression", "target_element_ref"}
        target_required = (
            definition.target_operation is not None
            and not target_fields.intersection(definition.optional_fields)
        )
        if target_required and self.target_expression is None and self.target_element_ref is None:
            raise ValueError(f"{self.action.value} requires a target.")
        if (
            definition.destination_operation is not None
            and self.destination_expression is None
            and self.destination_element_ref is None
        ):
            raise ValueError(f"{self.action.value} requires a destination.")
        return self

    @model_validator(mode="after")
    def _validate_planning_request_context(self, info: ValidationInfo) -> Self:
        """Reject a structurally valid proposal that cannot serve this planning request."""
        context = info.context
        if not isinstance(context, dict):
            return self
        available_actions = context.get("available_actions")
        if isinstance(available_actions, (tuple, list, frozenset, set)) and (
            self.action.value not in available_actions
        ):
            choices = ", ".join(sorted(str(action) for action in available_actions))
            raise ValueError(
                f"Action '{self.action.value}' is not currently available. "
                f"Choose one of: {choices}."
            )
        declared_value_names = context.get("declared_value_names")
        if (
            isinstance(self.text, DeclaredValueReference)
            and isinstance(declared_value_names, (tuple, list, frozenset, set))
            and self.text.value_name not in declared_value_names
        ):
            raise ValueError("Use only a declared value name listed in the planning context.")
        available_element_refs = context.get("available_element_refs")
        if isinstance(available_element_refs, (tuple, list, frozenset, set)):
            allowed_refs = set(available_element_refs)
            for reference in (self.target_element_ref, self.destination_element_ref):
                if reference is not None and reference not in allowed_refs:
                    raise ValueError(
                        "Use only an interactive element_ref from current_screen_analysis."
                    )
        if self.target_element_ref is not None and self.target_expression is not None:
            raise ValueError("Choose target_element_ref or target_expression, not both.")
        if self.destination_element_ref is not None and self.destination_expression is not None:
            raise ValueError("Choose destination_element_ref or destination_expression, not both.")
        correctable = context.get("correctable_grounding_proposals")
        failed_id = self.grounding_correction_for_proposal_id
        if failed_id is None or not isinstance(correctable, dict):
            return self
        failed = correctable.get(failed_id)
        if failed is None:
            raise ValueError(
                "Correct only a proposal whose grounded target failed and has no correction "
                "yet. When the target was grounded but the action had the wrong effect, use "
                "interaction_refinement_for_proposal_id instead."
            )
        failed_action, failed_target = failed
        if self.action.value != failed_action:
            raise ValueError(f"A grounding correction must repeat action '{failed_action}'.")
        if self.target_expression == failed_target:
            raise ValueError("A grounding correction must change the failed target expression.")
        return self


class ActionProposal(ActionProposalDecision):
    """A validated planner decision bound to one runtime observation."""

    schema_version: Literal[1] = 1
    proposal_id: RecordIdentifier
    source_observation_id: RecordIdentifier

    @model_validator(mode="after")
    def _validate_runtime_identity_references(self) -> Self:
        if self.grounding_correction_for_proposal_id == self.proposal_id:
            raise ValueError("A grounding correction cannot reference its own proposal.")
        if self.interaction_refinement_for_proposal_id == self.proposal_id:
            raise ValueError("An interaction refinement cannot reference its own proposal.")
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
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    INVALID_GROUNDING = "invalid_grounding"
    INVALID_INPUT = "invalid_input"
    ACTION_NOT_ALLOWED = "action_not_allowed"
    SIDE_EFFECT_NOT_ALLOWED = "side_effect_not_allowed"
    UNSUPPORTED_ACTION = "unsupported_action"
    INVALID_CONSTRAINT = "invalid_constraint"
    KEYBOARD_RULE = "keyboard_rule"
    APPLICATION_SCOPE = "application_scope"
    ACTION_BUDGET_EXHAUSTED = "action_budget_exhausted"
    NON_DISPATCH_ACTION = "non_dispatch_action"
    REPEATED_NO_PROGRESS = "repeated_no_progress"
    DISCOVERY_INCOMPLETE = "discovery_incomplete"
    COMPLETED_WORK_REGRESSION = "completed_work_regression"
    VERIFICATION_UNAVAILABLE = "verification_unavailable"
    EXCLUDED_MEMORY_TRANSITION = "excluded_memory_transition"


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
            ActionKind.FOCUS: frozenset({"target"}),
            ActionKind.TYPE: frozenset({"text"}),
            ActionKind.CLEAR: frozenset(),
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
            ActionKind.TYPE: frozenset({"target"}),
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
    DISPATCH_UNVERIFIED = "dispatch_unverified"
    TARGET_REJECTED = "target_rejected"
    INPUT_REJECTED = "input_rejected"
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
    dispatched_at: datetime | None = None
    reason: str | None = Field(default=None, min_length=1)
    sensitivity: ArtifactSensitivity
    retention: Literal[ArtifactRetention.RUN]

    @model_validator(mode="after")
    def _validate_disposition(self) -> Self:
        if self.dispatched_at is not None and self.dispatched_at.utcoffset() is None:
            raise ValueError("A dispatch timestamp must include a timezone.")
        if self.disposition is not DispatchDisposition.DISPATCHED and self.reason is None:
            raise ValueError("A non-accepted dispatch must contain a reason.")
        if self.disposition is DispatchDisposition.DISPATCHED and self.reason is not None:
            raise ValueError("A dispatched receipt cannot contain a refusal reason.")
        return self
