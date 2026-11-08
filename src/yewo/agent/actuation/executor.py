"""Single-dispatch executor for previously authorized device actions."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Protocol

from yewo.artifacts import new_record_identifier
from yewo.devices.errors import (
    DeviceActionRejectedError,
    DeviceInputRejectedError,
    DeviceTargetRejectedError,
)
from yewo.devices.focus import (
    FocusedFieldReading,
    TextFieldContext,
    TextFieldContextSource,
    TextFieldType,
)
from yewo.devices.interaction import ScrollDirection

from yewo.agent.actuation.outcomes import (
    TextDeliveryEvidence,
    TextDeliveryReason,
    TextDeliveryStatus,
)
from yewo.agent.actuation.action_catalog import ActionKind
from yewo.agent.actuation.records import (
    AuthorizedAction,
    DeclaredValueReference,
    DispatchDisposition,
    DispatchReceipt,
    LiteralText,
    ScrollDistance,
)


class ActionDevice(Protocol):
    """Existing device operations used by the executor."""

    def tap(self, x: int, y: int, *, count: int = 1, interval_seconds: float = 0.1) -> None: ...

    def focus(self, x: int, y: int) -> object: ...

    def type_focused_text(self, text: str) -> bool | None: ...

    def clear_focused_text(self) -> bool | None: ...

    def double_tap(self, x: int, y: int) -> None: ...

    def long_press(self, x: int, y: int, duration_seconds: float) -> None: ...

    def drag(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        *,
        duration_seconds: float = 1,
    ) -> None: ...

    def scroll(
        self,
        direction: ScrollDirection,
        *,
        left: int = 0,
        top: int = 0,
        right: int | None = None,
        bottom: int | None = None,
        distance_fraction: float = 0.6,
    ) -> None: ...

    def back(self) -> None: ...

    def dismiss_keyboard(self) -> bool: ...

    def wait(self, duration_seconds: float) -> None: ...

    def focused_field_reading(self) -> FocusedFieldReading: ...


class AuthorizationReplayError(RuntimeError):
    """The executor received one immutable authorization more than once."""


_SCROLL_FRACTIONS = {
    ScrollDistance.SHORT: 0.3,
    ScrollDistance.MEDIUM: 0.6,
    ScrollDistance.LONG: 0.85,
}


class ActionExecutor:
    """Consume one authorization once and dispatch exactly one device operation."""

    def __init__(
        self,
        device: ActionDevice,
        identifier_factory: Callable[[], str] = new_record_identifier,
        declared_value_resolver: Callable[[str], str] | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._device = device
        self._identifier_factory = identifier_factory
        self._declared_value_resolver = declared_value_resolver
        self._clock = clock
        self._consumed_authorizations: set[str] = set()
        self._text_delivery_by_authorization: dict[str, TextDeliveryEvidence] = {}

    def dispatch(self, action: AuthorizedAction) -> DispatchReceipt:
        if not isinstance(action, AuthorizedAction):
            raise TypeError("ActionExecutor requires an AuthorizedAction.")
        if action.authorization_id in self._consumed_authorizations:
            raise AuthorizationReplayError("The executor already consumed this authorization.")
        self._consumed_authorizations.add(action.authorization_id)
        try:
            text_delivery = self._dispatch(action)
        except DeviceTargetRejectedError as error:
            return DispatchReceipt(
                receipt_id=self._identifier_factory(),
                authorization_id=action.authorization_id,
                proposal_id=action.proposal_id,
                action=action.action,
                disposition=DispatchDisposition.TARGET_REJECTED,
                dispatched_at=self._clock(),
                reason=str(error),
                sensitivity=action.sensitivity,
                retention=action.retention,
            )
        except DeviceInputRejectedError as error:
            return DispatchReceipt(
                receipt_id=self._identifier_factory(),
                authorization_id=action.authorization_id,
                proposal_id=action.proposal_id,
                action=action.action,
                disposition=DispatchDisposition.INPUT_REJECTED,
                dispatched_at=self._clock(),
                reason=str(error),
                sensitivity=action.sensitivity,
                retention=action.retention,
            )
        except DeviceActionRejectedError:
            return DispatchReceipt(
                receipt_id=self._identifier_factory(),
                authorization_id=action.authorization_id,
                proposal_id=action.proposal_id,
                action=action.action,
                disposition=DispatchDisposition.DEVICE_REFUSED,
                dispatched_at=self._clock(),
                reason="The device rejected the authorized action.",
                sensitivity=action.sensitivity,
                retention=action.retention,
            )
        if text_delivery is not None:
            self._text_delivery_by_authorization[action.authorization_id] = text_delivery
        return DispatchReceipt(
            receipt_id=self._identifier_factory(),
            authorization_id=action.authorization_id,
            proposal_id=action.proposal_id,
            action=action.action,
            disposition=DispatchDisposition.DISPATCHED,
            dispatched_at=self._clock(),
            sensitivity=action.sensitivity,
            retention=action.retention,
        )

    def focused_field_reading(self) -> FocusedFieldReading:
        """Read focus evidence without exposing active-field text."""
        return self._device.focused_field_reading()

    def verify_text_delivery(self, action: AuthorizedAction) -> TextDeliveryEvidence | None:
        """Compare entered text inside the device boundary and discard the value."""
        if action.action not in {ActionKind.TYPE, ActionKind.CLEAR}:
            return None
        return self._text_delivery_by_authorization.pop(
            action.authorization_id,
            _unavailable_text_delivery(),
        )

    def discard_text_delivery(self, action: AuthorizedAction) -> None:
        """Discard a pending comparison value when post-action evidence is unavailable."""
        self._text_delivery_by_authorization.pop(action.authorization_id, None)

    def _dispatch(self, action: AuthorizedAction) -> TextDeliveryEvidence | None:
        target = action.target
        if action.action is ActionKind.TAP and target is not None:
            self._device.tap(
                target.x,
                target.y,
                count=action.tap_count,
                interval_seconds=action.tap_interval_seconds,
            )
        elif action.action is ActionKind.FOCUS and target is not None:
            self._device.focus(target.x, target.y)
        elif action.action is ActionKind.TYPE and action.text is not None:
            if target is not None:
                self._device.focus(target.x, target.y)
            expected_text = self._resolve_text(action.text)
            context = self._device.focused_field_reading().field_context
            is_text_match = self._device.type_focused_text(expected_text)
            return _delivery_evidence(is_text_match, context)
        elif action.action is ActionKind.CLEAR:
            context = self._device.focused_field_reading().field_context
            return _clear_delivery_evidence(self._device.clear_focused_text(), context)
        elif action.action is ActionKind.DOUBLE_TAP and target is not None:
            self._device.double_tap(target.x, target.y)
        elif (
            action.action is ActionKind.LONG_PRESS
            and target is not None
            and action.duration_seconds is not None
        ):
            self._device.long_press(target.x, target.y, action.duration_seconds)
        elif (
            action.action is ActionKind.DRAG
            and target is not None
            and action.destination is not None
            and action.duration_seconds is not None
        ):
            self._device.drag(
                target.x,
                target.y,
                action.destination.x,
                action.destination.y,
                duration_seconds=action.duration_seconds,
            )
        elif (
            action.action is ActionKind.SCROLL
            and action.direction is not None
            and action.distance is not None
        ):
            region = action.region
            # Grounded regions are half-open. Device coordinates use inclusive edges.
            self._device.scroll(
                action.direction,
                left=region.left if region else 0,
                top=region.top if region else 0,
                right=region.right - 1 if region else None,
                bottom=region.bottom - 1 if region else None,
                distance_fraction=_SCROLL_FRACTIONS[action.distance],
            )
        elif action.action is ActionKind.BACK:
            self._device.back()
        elif action.action is ActionKind.DISMISS_KEYBOARD:
            self._device.dismiss_keyboard()
        elif action.action is ActionKind.WAIT and action.duration_seconds is not None:
            self._device.wait(action.duration_seconds)
        else:
            raise DeviceActionRejectedError(
                "Authorization does not describe a dispatchable action."
            )
        return None

    def _resolve_text(self, text: LiteralText | DeclaredValueReference) -> str:
        if isinstance(text, LiteralText):
            return text.value
        if self._declared_value_resolver is None:
            raise DeviceActionRejectedError("A declared value resolver is unavailable.")
        value = self._declared_value_resolver(text.value_name)
        if not isinstance(value, str) or not value:
            raise DeviceActionRejectedError("The declared value resolver returned no text.")
        return value


def _delivery_evidence(
    is_text_match: bool | None,
    context: TextFieldContext | None,
) -> TextDeliveryEvidence:
    context = context or _unknown_text_field_context()
    if context.is_secure is True:
        return TextDeliveryEvidence(
            status=TextDeliveryStatus.UNVERIFIABLE,
            reason=TextDeliveryReason.SECURE_FIELD,
            is_secure=True,
            field_type=context.field_type,
            context_source=context.source,
        )
    return _matched_delivery_evidence(is_text_match, context)


def _clear_delivery_evidence(
    is_text_match: bool | None,
    context: TextFieldContext | None,
) -> TextDeliveryEvidence:
    return _matched_delivery_evidence(is_text_match, context or _unknown_text_field_context())


def _matched_delivery_evidence(
    is_text_match: bool | None,
    context: TextFieldContext,
) -> TextDeliveryEvidence:
    if is_text_match is None:
        return TextDeliveryEvidence(
            status=TextDeliveryStatus.UNAVAILABLE,
            reason=TextDeliveryReason.PLATFORM_EVIDENCE_UNAVAILABLE,
            is_secure=context.is_secure,
            field_type=context.field_type,
            context_source=context.source,
        )
    return TextDeliveryEvidence(
        status=TextDeliveryStatus.MATCHED if is_text_match else TextDeliveryStatus.NOT_LANDED,
        reason=(TextDeliveryReason.VALUE_MATCHED if is_text_match else TextDeliveryReason.VALUE_MISMATCH),
        is_secure=context.is_secure,
        field_type=context.field_type,
        context_source=context.source,
    )


def _unavailable_text_delivery() -> TextDeliveryEvidence:
    return _matched_delivery_evidence(None, _unknown_text_field_context())


def _unknown_text_field_context() -> TextFieldContext:
    return TextFieldContext(
        is_secure=None,
        field_type=TextFieldType.UNKNOWN,
        source=TextFieldContextSource.UNAVAILABLE,
    )
