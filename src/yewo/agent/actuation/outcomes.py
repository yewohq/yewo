"""Classify the observable outcome of one dispatched action."""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yewo.artifacts import (
    ArtifactRetention,
    ArtifactSensitivity,
    RecordIdentifier,
    new_record_identifier,
)
from yewo.devices.application_lifecycle import ApplicationState
from yewo.devices.focus import FocusedFieldReading
from yewo.observations import Observation

from yewo.agent.actuation.records import (
    ActionKind,
    AuthorizedAction,
    DispatchDisposition,
    DispatchReceipt,
)


class ActionOutcomeDisposition(StrEnum):
    """Strength of the available action-outcome evidence."""

    OBSERVED = "observed"
    NO_OBSERVABLE_CHANGE = "no_observable_change"
    AMBIGUOUS = "ambiguous"
    UNAVAILABLE = "unavailable"


class ActionOutcomeSignal(StrEnum):
    """One observed change associated with a dispatched action."""

    SCREEN_CHANGED = "screen_changed"
    CONTEXT_CHANGED = "context_changed"
    KEYBOARD_OPENED = "keyboard_opened"
    KEYBOARD_CLOSED = "keyboard_closed"
    FOCUS_CHANGED = "focus_changed"
    TEXT_LANDED = "text_landed"
    TEXT_NOT_LANDED = "text_not_landed"
    APP_BACKGROUNDED = "app_backgrounded"
    APP_EXITED = "app_exited"


class TextDeliveryStatus(StrEnum):
    """Secret-free result of checking the target field after text entry."""

    MATCHED = "matched"
    NOT_LANDED = "not_landed"
    UNAVAILABLE = "unavailable"


class ActionOutcome(BaseModel):
    """Durable evidence about the immediate result of one action."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    outcome_id: RecordIdentifier
    authorization_id: RecordIdentifier
    receipt_id: RecordIdentifier
    action: ActionKind
    source_observation_id: RecordIdentifier
    settled_observation_id: RecordIdentifier | None
    disposition: ActionOutcomeDisposition
    signals: frozenset[ActionOutcomeSignal]
    reason: str = Field(min_length=1)
    sensitivity: ArtifactSensitivity
    retention: Literal[ArtifactRetention.RUN]

    @model_validator(mode="after")
    def _validate_evidence_shape(self) -> Self:
        if self.disposition is ActionOutcomeDisposition.OBSERVED and not self.signals:
            raise ValueError("An observed outcome must contain an outcome signal.")
        if (
            self.disposition
            in {
                ActionOutcomeDisposition.NO_OBSERVABLE_CHANGE,
                ActionOutcomeDisposition.UNAVAILABLE,
            }
            and self.signals
        ):
            raise ValueError("This outcome disposition cannot contain outcome signals.")
        if (
            self.settled_observation_id is None
            and self.disposition is not ActionOutcomeDisposition.UNAVAILABLE
        ):
            raise ValueError("An outcome without a settled observation must be unavailable.")
        return self


class ActionOutcomeVerifier:
    """Compare trusted before-and-after evidence for one action."""

    def __init__(
        self,
        identifier_factory: Callable[[], str] = new_record_identifier,
    ) -> None:
        self._identifier_factory = identifier_factory

    def verify(
        self,
        action: AuthorizedAction,
        receipt: DispatchReceipt,
        before: Observation,
        after: Observation | None,
        *,
        before_focus: FocusedFieldReading | None = None,
        after_focus: FocusedFieldReading | None = None,
        text_delivery: TextDeliveryStatus | None = None,
    ) -> ActionOutcome:
        self._validate_dispatch_evidence(action, receipt, before)
        self._validate_optional_evidence(action, after, before_focus, after_focus, text_delivery)
        if after is None:
            return self._outcome(
                action,
                receipt,
                before,
                None,
                ActionOutcomeDisposition.UNAVAILABLE,
                frozenset(),
                "No settled observation was available after dispatch.",
            )
        self._validate_observation_pair(before, after)
        signals = self._observed_signals(
            action,
            before,
            after,
            before_focus,
            after_focus,
            text_delivery,
        )
        fingerprint_change = _fingerprint_changed(before, after)
        if fingerprint_change is True:
            signals.add(ActionOutcomeSignal.SCREEN_CHANGED)
        text_unresolved = action.action in {ActionKind.TYPE, ActionKind.CLEAR} and (
            text_delivery is None or text_delivery is TextDeliveryStatus.UNAVAILABLE
        )
        if text_unresolved:
            return self._outcome(
                action,
                receipt,
                before,
                after,
                ActionOutcomeDisposition.AMBIGUOUS,
                frozenset(signals),
                "The settled observation did not prove text delivery to the target field.",
            )
        if signals:
            return self._outcome(
                action,
                receipt,
                before,
                after,
                ActionOutcomeDisposition.OBSERVED,
                frozenset(signals),
                "The settled observation contains action-outcome signals.",
            )
        if fingerprint_change is False:
            return self._outcome(
                action,
                receipt,
                before,
                after,
                ActionOutcomeDisposition.NO_OBSERVABLE_CHANGE,
                frozenset(),
                "Comparable before-and-after evidence did not change.",
            )
        return self._outcome(
            action,
            receipt,
            before,
            after,
            ActionOutcomeDisposition.UNAVAILABLE,
            frozenset(),
            "The screen fingerprints were not comparable.",
        )

    @staticmethod
    def _validate_dispatch_evidence(
        action: AuthorizedAction,
        receipt: DispatchReceipt,
        before: Observation,
    ) -> None:
        if receipt.disposition is not DispatchDisposition.DISPATCHED:
            raise ValueError("Action outcome verification requires a dispatched action.")
        if (
            receipt.authorization_id != action.authorization_id
            or receipt.proposal_id != action.proposal_id
            or receipt.action is not action.action
        ):
            raise ValueError("The dispatch receipt does not match the authorization.")
        if (
            receipt.sensitivity is not action.sensitivity
            or receipt.retention is not action.retention
        ):
            raise ValueError("The dispatch receipt policy does not match the authorization.")
        if (
            before.observation_id != action.source_observation_id
            or before.redacted_screenshot.sha256 != action.source_screenshot_sha256
            or before.application.application_id != action.application_id
        ):
            raise ValueError("The source observation does not match the authorization.")

    @staticmethod
    def _validate_optional_evidence(
        action: AuthorizedAction,
        after: Observation | None,
        before_focus: FocusedFieldReading | None,
        after_focus: FocusedFieldReading | None,
        text_delivery: TextDeliveryStatus | None,
    ) -> None:
        if after is None and any(
            item is not None for item in (before_focus, after_focus, text_delivery)
        ):
            raise ValueError("Post-action evidence requires a settled observation.")
        if text_delivery is not None and action.action not in {ActionKind.TYPE, ActionKind.CLEAR}:
            raise ValueError("Text-delivery evidence requires a text action.")

    @staticmethod
    def _validate_observation_pair(before: Observation, after: Observation) -> None:
        if before.observation_id == after.observation_id:
            raise ValueError("Before and after observations must be different.")
        if after.predecessor_observation_id != before.observation_id:
            raise ValueError("The settled observation does not follow the source observation.")
        if before.device != after.device or before.application != after.application:
            raise ValueError("Before and after observations must describe the same target.")
        if after.captured_at < before.captured_at:
            raise ValueError("The settled observation predates the source observation.")

    @staticmethod
    def _observed_signals(
        action: AuthorizedAction,
        before: Observation,
        after: Observation,
        before_focus: FocusedFieldReading | None,
        after_focus: FocusedFieldReading | None,
        text_delivery: TextDeliveryStatus | None,
    ) -> set[ActionOutcomeSignal]:
        signals = _application_signals(before, after)
        signals.update(_keyboard_signals(before, after))
        if _focus_changed(before_focus, after_focus):
            signals.add(ActionOutcomeSignal.FOCUS_CHANGED)
        text_signal = _text_signal(action.action, text_delivery)
        if text_signal is not None:
            signals.add(text_signal)
        return signals

    def _outcome(
        self,
        action: AuthorizedAction,
        receipt: DispatchReceipt,
        before: Observation,
        after: Observation | None,
        disposition: ActionOutcomeDisposition,
        signals: frozenset[ActionOutcomeSignal],
        reason: str,
    ) -> ActionOutcome:
        return ActionOutcome(
            outcome_id=self._identifier_factory(),
            authorization_id=action.authorization_id,
            receipt_id=receipt.receipt_id,
            action=action.action,
            source_observation_id=before.observation_id,
            settled_observation_id=after.observation_id if after is not None else None,
            disposition=disposition,
            signals=signals,
            reason=reason,
            sensitivity=action.sensitivity,
            retention=action.retention,
        )


def _application_signals(
    before: Observation,
    after: Observation,
) -> set[ActionOutcomeSignal]:
    if after.application_state in {ApplicationState.NOT_INSTALLED, ApplicationState.NOT_RUNNING}:
        return {ActionOutcomeSignal.APP_EXITED}
    signals: set[ActionOutcomeSignal] = set()
    if after.application_state is not ApplicationState.FOREGROUND or (
        after.foreground_application_id is not None
        and after.foreground_application_id != after.application.application_id
    ):
        signals.add(ActionOutcomeSignal.APP_BACKGROUNDED)
    if (
        before.platform_context is not None
        and after.platform_context is not None
        and before.platform_context != after.platform_context
    ):
        signals.add(ActionOutcomeSignal.CONTEXT_CHANGED)
    return signals


def _keyboard_signals(
    before: Observation,
    after: Observation,
) -> set[ActionOutcomeSignal]:
    if before.keyboard_visible is False and after.keyboard_visible is True:
        return {ActionOutcomeSignal.KEYBOARD_OPENED}
    if before.keyboard_visible is True and after.keyboard_visible is False:
        return {ActionOutcomeSignal.KEYBOARD_CLOSED}
    return set()


def _focus_changed(
    before: FocusedFieldReading | None,
    after: FocusedFieldReading | None,
) -> bool:
    if before is None or after is None:
        return False
    if (
        before.is_focused is not None
        and after.is_focused is not None
        and before.is_focused != after.is_focused
    ):
        return True
    return before.driver_bounds != after.driver_bounds


def _text_signal(
    action: ActionKind,
    delivery: TextDeliveryStatus | None,
) -> ActionOutcomeSignal | None:
    if action not in {ActionKind.TYPE, ActionKind.CLEAR} or delivery is None:
        return None
    return {
        TextDeliveryStatus.MATCHED: ActionOutcomeSignal.TEXT_LANDED,
        TextDeliveryStatus.NOT_LANDED: ActionOutcomeSignal.TEXT_NOT_LANDED,
        TextDeliveryStatus.UNAVAILABLE: None,
    }[delivery]


def _fingerprint_changed(before: Observation, after: Observation) -> bool | None:
    before_fingerprint = before.screen_fingerprint
    after_fingerprint = after.screen_fingerprint
    if (
        before_fingerprint.algorithm != after_fingerprint.algorithm
        or before_fingerprint.system_region_mask_sha256
        != after_fingerprint.system_region_mask_sha256
    ):
        return None
    return before_fingerprint.masked_pixels_sha256 != after_fingerprint.masked_pixels_sha256
