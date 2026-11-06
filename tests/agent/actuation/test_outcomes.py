"""Deterministic action-outcome classification tests."""

from datetime import timedelta
from hashlib import sha256

import pytest

from yewo.agent.actuation import (
    ACTION_DEFINITIONS,
    ActionKind,
    ActionOutcome,
    ActionOutcomeDisposition,
    ActionOutcomeSignal,
    ActionOutcomeVerifier,
    ActionSource,
    AuthorizedAction,
    DeclaredValueReference,
    DispatchDisposition,
    DispatchReceipt,
    LiteralText,
    TextDeliveryEvidence,
    TextDeliveryReason,
    TextDeliveryStatus,
)
from yewo.artifacts import ArtifactRetention, ArtifactSensitivity
from yewo.devices.application_lifecycle import ApplicationState
from yewo.devices.focus import TextFieldContextSource, TextFieldType, FocusedFieldReading
from yewo.devices.screen import SystemRegionKind
from yewo.grounding import PixelPoint
from yewo.observations import Observation, ScreenFingerprint, ScreenRegion

from observations.factories import build_observation, build_screenshot


def _authorization(
    action: ActionKind = ActionKind.TAP,
    *,
    declared_text: bool = False,
) -> AuthorizedAction:
    values: dict[str, object] = {
        "authorization_id": "authorization-1",
        "proposal_id": "proposal-1",
        "source_observation_id": "obs_1",
        "source_screenshot_sha256": "a" * 64,
        "application_id": "com.example.app",
        "action": action,
        "source": ActionSource.PLANNER,
        "side_effect": ACTION_DEFINITIONS[action].side_effect,
        "expected_outcome": "The expected visible change occurs.",
        "sensitivity": ArtifactSensitivity.DEVICE_CONTENT,
        "retention": ArtifactRetention.RUN,
    }
    if action in {ActionKind.TAP, ActionKind.FOCUS}:
        values["target"] = PixelPoint(x=50, y=100)
    if action is ActionKind.TYPE:
        values["text"] = (
            DeclaredValueReference(value_name="declared_secret")
            if declared_text
            else LiteralText(value="hello")
        )
        if declared_text:
            values["sensitivity"] = ArtifactSensitivity.DECLARED_SENSITIVE
    return AuthorizedAction.model_validate(values)


def _receipt(
    action: AuthorizedAction,
    *,
    disposition: DispatchDisposition = DispatchDisposition.DISPATCHED,
) -> DispatchReceipt:
    return DispatchReceipt(
        receipt_id="receipt-1",
        authorization_id=action.authorization_id,
        proposal_id=action.proposal_id,
        action=action.action,
        disposition=disposition,
        reason="The device rejected the action."
        if disposition is not DispatchDisposition.DISPATCHED
        else None,
        sensitivity=action.sensitivity,
        retention=action.retention,
    )


def _delivery(status: TextDeliveryStatus) -> TextDeliveryEvidence:
    reasons = {
        TextDeliveryStatus.MATCHED: TextDeliveryReason.VALUE_MATCHED,
        TextDeliveryStatus.NOT_LANDED: TextDeliveryReason.VALUE_MISMATCH,
        TextDeliveryStatus.UNVERIFIABLE: TextDeliveryReason.SECURE_FIELD,
        TextDeliveryStatus.UNAVAILABLE: TextDeliveryReason.PLATFORM_EVIDENCE_UNAVAILABLE,
    }
    return TextDeliveryEvidence(
        status=status,
        reason=reasons[status],
        is_secure=status is TextDeliveryStatus.UNVERIFIABLE,
        field_type=(
            TextFieldType.PASSWORD
            if status is TextDeliveryStatus.UNVERIFIABLE
            else TextFieldType.TEXT
        ),
        context_source=TextFieldContextSource.PLATFORM_METADATA,
    )


def _settled_observation(
    **updates: object,
) -> Observation:
    fingerprint = updates.pop("fingerprint", "0123456789abcdef")
    if not isinstance(fingerprint, str):
        raise TypeError("The test fingerprint must be a string.")
    masked_pixels_sha256 = updates.pop(
        "masked_pixels_sha256",
        sha256(fingerprint.encode()).hexdigest(),
    )
    if not isinstance(masked_pixels_sha256, str):
        raise TypeError("The test masked-pixel digest must be a string.")
    source = build_observation()
    screenshot = build_screenshot(artifact_id="shot_2")
    values: dict[str, object] = {
        "observation_id": "obs_2",
        "predecessor_observation_id": source.observation_id,
        "captured_at": source.captured_at + timedelta(seconds=1),
        "redacted_screenshot": screenshot,
        "screen_fingerprint": ScreenFingerprint(
            value=fingerprint,
            masked_pixels_sha256=masked_pixels_sha256,
            system_region_mask_sha256=None,
            source_artifact_id=screenshot.artifact_id,
        ),
    }
    values.update(updates)
    return build_observation(**values)


def test_records_multiple_observed_signals_for_one_text_action() -> None:
    action = _authorization(ActionKind.TYPE)
    outcome = ActionOutcomeVerifier(lambda: "outcome-1").verify(
        action,
        _receipt(action),
        build_observation(),
        _settled_observation(
            fingerprint="1123456789abcdef",
            keyboard_visible=True,
        ),
        before_focus=FocusedFieldReading(
            is_focused=False,
            driver_bounds=None,
            keyboard_visible=False,
        ),
        after_focus=FocusedFieldReading(
            is_focused=True,
            driver_bounds=(10, 20, 80, 60),
            keyboard_visible=True,
        ),
        text_delivery=_delivery(TextDeliveryStatus.MATCHED),
    )

    assert outcome.disposition is ActionOutcomeDisposition.OBSERVED
    assert outcome.signals == {
        ActionOutcomeSignal.PIXELS_CHANGED,
        ActionOutcomeSignal.KEYBOARD_OPENED,
        ActionOutcomeSignal.FOCUS_CHANGED,
        ActionOutcomeSignal.TEXT_LANDED,
    }
    assert outcome.outcome_id == "outcome-1"


def test_exact_pixel_digest_detects_change_when_phash_is_unchanged() -> None:
    action = _authorization(ActionKind.TAP)
    outcome = ActionOutcomeVerifier().verify(
        action,
        _receipt(action),
        build_observation(),
        _settled_observation(masked_pixels_sha256="f" * 64),
    )

    assert outcome.disposition is ActionOutcomeDisposition.OBSERVED
    assert outcome.signals == {ActionOutcomeSignal.PIXELS_CHANGED}


def test_legacy_screen_changed_signal_parses_as_pixels_changed() -> None:
    assert ActionOutcomeSignal("screen_changed") is ActionOutcomeSignal.PIXELS_CHANGED


def test_permission_controller_is_a_system_surface_not_app_backgrounding() -> None:
    action = _authorization(ActionKind.TAP)
    outcome = ActionOutcomeVerifier().verify(
        action,
        _receipt(action),
        build_observation(),
        _settled_observation(
            application_state=ApplicationState.BACKGROUND,
            foreground_application_id="com.google.android.permissioncontroller",
        ),
    )

    assert ActionOutcomeSignal.SYSTEM_SURFACE_OPENED in outcome.signals
    assert ActionOutcomeSignal.APP_BACKGROUNDED not in outcome.signals


@pytest.mark.parametrize(
    ("delivery", "expected_signal"),
    [
        (TextDeliveryStatus.NOT_LANDED, ActionOutcomeSignal.TEXT_NOT_LANDED),
    ],
)
def test_distinguishes_failed_text_delivery(
    delivery: TextDeliveryStatus,
    expected_signal: ActionOutcomeSignal,
) -> None:
    action = _authorization(ActionKind.TYPE)

    outcome = ActionOutcomeVerifier().verify(
        action,
        _receipt(action),
        build_observation(),
        _settled_observation(),
        text_delivery=_delivery(delivery),
    )

    assert outcome.disposition is ActionOutcomeDisposition.OBSERVED
    assert outcome.signals == {expected_signal}


@pytest.mark.parametrize("delivery", [None, _delivery(TextDeliveryStatus.UNAVAILABLE)])
def test_unresolved_text_delivery_is_ambiguous_even_when_the_screen_changes(
    delivery: TextDeliveryEvidence | None,
) -> None:
    action = _authorization(ActionKind.TYPE)

    outcome = ActionOutcomeVerifier().verify(
        action,
        _receipt(action),
        build_observation(),
        _settled_observation(fingerprint="1123456789abcdef"),
        text_delivery=delivery,
    )

    assert outcome.disposition is ActionOutcomeDisposition.AMBIGUOUS
    assert outcome.signals == {ActionOutcomeSignal.PIXELS_CHANGED}


@pytest.mark.parametrize(
    ("updates", "expected_signal"),
    [
        (
            {"platform_context": "com.example.DetailActivity"},
            ActionOutcomeSignal.CONTEXT_CHANGED,
        ),
        (
            {
                "application_state": ApplicationState.BACKGROUND,
                "foreground_application_id": "com.example.other",
            },
            ActionOutcomeSignal.APP_BACKGROUNDED,
        ),
        (
            {
                "application_state": ApplicationState.NOT_RUNNING,
                "foreground_application_id": None,
            },
            ActionOutcomeSignal.APP_EXITED,
        ),
    ],
)
def test_classifies_application_changes(
    updates: dict[str, object],
    expected_signal: ActionOutcomeSignal,
) -> None:
    action = _authorization()

    outcome = ActionOutcomeVerifier().verify(
        action,
        _receipt(action),
        build_observation(),
        _settled_observation(**updates),
    )

    assert outcome.disposition is ActionOutcomeDisposition.OBSERVED
    assert expected_signal in outcome.signals


def test_identical_comparable_evidence_reports_no_observable_change() -> None:
    action = _authorization()

    outcome = ActionOutcomeVerifier().verify(
        action,
        _receipt(action),
        build_observation(),
        _settled_observation(),
    )

    assert outcome.disposition is ActionOutcomeDisposition.NO_OBSERVABLE_CHANGE
    assert not outcome.signals


def test_records_keyboard_closing() -> None:
    action = _authorization()

    outcome = ActionOutcomeVerifier().verify(
        action,
        _receipt(action),
        build_observation(keyboard_visible=True),
        _settled_observation(keyboard_visible=False),
    )

    assert ActionOutcomeSignal.KEYBOARD_CLOSED in outcome.signals


def test_missing_settled_observation_reports_unavailable() -> None:
    action = _authorization()

    outcome = ActionOutcomeVerifier().verify(
        action,
        _receipt(action),
        build_observation(),
        None,
    )

    assert outcome.disposition is ActionOutcomeDisposition.UNAVAILABLE
    assert outcome.settled_observation_id is None


def test_incompatible_fingerprints_report_unavailable() -> None:
    action = _authorization()
    region = ScreenRegion(
        left=0,
        top=0,
        right=200,
        bottom=20,
        kind=SystemRegionKind.STATUS_BAR,
    )

    outcome = ActionOutcomeVerifier().verify(
        action,
        _receipt(action),
        build_observation(),
        build_observation(
            observation_id="obs_2",
            predecessor_observation_id="obs_1",
            captured_at=build_observation().captured_at + timedelta(seconds=1),
            system_regions=(region,),
        ),
    )

    assert outcome.disposition is ActionOutcomeDisposition.UNAVAILABLE


def test_refuses_a_receipt_that_was_not_dispatched() -> None:
    action = _authorization()

    with pytest.raises(ValueError, match="requires an attempted device dispatch"):
        ActionOutcomeVerifier().verify(
            action,
            _receipt(action, disposition=DispatchDisposition.DEVICE_REFUSED),
            build_observation(),
            _settled_observation(),
        )


def test_unverified_dispatch_uses_fresh_evidence_without_claiming_success() -> None:
    action = _authorization()
    receipt = _receipt(
        action,
        disposition=DispatchDisposition.DISPATCH_UNVERIFIED,
    ).model_copy(update={"reason": "Focus verification timed out."})

    outcome = ActionOutcomeVerifier().verify(
        action,
        receipt,
        build_observation(),
        _settled_observation(),
    )

    assert outcome.disposition is ActionOutcomeDisposition.AMBIGUOUS
    assert outcome.settled_observation_id == "obs_2"
    assert outcome.signals == frozenset()


def test_refuses_mismatched_evidence() -> None:
    action = _authorization()
    wrong_receipt = _receipt(action).model_copy(update={"authorization_id": "other"})

    with pytest.raises(ValueError, match="does not match"):
        ActionOutcomeVerifier().verify(
            action,
            wrong_receipt,
            build_observation(),
            _settled_observation(),
        )


def test_refuses_an_unlinked_settled_observation() -> None:
    action = _authorization()

    with pytest.raises(ValueError, match="does not follow"):
        ActionOutcomeVerifier().verify(
            action,
            _receipt(action),
            build_observation(),
            _settled_observation(predecessor_observation_id="unrelated"),
        )


def test_refuses_text_evidence_for_a_non_text_action() -> None:
    action = _authorization()

    with pytest.raises(ValueError, match="requires a text action"):
        ActionOutcomeVerifier().verify(
            action,
            _receipt(action),
            build_observation(),
            _settled_observation(),
            text_delivery=_delivery(TextDeliveryStatus.MATCHED),
        )


def test_outcome_serialization_never_contains_a_declared_value() -> None:
    secret = "correct horse battery staple"
    action = _authorization(ActionKind.TYPE, declared_text=True)

    outcome = ActionOutcomeVerifier().verify(
        action,
        _receipt(action),
        build_observation(),
        _settled_observation(),
        text_delivery=_delivery(TextDeliveryStatus.MATCHED),
    )
    encoded = outcome.model_dump_json()

    assert ActionOutcome.model_validate_json(encoded) == outcome
    assert secret not in encoded
    assert "declared_secret" not in encoded
