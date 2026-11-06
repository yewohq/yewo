"""Actuation record schema and serialization tests."""

import pytest
from pydantic import TypeAdapter, ValidationError

from yewo.agent.actuation import (
    ACTION_DEFINITIONS,
    ActionHandling,
    ActionKind,
    ActionProposal,
    ActionSource,
    AuthorizedAction,
    DeclaredValueReference,
    DispatchDisposition,
    DispatchReceipt,
    LiteralText,
    TextInput,
)
from yewo.artifacts import ArtifactRetention, ArtifactSensitivity
from agent.actuation.support import proposal


def test_proposal_round_trip_preserves_every_field() -> None:
    original = proposal(ActionKind.TYPE)

    restored = ActionProposal.model_validate_json(original.model_dump_json())

    assert restored == original


@pytest.mark.parametrize(
    "text",
    [
        LiteralText(value="hello"),
        DeclaredValueReference(value_name="declared_secret"),
    ],
)
def test_text_input_round_trip_preserves_its_variant(text: object) -> None:
    adapter = TypeAdapter(TextInput)
    encoded = adapter.dump_json(text)

    assert adapter.validate_json(encoded) == text


def test_authorization_round_trip_contains_only_a_declared_value_reference() -> None:
    secret = "correct horse battery staple"
    original = AuthorizedAction(
        authorization_id="authorization-1",
        proposal_id="proposal-1",
        source_observation_id="obs_1",
        source_screenshot_sha256="a" * 64,
        application_id="com.example.app",
        action=ActionKind.TYPE,
        source=ActionSource.PLANNER,
        side_effect=ACTION_DEFINITIONS[ActionKind.TYPE].side_effect,
        expected_outcome="The password field contains the declared value.",
        text=DeclaredValueReference(value_name="declared_secret"),
        sensitivity=ArtifactSensitivity.DECLARED_SENSITIVE,
        retention=ArtifactRetention.RUN,
    )

    encoded = original.model_dump_json()
    restored = AuthorizedAction.model_validate_json(encoded)

    assert restored == original
    assert secret not in encoded


def test_every_action_has_one_definition() -> None:
    assert set(ACTION_DEFINITIONS) == set(ActionKind)


def test_action_definitions_classify_runtime_handling() -> None:
    assert ACTION_DEFINITIONS[ActionKind.TAP].handling is ActionHandling.EXECUTE
    assert ACTION_DEFINITIONS[ActionKind.DONE].handling is ActionHandling.TERMINAL
    assert (
        ACTION_DEFINITIONS[ActionKind.SET_DEVICE_CAPABILITY].handling is ActionHandling.CAPABILITY
    )


def test_dispatch_receipt_round_trip_has_no_outcome_fields() -> None:
    original = DispatchReceipt(
        receipt_id="receipt-1",
        authorization_id="authorization-1",
        proposal_id="proposal-1",
        action=ActionKind.TYPE,
        disposition=DispatchDisposition.DISPATCHED,
        sensitivity=ArtifactSensitivity.DECLARED_SENSITIVE,
        retention=ArtifactRetention.RUN,
    )

    encoded = original.model_dump_json()

    assert DispatchReceipt.model_validate_json(encoded) == original
    assert "reported_text" not in encoded
    assert "keyboard_dismissed" not in encoded


def test_planner_proposal_cannot_request_repeated_taps() -> None:
    with pytest.raises(ValidationError):
        ActionProposal.model_validate(
            {
                **proposal().model_dump(),
                "tap_count": 4,
            }
        )


def test_interaction_refinement_requires_one_point_target() -> None:
    with pytest.raises(ValidationError, match="requires one visual point target"):
        proposal(
            ActionKind.BACK,
            interaction_refinement_for_proposal_id="proposal-original",
        )


def test_proposal_cannot_mix_grounding_correction_and_interaction_refinement() -> None:
    with pytest.raises(ValidationError, match="cannot be both"):
        proposal(
            grounding_correction_for_proposal_id="proposal-grounding",
            interaction_refinement_for_proposal_id="proposal-interaction",
        )


def test_planner_validation_context_rejects_an_unavailable_action() -> None:
    candidate = proposal(ActionKind.TAP)

    with pytest.raises(ValidationError, match="not currently available"):
        ActionProposal.model_validate(
            candidate.model_dump(),
            context={"available_actions": (ActionKind.DISMISS_KEYBOARD.value,)},
        )
