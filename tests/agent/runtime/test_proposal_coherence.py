"""Deterministic coherence checks for model-authored mutation proposals."""

from observations.factories import build_observation

from yewo.agent.actuation import ActionKind, ActionProposal, LiteralText
from yewo.agent.runtime.proposal_coherence import proposal_coherence_refusal
from yewo.agent.screen_analysis import (
    CurrentScreenAnalysis,
    ScreenElement,
    ScreenElementConfidence,
    ScreenElementKind,
    ScreenSemanticRole,
    ScreenVisualKind,
)


def _analysis() -> CurrentScreenAnalysis:
    observation = build_observation()
    return CurrentScreenAnalysis(
        analysis_id="analysis-1",
        observation_id=observation.observation_id,
        screenshot_sha256=observation.screenshot.sha256,
        presentation=observation.presentation_key,
        screen_summary="Visible form with two fields.",
        elements=(
            _field("first-field", "First value", "10"),
            _field("second-field", "Second value", "20"),
        ),
    )


def _field(reference: str, name: str, state: str) -> ScreenElement:
    return ScreenElement(
        element_ref=reference,
        name=name,
        kind=ScreenElementKind(
            visual=ScreenVisualKind.INPUT_SURFACE,
            semantic=ScreenSemanticRole.TEXT_FIELD,
        ),
        state=state,
        confidence=ScreenElementConfidence.HIGH,
        visual_cues=("under its visible label",),
    )


def _type(**updates: object) -> ActionProposal:
    values: dict[str, object] = {
        "proposal_id": "proposal-1",
        "source_observation_id": "observation-1",
        "action": ActionKind.TYPE,
        "target_element_ref": "first-field",
        "text": LiteralText(value="12"),
        "expected_outcome": "First value contains the entered text.",
        "reason": "Update First value.",
    }
    values.update(updates)
    return ActionProposal.model_validate(values)


def test_model_authored_prose_does_not_create_a_deterministic_refusal() -> None:
    refusal = proposal_coherence_refusal(
        _type(
            expected_outcome="Second value contains the entered text.",
            reason="Update Second value.",
        ),
        _analysis(),
    )

    assert refusal is None


def test_type_uses_structured_text_instead_of_interpreting_prediction_prose() -> None:
    refusal = proposal_coherence_refusal(
        _type(expected_outcome="First value gains focus and the keyboard opens."),
        _analysis(),
    )

    assert refusal is None


def test_type_refuses_a_value_that_the_target_already_contains() -> None:
    refusal = proposal_coherence_refusal(
        _type(text=LiteralText(value="10")),
        _analysis(),
    )

    assert refusal is not None
    assert "already contains" in refusal.reason


def test_coherent_type_remains_dispatchable() -> None:
    assert proposal_coherence_refusal(_type(), _analysis()) is None


def test_focus_can_describe_later_value_entry_without_claiming_mutation() -> None:
    candidate = ActionProposal(
        proposal_id="proposal-focus",
        source_observation_id="observation-1",
        action=ActionKind.FOCUS,
        target_element_ref="first-field",
        expected_outcome="The field is focused so the value can be entered.",
        reason="Prepare the field for entry.",
    )

    assert proposal_coherence_refusal(candidate, _analysis()) is None
