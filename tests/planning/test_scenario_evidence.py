"""Generated UI mechanics require journey evidence; objectives do not."""

import pytest
from pydantic import ValidationError

from yewo.testing.models import ScenarioDraft
from yewo.testing.obligations import ObligationActivation
from yewo.testing.taxonomy import ScenarioType


def _draft(step: dict, source: str) -> ScenarioDraft:
    return ScenarioDraft.model_validate(
        {
            "scenario_name": "Create a schedule",
            "steps": [{"kind": "action", "instruction": "Launch the app."}, step],
            "expected_outcomes": ["The schedule contains an exercise."],
        },
        context={"declared": frozenset(), "source_description": source},
    )


def test_an_invented_add_exercise_button_is_refused():
    with pytest.raises(ValidationError, match="Use a verbatim intent"):
        _draft(
            {"kind": "action", "instruction": "Tap the Add Exercise button."},
            "A user adds exercises to reusable schedules.",
        )


@pytest.mark.parametrize("instruction", ["Launch the app", "Launch the app."])
def test_synthetic_launch_accepts_optional_terminal_punctuation(instruction: str):
    draft = ScenarioDraft.model_validate(
        {
            "scenario_name": "Launch",
            "steps": [{"kind": "action", "instruction": instruction}],
            "expected_outcomes": ["The app opens."],
        },
        context={"declared": frozenset(), "source_description": "A user manages schedules."},
    )

    assert draft.steps[0].rendered() == instruction


def test_an_objective_without_assumed_mechanics_is_accepted():
    objective = "A user adds exercises to reusable schedules."
    draft = _draft(
        {
            "kind": "intent",
            "instruction": objective,
            "evidence": objective,
        },
        objective,
    )

    assert draft.steps[1].rendered() == objective


def test_an_outcome_can_name_its_evidence_window_without_inventing_mechanics():
    draft = ScenarioDraft.model_validate(
        {
            "scenario_name": "Create a schedule",
            "steps": [
                {"kind": "action", "instruction": "Launch the app."},
                {
                    "kind": "intent",
                    "instruction": "add exercises",
                    "evidence": "add exercises",
                },
            ],
            "expected_outcomes": [
                {
                    "statement": "The exercise is visible after it is added.",
                    "obligation": {
                        "activation": "during_instruction",
                        "instruction_index": 2,
                    },
                }
            ],
        },
        context={
            "declared": frozenset(),
            "source_description": "A user can add exercises.",
        },
    )

    obligation = draft.expected_outcomes[0].obligation
    assert obligation.activation is ObligationActivation.DURING_INSTRUCTION
    assert obligation.instruction_index == 2


def test_yaml_line_wrapping_does_not_break_verbatim_provenance():
    draft = _draft(
        {
            "kind": "intent",
            "instruction": "add exercises",
            "evidence": "add exercises",
        },
        "A user can add\n      exercises to reusable schedules.",
    )

    assert draft.steps[1].rendered() == "add exercises"


def test_an_invented_control_cannot_be_disguised_as_an_intent():
    with pytest.raises(ValidationError, match="intent evidence is not an exact excerpt"):
        _draft(
            {
                "kind": "intent",
                "instruction": "Tap the Add Exercise button.",
                "evidence": "Tap the Add Exercise button.",
            },
            "A user adds exercises to reusable schedules.",
        )


def test_an_explicitly_documented_control_keeps_its_source_excerpt():
    source = "Tap the Add Exercise button."
    draft = _draft(
        {
            "kind": "action",
            "instruction": "Tap the Add Exercise button.",
            "evidence": "Tap the Add Exercise button.",
        },
        source,
    )

    assert draft.steps[1].evidence == "Tap the Add Exercise button."


def test_a_description_cannot_be_paraphrased_into_an_executable_action():
    with pytest.raises(ValidationError, match="not copied verbatim"):
        _draft(
            {
                "kind": "action",
                "instruction": "Tap the Add Exercise button.",
                "evidence": "A user adds exercises to reusable schedules.",
            },
            "A user adds exercises to reusable schedules.",
        )


def test_an_invented_control_without_a_control_noun_is_still_refused():
    with pytest.raises(ValidationError, match="provides no verbatim source"):
        _draft(
            {"kind": "action", "instruction": "Tap Add Exercise."},
            "A user adds exercises to reusable schedules.",
        )


@pytest.mark.parametrize("label", ["保存", "حفظ"])
def test_localized_control_labels_need_no_language_specific_vocabulary(label: str):
    instruction = f"Tap {label}."

    draft = _draft(
        {"kind": "action", "instruction": instruction, "evidence": instruction},
        instruction,
    )

    assert draft.steps[1].rendered() == instruction


@pytest.mark.parametrize(
    "objective",
    ["ユーザーは再利用可能な予定を作成する。", "ينشئ المستخدم جدولاً قابلاً لإعادة الاستخدام."],
)
def test_localized_intents_keep_verbatim_provenance(objective: str):
    draft = _draft(
        {"kind": "intent", "instruction": objective, "evidence": objective},
        objective,
    )

    assert draft.steps[1].rendered() == objective


@pytest.mark.parametrize(
    "objective",
    ["ユーザーは予定を作成する。", "ينشئ المستخدم جدولاً."],
)
def test_localized_variations_preserve_the_complete_journey(objective: str):
    draft = ScenarioDraft.model_validate(
        {
            "scenario_name": "Boundary user",
            "steps": [
                {"kind": "action", "instruction": "Launch the app."},
                {
                    "kind": "variation",
                    "goal": objective,
                    "objective": objective,
                    "evidence": objective,
                    "strategy": "boundary",
                    "repetitions": 100,
                },
            ],
            "expected_outcomes": ["The attempt has a coherent result."],
        },
        context={
            "declared": frozenset(),
            "source_description": objective,
            "scenario_type": ScenarioType.BOUNDARY,
        },
    )

    assert objective in draft.steps[1].rendered()


def test_a_variation_cannot_replace_the_complete_journey_with_one_step():
    source = "A user creates a schedule. They add exercises and save it."

    with pytest.raises(ValidationError, match="complete journey verbatim"):
        ScenarioDraft.model_validate(
            {
                "scenario_name": "Boundary user",
                "steps": [
                    {"kind": "action", "instruction": "Launch the app."},
                    {
                        "kind": "variation",
                        "goal": "add exercises",
                        "objective": "add exercises",
                        "evidence": "add exercises",
                        "strategy": "boundary",
                        "repetitions": 100,
                    },
                ],
                "expected_outcomes": ["The attempt has a coherent result."],
            },
            context={
                "declared": frozenset(),
                "source_description": source,
                "scenario_type": ScenarioType.BOUNDARY,
            },
        )


def test_a_declared_refusal_still_preserves_the_complete_journey_strategy():
    source = "A user signs in."
    draft = ScenarioDraft.model_validate(
        {
            "scenario_name": "Rejected password user",
            "steps": [
                {"kind": "action", "instruction": "Launch the app."},
                {"kind": "enter", "field": "Password", "value": "rejected"},
                {
                    "kind": "variation",
                    "goal": source,
                    "objective": source,
                    "evidence": source,
                    "strategy": "negative",
                },
            ],
            "expected_outcomes": ["The attempt has a coherent result."],
        },
        context={
            "declared": frozenset({"rejected"}),
            "refused": frozenset({"rejected"}),
            "source_description": source,
            "scenario_type": ScenarioType.NEGATIVE,
        },
    )

    assert "declared refused value" in draft.steps[2].rendered()


def test_a_declared_refusal_cannot_skip_the_complete_journey_strategy():
    with pytest.raises(ValidationError, match="exactly one 'variation' step"):
        ScenarioDraft.model_validate(
            {
                "scenario_name": "Rejected password user",
                "steps": [
                    {"kind": "action", "instruction": "Launch the app."},
                    {"kind": "enter", "field": "Password", "value": "rejected"},
                ],
                "expected_outcomes": ["The password is refused."],
            },
            context={
                "declared": frozenset({"rejected"}),
                "refused": frozenset({"rejected"}),
                "source_description": "A user signs in.",
                "scenario_type": ScenarioType.NEGATIVE,
            },
        )
