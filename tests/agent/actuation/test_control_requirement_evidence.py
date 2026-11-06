"""Control-specific requirement-evidence validation."""

import pytest

from yewo.agent.actuation.verifier import (
    ActionReadinessAssessment,
    evidence_identifies_control,
)


@pytest.mark.parametrize(
    ("control_name", "evidence"),
    (
        ("Quantity", "Complete Quantity before saving."),
        ("Primary field", "primary_field"),
        ("姓名", "填写姓名字段"),
    ),
)
def test_evidence_identifies_exact_control_in_any_language(
    control_name: str,
    evidence: str,
) -> None:
    assert evidence_identifies_control(control_name, evidence)


def test_evidence_does_not_match_inside_an_unrelated_word() -> None:
    assert not evidence_identifies_control("must", "Choose mustard")


def test_assessment_accepts_control_evidence() -> None:
    control_name = "Quantity"
    control_reason = "The field is empty."
    assessment = ActionReadinessAssessment.model_validate(
        {
            "verification_id": "verification_1",
            "proposal_id": "proposal_1",
            "disposition": "refuse",
            "reason": "The workflow is incomplete.",
            "unresolved_controls": [
                {
                    "control_name": control_name,
                    "required_by": "objective_or_open_work",
                    "requirement_evidence": "Complete Quantity before saving.",
                    "reason": control_reason,
                }
            ],
            "completed_work_items": [],
            "proposal_work_item": None,
            "restarts_completed_work_item": None,
        },
        context={
            "interactive_control_names": ("Quantity",),
            "required_control_evidence": {
                "Quantity": {
                    "objective_or_open_work": ("Complete Quantity before saving.",)
                }
            },
        },
    )

    refusal = assessment.refusal()
    assert refusal is not None
    assert control_name in refusal.reason
    assert control_reason in refusal.reason
