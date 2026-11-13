"""Which evidence each proposal-verifier refusal basis requires."""

import pytest
from pydantic import ValidationError

from yewo.agent.actuation.proposal_verifier import (
    ActionReadinessAssessment,
    ActionReadinessDisposition,
    ActionReadinessRefusalBasis,
    ModelProposalVerifier,
)
from yewo.agent.actuation.proposal_verifier_prompt import PROPOSAL_VERIFIER_SYSTEM_PROMPT
from yewo.agent.actuation import (
    ActionRefusalCode,
)

from agent.actuation.proposal_verifier_support import discovery_request, save_proposal
from agent.planning.support import StubPlannerModel


def _refusal_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "verification_id": "verification_1",
        "proposal_id": "proposal_1",
        "disposition": "refuse",
        "reason": "The reason names Save while the target identifies Cancel.",
        "unresolved_controls": [],
        "completed_work_items": [],
        "proposal_work_item": None,
        "restarts_completed_work_item": None,
    }
    payload.update(overrides)
    return payload


@pytest.mark.parametrize(
    ("refusal_basis", "code"),
    [
        ("incoherent_proposal", ActionRefusalCode.DISCOVERY_INCOMPLETE),
        ("reopened_stage", ActionRefusalCode.COMPLETED_WORK_REGRESSION),
    ],
)
def test_proposal_verifier_accepts_a_basis_that_needs_no_control_evidence(
    refusal_basis: str,
    code: ActionRefusalCode,
) -> None:
    # The prompt mandates both bases with no unresolved control and no restarted work item.
    # verify() required one anyway, so both arrived as verification_unavailable, which is a
    # transient code: no screen fact was recorded and the planner re-proposed the same action.
    model = StubPlannerModel(
        {
            "disposition": "refuse",
            "reason": "The reason names Save while the target identifies Cancel.",
            "refusal_basis": refusal_basis,
            "unresolved_controls": [],
            "completed_work_items": [],
            "proposal_work_item": "Cover every relevant visible field.",
            "restarts_completed_work_item": None,
        }
    )
    identifiers = iter(("verification_1", "request_1"))

    assessment = ModelProposalVerifier(
        model,
        "proposal-verifier-primary",
        identifier_factory=lambda: next(identifiers),
    ).verify(discovery_request(), save_proposal())

    assert assessment.disposition is ActionReadinessDisposition.REFUSE
    refusal = assessment.refusal()
    assert refusal is not None
    assert refusal.code is code


def test_proposal_verifier_rejects_a_basis_without_the_evidence_it_requires() -> None:
    with pytest.raises(ValidationError, match="must name one unresolved control"):
        ActionReadinessAssessment.model_validate(
            _refusal_payload(refusal_basis="unresolved_control")
        )


def test_proposal_verifier_rejects_evidence_its_basis_does_not_support() -> None:
    with pytest.raises(ValidationError, match="cannot carry unresolved evidence"):
        ActionReadinessAssessment.model_validate(
            _refusal_payload(
                refusal_basis="incoherent_proposal",
                restarts_completed_work_item="Create the record.",
            )
        )


def test_proposal_verifier_rejects_a_refusal_basis_on_an_allowed_action() -> None:
    with pytest.raises(ValidationError, match="Only a refusal"):
        ActionReadinessAssessment.model_validate(
            _refusal_payload(disposition="allow", refusal_basis="incoherent_proposal")
        )


def test_proposal_verifier_still_requires_evidence_when_no_basis_is_named() -> None:
    with pytest.raises(ValidationError, match="must identify unresolved evidence"):
        ActionReadinessAssessment.model_validate(_refusal_payload())


def test_verifier_prompt_names_every_refusal_basis() -> None:
    for basis in ActionReadinessRefusalBasis:
        assert basis.value in PROPOSAL_VERIFIER_SYSTEM_PROMPT
