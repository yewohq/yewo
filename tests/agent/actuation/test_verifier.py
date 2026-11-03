"""Independent pre-dispatch discovery action verification."""

import json

import pytest
from pydantic import ValidationError

from yewo.agent.actuation.verifier import (
    ACTION_VERIFIER_MAX_COMPLETION_TOKENS,
    ActionReadinessAssessment,
    ActionReadinessDisposition,
    ModelActionVerifier,
)
from yewo.agent.actuation import (
    ActionKind,
    ActionOutcome,
    ActionOutcomeDisposition,
    ActionOutcomeSignal,
    ActionProposal,
    ActionRefusalCode,
)
from yewo.agent.planning.planner import (
    PlanningRequest,
    ProgressRouteAttempt,
    TaskProgressContext,
)
from yewo.agent.planning.history import PlanningHistoryEntry, planning_screen_reference
from yewo.artifacts import ArtifactRetention, ArtifactSensitivity
from yewo.models import MINIMUM_COMPLETION_TOKENS, ModelRole, TextModelContent

from agent.discovery.support import observation
from agent.planning.support import StubPlannerModel


def _proposal() -> ActionProposal:
    return ActionProposal(
        proposal_id="proposal_1",
        source_observation_id="obs_1",
        action=ActionKind.TAP,
        target_expression="the visible Save button",
        expected_outcome="The editor saves and closes.",
        reason="Save the current editor.",
    )


def _request() -> PlanningRequest:
    source = observation("dropdown_closed", "1111111111111111")
    destination = observation("dropdown_open", "2222222222222222")
    dropdown_proposal = ActionProposal(
        proposal_id="dropdown_proposal",
        source_observation_id=source.observation_id,
        action=ActionKind.TAP,
        target_expression="the Trainings Type dropdown",
        expected_outcome="The available training types become visible.",
        reason="Inspect the dropdown choices.",
    )
    dropdown_outcome = ActionOutcome(
        outcome_id="dropdown_outcome",
        authorization_id="dropdown_authorization",
        receipt_id="dropdown_receipt",
        action=ActionKind.TAP,
        source_observation_id=source.observation_id,
        settled_observation_id=destination.observation_id,
        disposition=ActionOutcomeDisposition.OBSERVED,
        signals=frozenset({ActionOutcomeSignal.PIXELS_CHANGED}),
        reason="The dropdown opened.",
        sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
        retention=ArtifactRetention.RUN,
    )
    return PlanningRequest(
        observation=observation("obs_1", "1111111111111111"),
        objective="Discover how to create a schedule.",
        open_work=("Exercise every relevant visible field.",),
        completion_conditions=("The saved schedule can be reopened.",),
        available_actions=frozenset({ActionKind.TAP}),
        history=(
            PlanningHistoryEntry(
                proposal=dropdown_proposal,
                source_screen=planning_screen_reference(source),
                destination_screen=planning_screen_reference(destination),
                outcome=dropdown_outcome,
            ),
        ),
    )


def test_action_verifier_uses_dedicated_profile_and_refuses_unresolved_work() -> None:
    model = StubPlannerModel(
        {
            "verification_id": "verification_1",
            "proposal_id": "proposal_1",
            "disposition": "refuse",
            "reason": "The action would leave visible discovery work unresolved.",
            "unresolved_controls": ["The visible Heart field has not been exercised."],
            "visible_controls": [
                {
                    "name": "Heart",
                    "kind": "text_field",
                    "state": "empty",
                    "matches_proposal_target": True,
                }
            ],
            "completed_work_items": [],
            "proposal_work_item": "Exercise every relevant visible field.",
            "restarts_completed_work_item": None,
        }
    )
    identifiers = iter(("verification_1", "request_1"))
    verifier = ModelActionVerifier(
        model,
        "action-verifier-primary",
        identifier_factory=lambda: next(identifiers),
    )

    assessment = verifier.verify(_request(), _proposal())

    assert assessment.disposition is ActionReadinessDisposition.REFUSE
    assert assessment.refusal() is not None
    assert assessment.visible_controls[0].name == "Heart"
    assert assessment.visible_controls[0].matches_proposal_target is True
    assert model.requests[0].role is ModelRole.ACTION_VERIFIER
    assert model.requests[0].profile_id == "action-verifier-primary"
    assert model.requests[0].max_completion_tokens == ACTION_VERIFIER_MAX_COMPLETION_TOKENS
    assert model.validation_contexts[0] == {
        "allowed_work_items": ("Exercise every relevant visible field.",),
        "completed_work_items": (),
    }
    assert ACTION_VERIFIER_MAX_COMPLETION_TOKENS == MINIMUM_COMPLETION_TOKENS
    user_content = model.requests[0].messages[1].content[0]
    assert isinstance(user_content, TextModelContent)
    context = json.loads(user_content.text.split("Context JSON follows:\n", 1)[1])
    assert context["recent_history"][0]["target_expression"] == ("the Trainings Type dropdown")
    assert context["recent_history"][0]["feedback"]["disposition"] == "observed"


def test_action_verifier_fails_closed_on_invalid_allow_response() -> None:
    model = StubPlannerModel(
        {
            "verification_id": "verification_1",
            "proposal_id": "proposal_1",
            "disposition": "allow",
            "reason": "The action is safe.",
            "unresolved_controls": ["A field is unresolved."],
            "completed_work_items": [],
            "proposal_work_item": "Exercise every relevant visible field.",
            "restarts_completed_work_item": None,
        }
    )
    identifiers = iter(("verification_1", "request_1"))
    verifier = ModelActionVerifier(
        model,
        "action-verifier-primary",
        identifier_factory=lambda: next(identifiers),
    )

    assessment = verifier.verify(_request(), _proposal())

    assert assessment.disposition is ActionReadinessDisposition.UNAVAILABLE
    assert assessment.unresolved_controls == ()
    refusal = assessment.refusal()
    assert refusal is not None
    assert refusal.code is ActionRefusalCode.VERIFICATION_UNAVAILABLE


def test_action_verifier_refuses_a_route_that_restarts_completed_work() -> None:
    create = "Create and save one reusable training schedule."
    reopen = "Return to the saved schedule and establish that it can be repeated or updated."
    request = _request().model_copy(
        update={
            "open_work": (reopen,),
            "progress_context": TaskProgressContext(
                completed_work_items=(create,),
                remaining_work=(reopen,),
                route_attempts=(
                    ProgressRouteAttempt(
                        step_index=4,
                        work_item=create,
                        source_context="ScheduleList",
                        action=ActionKind.TAP,
                        target="the floating add button",
                        destination_context="ScheduleEditor",
                        outcome="observed",
                    ),
                ),
            ),
        }
    )
    candidate = _proposal().model_copy(
        update={"target_expression": "the floating add button"}
    )
    model = StubPlannerModel(
        {
            "verification_id": "verification_1",
            "proposal_id": candidate.proposal_id,
            "disposition": "refuse",
            "reason": "This route opens a new schedule editor.",
            "unresolved_controls": [],
            "completed_work_items": [create],
            "proposal_work_item": reopen,
            "restarts_completed_work_item": create,
        }
    )
    identifiers = iter(("verification_1", "request_1"))
    verifier = ModelActionVerifier(
        model,
        "action-verifier-primary",
        identifier_factory=lambda: next(identifiers),
    )

    assessment = verifier.verify(request, candidate)

    refusal = assessment.refusal()
    assert refusal is not None
    assert refusal.code is ActionRefusalCode.COMPLETED_WORK_REGRESSION
    assert assessment.restarts_completed_work_item == create
    user_content = model.requests[0].messages[1].content[0]
    assert isinstance(user_content, TextModelContent)
    context = json.loads(user_content.text.split("Context JSON follows:\n", 1)[1])
    assert context["progress_context"]["route_attempts"][0]["target"] == (
        "the floating add button"
    )


def test_action_verifier_requires_an_exact_work_item_for_correction() -> None:
    work_item = "Inspect every relevant visible control."
    payload = {
        "verification_id": "verification_1",
        "proposal_id": "proposal_1",
        "disposition": "allow",
        "reason": "Open the dropdown.",
        "unresolved_controls": [],
        "completed_work_items": [],
        "proposal_work_item": work_item + " (open the dropdown)",
        "restarts_completed_work_item": None,
    }

    with pytest.raises(ValidationError, match="Copy each work item exactly"):
        ActionReadinessAssessment.model_validate(
            payload,
            context={
                "allowed_work_items": (work_item,),
                "completed_work_items": (),
            },
        )
