"""Independent pre-dispatch discovery action verification."""

import json

import pytest
from pydantic import ValidationError

from yewo.agent.actuation.proposal_verifier import (
    PROPOSAL_VERIFIER_MAX_COMPLETION_TOKENS,
    ActionReadinessAssessment,
    ActionReadinessDisposition,
    ActionReadinessRefusalBasis,
    ModelProposalVerifier,
)
from yewo.agent.actuation import (
    ActionKind,
    ActionOutcome,
    ActionOutcomeDisposition,
    ActionOutcomeSignal,
    ActionProposal,
    ActionRefusalCode,
)
from yewo.agent.planning.records import (
    PlanningFocusCause,
    PlanningFocusContext,
    PlanningRequest,
    ProgressRouteAttempt,
    TaskProgressContext,
)
from yewo.agent.planning.history import PlanningHistoryEntry, planning_screen_reference
from yewo.artifacts import ArtifactRetention
from yewo.models import MINIMUM_COMPLETION_TOKENS, ModelRole, TextModelContent
from yewo.agent.screen_analysis import (
    CurrentScreenAnalysis,
    ScreenElement,
    ScreenElementConfidence,
    ScreenElementKind,
    ScreenSemanticRole,
    ScreenVisualKind,
)

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
        target_expression="the Category dropdown",
        expected_outcome="The available record types become visible.",
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
        retention=ArtifactRetention.RUN,
    )
    return PlanningRequest(
        observation=observation("obs_1", "1111111111111111"),
        objective="Discover how to create a record.",
        open_work=("Cover every relevant visible field.",),
        completion_conditions=("The saved record can be reopened.",),
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


def _analysis(request: PlanningRequest) -> CurrentScreenAnalysis:
    observed = request.observation
    return CurrentScreenAnalysis(
        analysis_id="analysis-1",
        observation_id=observed.observation_id,
        screenshot_sha256=observed.screenshot.sha256,
        presentation=observed.presentation_key,
        screen_summary="Visible workflow form.",
        elements=(
            ScreenElement(
                element_ref="section-heading",
                name="Section heading",
                kind=ScreenElementKind(
                    visual=ScreenVisualKind.TEXT,
                    semantic=ScreenSemanticRole.LABEL,
                ),
                confidence=ScreenElementConfidence.HIGH,
            ),
            ScreenElement(
                element_ref="primary-field",
                name="Primary field",
                kind=ScreenElementKind(
                    visual=ScreenVisualKind.INPUT_SURFACE,
                    semantic=ScreenSemanticRole.TEXT_FIELD,
                ),
                confidence=ScreenElementConfidence.HIGH,
            ),
        ),
    )


def test_proposal_verifier_uses_dedicated_profile_and_refuses_unresolved_work() -> None:
    request = _request().model_copy(
        update={"open_work": ("Complete Primary field before saving.",)}
    )
    request = request.model_copy(update={"current_screen_analysis": _analysis(request)})
    model = StubPlannerModel(
        {
            "disposition": "refuse",
            "reason": "The action would leave visible discovery work unresolved.",
            "unresolved_controls": [
                {
                    "control_name": "Primary field",
                    "required_by": "objective_or_open_work",
                    "requirement_evidence": "Complete Primary field before saving.",
                    "reason": "The visible field has not been exercised.",
                }
            ],
            "completed_work_items": [],
            "proposal_work_item": "Complete Primary field before saving.",
            "restarts_completed_work_item": None,
        }
    )
    identifiers = iter(("verification_1", "request_1"))
    verifier = ModelProposalVerifier(
        model,
        "proposal-verifier-primary",
        identifier_factory=lambda: next(identifiers),
    )

    assessment = verifier.verify(request, _proposal())

    assert assessment.disposition is ActionReadinessDisposition.REFUSE
    assert assessment.refusal() is not None
    assert model.requests[0].role is ModelRole.PROPOSAL_VERIFIER
    assert model.requests[0].profile_id == "proposal-verifier-primary"
    assert model.requests[0].max_completion_tokens == PROPOSAL_VERIFIER_MAX_COMPLETION_TOKENS
    validation_context = model.validation_contexts[0]
    assert validation_context is not None
    assert validation_context["allowed_work_items"] == ("Complete Primary field before saving.",)
    requirement_evidence = validation_context["required_control_evidence"]
    assert isinstance(requirement_evidence, dict)
    assert requirement_evidence["Primary field"]["objective_or_open_work"] == (
        "Complete Primary field before saving.",
    )
    assert PROPOSAL_VERIFIER_MAX_COMPLETION_TOKENS == MINIMUM_COMPLETION_TOKENS
    assert model.requests[0].messages[1].cache_prefix_end
    assert model.requests[0].messages[2].cache_prefix_end is False
    history_content = model.requests[0].messages[2].content[0]
    assert isinstance(history_content, TextModelContent)
    history = json.loads(history_content.text.split("\n", 1)[1])
    assert history["target_expression"] == "the Category dropdown"
    assert history["feedback"]["disposition"] == "observed"


def test_proposal_verifier_receives_the_planners_exact_no_progress_constraints() -> None:
    focus = PlanningFocusContext(
        focus_id="focus-1",
        cause=PlanningFocusCause.SCREEN_NO_PROGRESS,
        reason="The screen did not advance.",
        unresolved_work=("Cover every relevant visible field.",),
        prohibited_interaction_intents=("tap|ref:save-control||||",),
    )
    request = _request().model_copy(update={"planning_focus": focus})
    model = StubPlannerModel(
        {
            "disposition": "allow",
            "reason": "The proposal follows a different route.",
            "unresolved_controls": [],
            "completed_work_items": [],
            "proposal_work_item": "Cover every relevant visible field.",
            "restarts_completed_work_item": None,
        }
    )
    identifiers = iter(("verification_1", "request_1"))

    ModelProposalVerifier(
        model,
        "proposal-verifier-primary",
        identifier_factory=identifiers.__next__,
    ).verify(request, _proposal())

    current = model.requests[0].messages[2].content[-2]
    assert isinstance(current, TextModelContent)
    context = json.loads(current.text.split("Current verification context:\n", 1)[1])
    assert context["planning_focus"]["prohibited_interaction_intents"] == [
        "tap|ref:save-control||||"
    ]


def test_proposal_verifier_fails_closed_on_invalid_allow_response() -> None:
    model = StubPlannerModel(
        {
            "disposition": "allow",
            "reason": "The action is safe.",
            "unresolved_controls": [
                {
                    "control_name": "Missing field",
                    "required_by": "objective_or_open_work",
                    "requirement_evidence": "Cover every relevant visible field.",
                    "reason": "It is unresolved.",
                }
            ],
            "completed_work_items": [],
            "proposal_work_item": "Cover every relevant visible field.",
            "restarts_completed_work_item": None,
        }
    )
    identifiers = iter(("verification_1", "request_1"))
    verifier = ModelProposalVerifier(
        model,
        "proposal-verifier-primary",
        identifier_factory=lambda: next(identifiers),
    )

    assessment = verifier.verify(_request(), _proposal())

    assert assessment.disposition is ActionReadinessDisposition.UNAVAILABLE
    assert assessment.unresolved_controls == ()
    refusal = assessment.refusal()
    assert refusal is not None
    assert refusal.code is ActionRefusalCode.VERIFICATION_UNAVAILABLE


def test_proposal_verifier_uses_analysis_and_rejects_a_label_as_unresolved() -> None:
    model = StubPlannerModel(
        {
            "disposition": "refuse",
            "reason": "The section heading remains unresolved.",
            "unresolved_controls": [
                {
                    "control_name": "Section heading",
                    "required_by": "objective_or_open_work",
                    "requirement_evidence": "Cover every relevant visible field.",
                    "reason": "No value is visible.",
                }
            ],
            "completed_work_items": [],
            "proposal_work_item": "Cover every relevant visible field.",
            "restarts_completed_work_item": None,
        }
    )
    request = _request()
    request = request.model_copy(update={"current_screen_analysis": _analysis(request)})
    identifiers = iter(("verification_1", "request_1"))

    assessment = ModelProposalVerifier(
        model,
        "proposal-verifier-primary",
        identifier_factory=lambda: next(identifiers),
    ).verify(request, _proposal())

    assert assessment.disposition is ActionReadinessDisposition.UNAVAILABLE
    validation_context = model.validation_contexts[0]
    assert validation_context is not None
    assert validation_context["interactive_control_names"] == ("Primary field",)
    current_content = model.requests[0].messages[2].content[-2]
    assert isinstance(current_content, TextModelContent)
    context = json.loads(current_content.text.split("\n", 1)[1])
    screen_context = context["current_screen_analysis"]
    assert screen_context == {
        "elements": [
            {
                "affordance": "interactive",
                "confidence": "high",
                "element_ref": "primary-field",
                "kind": "text_field",
                "name": "Primary field",
            }
        ]
    }
    assert "current_observation_id" not in context
    assert "keyboard_visible" not in context
    assert context["proposal"] == {
        "action": "tap",
        "expected_outcome": "The editor saves and closes.",
        "reason": "Save the current editor.",
        "target_expression": "the visible Save button",
    }


def test_proposal_verifier_rejects_control_agnostic_requirement_claim() -> None:
    request = _request()
    request = request.model_copy(update={"current_screen_analysis": _analysis(request)})
    model = StubPlannerModel(
        {
            "disposition": "refuse",
            "reason": "Every visible field must be completed.",
            "unresolved_controls": [
                {
                    "control_name": "Primary field",
                    "required_by": "objective_or_open_work",
                    "requirement_evidence": "Every visible field must be completed.",
                    "reason": "The visible field is empty.",
                }
            ],
            "completed_work_items": [],
            "proposal_work_item": "Cover every relevant visible field.",
            "restarts_completed_work_item": None,
        }
    )
    identifiers = iter(("verification_1", "request_1"))

    assessment = ModelProposalVerifier(
        model,
        "proposal-verifier-primary",
        identifier_factory=identifiers.__next__,
    ).verify(request, _proposal())

    assert assessment.disposition is ActionReadinessDisposition.UNAVAILABLE


def test_proposal_verifier_accepts_input_contract_as_a_producible_requirement_basis() -> None:
    request = _request().model_copy(update={"declared_value_names": ("primary_field",)})
    request = request.model_copy(update={"current_screen_analysis": _analysis(request)})
    model = StubPlannerModel(
        {
            "disposition": "refuse",
            "reason": "The declared primary input is unresolved.",
            "unresolved_controls": [
                {
                    "control_name": "Primary field",
                    "required_by": "input_contract",
                    "requirement_evidence": "primary_field",
                    "reason": "The declared input has not been entered.",
                }
            ],
            "completed_work_items": [],
            "proposal_work_item": "Cover every relevant visible field.",
            "restarts_completed_work_item": None,
        }
    )
    identifiers = iter(("verification_1", "request_1"))

    assessment = ModelProposalVerifier(
        model,
        "proposal-verifier-primary",
        identifier_factory=identifiers.__next__,
    ).verify(request, _proposal())

    assert assessment.disposition is ActionReadinessDisposition.REFUSE
    assert assessment.unresolved_controls[0].required_by.value == "input_contract"
    task_content = next(
        content
        for content in model.requests[0].messages[2].content
        if isinstance(content, TextModelContent)
        and content.text.startswith("Task context for this run:\n")
    )
    task_context = json.loads(task_content.text.split("Task context for this run:\n", 1)[1])
    assert task_context["declared_value_names"] == ["primary_field"]


def test_proposal_verifier_refuses_a_route_that_restarts_completed_work() -> None:
    create = "Create and save one reusable record."
    reopen = "Return to the saved record and establish that it can be repeated or updated."
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
                        source_context="RecordList",
                        action=ActionKind.TAP,
                        target="the visible create control",
                        destination_context="RecordEditor",
                        outcome="observed",
                    ),
                ),
            ),
        }
    )
    candidate = _proposal().model_copy(update={"target_expression": "the visible create control"})
    model = StubPlannerModel(
        {
            "disposition": "refuse",
            "reason": "This route opens a new record editor.",
            "unresolved_controls": [],
            "completed_work_items": [create],
            "proposal_work_item": reopen,
            "restarts_completed_work_item": create,
        }
    )
    identifiers = iter(("verification_1", "request_1"))
    verifier = ModelProposalVerifier(
        model,
        "proposal-verifier-primary",
        identifier_factory=lambda: next(identifiers),
    )

    assessment = verifier.verify(request, candidate)

    refusal = assessment.refusal()
    assert refusal is not None
    assert refusal.code is ActionRefusalCode.COMPLETED_WORK_REGRESSION
    assert assessment.restarts_completed_work_item == create
    user_content = next(
        content
        for content in model.requests[0].messages[2].content
        if isinstance(content, TextModelContent)
        and content.text.startswith("Current verification context:")
    )
    context = json.loads(user_content.text.split("Current verification context:\n", 1)[1])
    assert context["progress_context"]["route_attempts"][0]["target"] == (
        "the visible create control"
    )
    assert "remaining_work" not in context["progress_context"]
    assert "interaction_intent_key" not in context["progress_context"]["route_attempts"][0]
    assert "source_screen_fingerprint" not in context["progress_context"]["route_attempts"][0]


def test_proposal_verifier_requires_an_exact_work_item_for_correction() -> None:
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


def test_unexplored_route_refusal_requires_one_remaining_route() -> None:
    payload = {
        "verification_id": "verification_1",
        "proposal_id": "proposal_1",
        "disposition": "refuse",
        "reason": "Discovery has another goal-relevant entry route.",
        "unresolved_controls": [],
        "completed_work_items": [],
        "proposal_work_item": None,
        "restarts_completed_work_item": None,
        "refusal_basis": "unexplored_route",
        "unexplored_route": None,
    }

    with pytest.raises(ValidationError, match="must name the remaining route"):
        ActionReadinessAssessment.model_validate(payload)


def test_unexplored_route_refusal_reports_discovery_incomplete() -> None:
    assessment = ActionReadinessAssessment(
        verification_id="verification_1",
        proposal_id="proposal_1",
        disposition=ActionReadinessDisposition.REFUSE,
        reason="The completed checkout route does not cover another visible entry route.",
        completed_work_items=(),
        proposal_work_item=None,
        restarts_completed_work_item=None,
        refusal_basis=ActionReadinessRefusalBasis.UNEXPLORED_ROUTE,
        unexplored_route="Use Search to enter the ordering flow.",
    )

    refusal = assessment.refusal()

    assert refusal is not None
    assert refusal.code is ActionRefusalCode.DISCOVERY_INCOMPLETE
    assert "Use Search to enter the ordering flow" in refusal.reason


def test_proposal_verifier_does_not_treat_recovery_guidance_as_task_work() -> None:
    work_item = "Create and save one reusable record."
    recovery_step = "Refill fields lost after the application crash."
    request = _request().model_copy(
        update={
            "open_work": (work_item,),
            "progress_context": TaskProgressContext(remaining_work=(recovery_step,)),
        }
    )
    model = StubPlannerModel(
        {
            "disposition": "allow",
            "reason": "Continue the canonical task through the recovery route.",
            "unresolved_controls": [],
            "completed_work_items": [],
            "proposal_work_item": work_item,
            "restarts_completed_work_item": None,
        }
    )
    identifiers = iter(("verification_1", "request_1"))
    verifier = ModelProposalVerifier(
        model,
        "proposal-verifier-primary",
        identifier_factory=lambda: next(identifiers),
    )

    verifier.verify(request, _proposal())

    validation_context = model.validation_contexts[0]
    assert validation_context is not None
    assert validation_context["allowed_work_items"] == (work_item,)


def test_proposal_verifier_rejects_unresolved_control_without_analyzed_evidence() -> None:
    payload = {
        "verification_id": "verification_1",
        "proposal_id": "proposal_1",
        "disposition": "refuse",
        "reason": "The workflow is incomplete.",
        "unresolved_controls": [
            {
                "control_name": "Unreported control",
                "required_by": "objective_or_open_work",
                "requirement_evidence": "Complete the workflow.",
                "reason": "No value is visible.",
            }
        ],
        "completed_work_items": [],
        "proposal_work_item": None,
        "restarts_completed_work_item": None,
    }

    with pytest.raises(ValidationError, match="analyzed interactive control"):
        ActionReadinessAssessment.model_validate(payload, context={"interactive_control_names": ()})


def test_proposal_verifier_rejects_noninteractive_unresolved_evidence() -> None:
    payload = {
        "verification_id": "verification_1",
        "proposal_id": "proposal_1",
        "disposition": "refuse",
        "reason": "The workflow is incomplete.",
        "unresolved_controls": [
            {
                "control_name": "Section title",
                "required_by": "objective_or_open_work",
                "requirement_evidence": "Complete the workflow.",
                "reason": "No value is visible.",
            }
        ],
        "completed_work_items": [],
        "proposal_work_item": None,
        "restarts_completed_work_item": None,
    }

    with pytest.raises(ValidationError, match="analyzed interactive control"):
        ActionReadinessAssessment.model_validate(
            payload, context={"interactive_control_names": ("Quantity",)}
        )
