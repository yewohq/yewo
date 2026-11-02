"""Independent pre-dispatch discovery action verification."""

import json

from yewo.agent.action_verifier import (
    ACTION_VERIFIER_MAX_COMPLETION_TOKENS,
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
from yewo.agent.planner import PlanningRequest
from yewo.agent.planning_history import PlanningHistoryEntry, planning_screen_reference
from yewo.artifacts import ArtifactRetention, ArtifactSensitivity
from yewo.models import MINIMUM_COMPLETION_TOKENS, ModelRole, TextModelContent

from agent.discovery.support import observation
from agent.planner_support import StubPlannerModel


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
        signals=frozenset({ActionOutcomeSignal.SCREEN_CHANGED}),
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
    assert model.requests[0].role is ModelRole.ACTION_VERIFIER
    assert model.requests[0].profile_id == "action-verifier-primary"
    assert model.requests[0].max_completion_tokens == ACTION_VERIFIER_MAX_COMPLETION_TOKENS
    assert ACTION_VERIFIER_MAX_COMPLETION_TOKENS == MINIMUM_COMPLETION_TOKENS
    user_content = model.requests[0].messages[1].content[0]
    assert isinstance(user_content, TextModelContent)
    context = json.loads(user_content.text.split("Context JSON follows:\n", 1)[1])
    assert context["recent_history"][0]["target_expression"] == (
        "the Trainings Type dropdown"
    )
    assert context["recent_history"][0]["feedback"]["disposition"] == "observed"


def test_action_verifier_fails_closed_on_invalid_allow_response() -> None:
    model = StubPlannerModel(
        {
            "verification_id": "verification_1",
            "proposal_id": "proposal_1",
            "disposition": "allow",
            "reason": "The action is safe.",
            "unresolved_controls": ["A field is unresolved."],
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
