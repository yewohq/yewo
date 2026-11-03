"""Offline tests for the shared semantic task planner."""

import json
from typing import Any

import pytest
from pydantic import ValidationError

from yewo.agent.actuation import (
    ActionKind,
    ActionOutcome,
    ActionOutcomeDisposition,
    ActionOutcomeSignal,
    ActionProposal,
    ActionRefusal,
    ActionRefusalCode,
    DispatchDisposition,
    DispatchReceipt,
    LiteralText,
    ScrollDistance,
)
from yewo.devices.interaction import ScrollDirection
from yewo.agent.planning.planner import (
    GroundedTargetPoint,
    PLANNER_MAX_COMPLETION_TOKENS,
    PlanningHistoryEntry,
    PlanningRequest,
    TASK_PLANNER_SYSTEM_PROMPT,
    TASK_PLANNER_USER_PROMPT,
    TaskPlanningError,
    TaskProgressContext,
)
from yewo.artifacts import ArtifactRetention, ArtifactSensitivity
from yewo.grounding import PixelPoint
from yewo.models import ModelCallDisposition, ModelMessageRole, ModelRole

from agent.planning.support import (
    StubPlannerModel,
    planner as _planner,
    planning_request as _planning_request,
    proposal_output as _proposal_output,
    user_context as _user_context,
)


def test_planner_returns_the_existing_action_proposal_contract() -> None:
    model = StubPlannerModel(_proposal_output())

    proposal = _planner(model).plan(_planning_request())

    assert proposal == ActionProposal.model_validate(_proposal_output())
    assert model.response_models == [ActionProposal]
    assert model.validation_contexts == [
        {
            "expected_proposal_id": "proposal-1",
            "expected_observation_id": "obs_1",
            "available_actions": ("cannot_progress", "done", "tap", "type", "wait"),
            "declared_value_names": ("account_password",),
        }
    ]
    model_request = model.requests[0]
    assert model_request.request_id == "model-request-1"
    assert model_request.profile_id == "planner-primary"
    assert model_request.role is ModelRole.TASK_PLANNER
    assert model_request.max_completion_tokens == PLANNER_MAX_COMPLETION_TOKENS
    assert model_request.response_contract is not None
    assert model_request.response_contract.name == "action-proposal-v1"
    assert tuple(message.role for message in model_request.messages) == (
        ModelMessageRole.SYSTEM,
        ModelMessageRole.USER,
    )
    system_content = model_request.messages[0].content[0]
    assert system_content.kind == "text"
    assert system_content.text == TASK_PLANNER_SYSTEM_PROMPT
    section_headings = (
        "## Trusted evidence",
        "## Workflow progress",
        "## Describing a target",
        "## UI interactions",
        "## Action usage",
        "## Input handling",
        "## Output",
    )
    section_positions = tuple(system_content.text.index(heading) for heading in section_headings)
    assert section_positions == tuple(sorted(section_positions))
    assert all(system_content.text.count(heading) == 1 for heading in section_headings)
    assert "Target, Attributes, Relationships, and Scope" in system_content.text
    assert "interaction_refinement_for_proposal_id" in system_content.text
    assert "For every action, state a concise reason" in system_content.text
    for action in ActionKind:
        assert f"- `{action.value}`:" in system_content.text
    assert "field has no reusable input contract" in system_content.text
    assert "use `type` directly" in system_content.text
    assert "visual keyboard type" in system_content.text
    assert "Never use untargeted `type` to move focus" in system_content.text
    assert "focus another visible field while the keyboard remains" in system_content.text
    assert "dropdown or spinner" in system_content.text
    assert "newly opened overlay has no observable effect" in system_content.text
    assert "not merely its label" in system_content.text
    assert "keyboard obscures" in system_content.text
    assert "compare current evidence with open_work" in system_content.text
    prompt, image = _user_context(model_request)
    assert prompt.startswith(f"{TASK_PLANNER_USER_PROMPT}\n")
    context = json.loads(prompt.split("\n", 1)[1])
    assert context["proposal_id"] == "proposal-1"
    assert context["source_observation_id"] == "obs_1"
    assert context["objective"] == "Add one account."
    assert "done" in context["available_actions"]
    assert "cannot_progress" in context["available_actions"]
    assert image.screenshot.sha256 == "a" * 64


@pytest.mark.parametrize(
    "updates",
    [
        {"objective": "Open settings."},
        {
            "objective": "Learn the account-creation route.",
            "open_work": ("Find the entry point that opens the account form.",),
        },
        {
            "objective": "Execute the invalid-password scenario.",
            "completion_conditions": ("The password validation message is visible.",),
        },
    ],
)
def test_one_request_contract_accepts_each_workflow_task(updates: dict[str, Any]) -> None:
    request = _planning_request(**updates)

    assert request.objective == updates["objective"]


def test_planner_receives_reconciled_progress_without_changing_completion_conditions() -> None:
    model = StubPlannerModel(_proposal_output())
    request = _planning_request(
        open_work=("Add one exercise.",),
        progress_context=TaskProgressContext(
            current_progress=("The schedule is visibly saved.",),
            remaining_work=("Add one exercise.",),
            historical_memory_facts=("The Add control previously opened the exercise editor.",),
            exhausted_branches=("The overflow menu showed no relevant command.",),
        ),
    )

    _planner(model).plan(request)

    prompt, unused_image = _user_context(model.requests[0])
    context = json.loads(prompt.split("\n", 1)[1])
    assert context["open_work"] == ["Add one exercise."]
    assert context["completion_conditions"] == ["The new account appears in the list."]
    assert context["progress_context"] == {
        "completed_work_items": [],
        "current_progress": ["The schedule is visibly saved."],
        "exhausted_branches": ["The overflow menu showed no relevant command."],
        "historical_memory_facts": ["The Add control previously opened the exercise editor."],
        "known_procedures": [],
        "recovery_guidance": [],
        "remaining_work": ["Add one exercise."],
        "route_attempts": [],
    }


def test_history_contains_trusted_feedback_without_repeating_literal_text() -> None:
    refused_proposal = ActionProposal(
        proposal_id="proposal-refused",
        source_observation_id="obs_0",
        action=ActionKind.TAP,
        target_expression="the disabled Save button",
        expected_outcome="The form saves.",
        reason="Saving is required to finish the form.",
    )
    refused = PlanningHistoryEntry(
        proposal=refused_proposal,
        refusal=ActionRefusal(
            proposal_id=refused_proposal.proposal_id,
            code=ActionRefusalCode.ACTION_NOT_ALLOWED,
            reason="The action is not allowed in this mode.",
        ),
    )
    typed_proposal = ActionProposal(
        proposal_id="proposal-typed",
        source_observation_id="obs_0",
        action=ActionKind.TYPE,
        text=LiteralText(value="private literal text"),
        expected_outcome="The password field contains the text.",
        reason="The password field must be completed.",
    )
    observed = PlanningHistoryEntry(
        proposal=typed_proposal,
        outcome=ActionOutcome(
            outcome_id="outcome-1",
            authorization_id="authorization-1",
            receipt_id="receipt-1",
            action=ActionKind.TYPE,
            source_observation_id="obs_0",
            settled_observation_id="obs_1",
            disposition=ActionOutcomeDisposition.OBSERVED,
            signals=frozenset({ActionOutcomeSignal.TEXT_NOT_LANDED}),
            reason="The target field did not contain the entered text.",
            sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
            retention=ArtifactRetention.RUN,
        ),
    )
    scrolled_proposal = ActionProposal(
        proposal_id="proposal-scrolled",
        source_observation_id="obs_0",
        action=ActionKind.SCROLL,
        direction=ScrollDirection.DOWN,
        distance=ScrollDistance.SHORT,
        expected_outcome="More accounts become visible.",
        reason="The requested account is not visible in the current viewport.",
    )
    scrolled = PlanningHistoryEntry(
        proposal=scrolled_proposal,
        outcome=ActionOutcome(
            outcome_id="outcome-2",
            authorization_id="authorization-2",
            receipt_id="receipt-2",
            action=ActionKind.SCROLL,
            source_observation_id="obs_0",
            settled_observation_id="obs_2",
            disposition=ActionOutcomeDisposition.NO_OBSERVABLE_CHANGE,
            signals=frozenset(),
            reason="The screen did not visibly change.",
            sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
            retention=ArtifactRetention.RUN,
        ),
    )
    model = StubPlannerModel(_proposal_output())

    _planner(model).plan(_planning_request(history=(refused, observed, scrolled)))

    prompt, _ = _user_context(model.requests[0])
    context = json.loads(prompt.split("\n", 1)[1])
    assert all("expected_outcome" not in entry for entry in context["history"])
    assert context["history"][0]["feedback"]["code"] == "action_not_allowed"
    assert context["history"][1]["feedback"]["signals"] == ["text_not_landed"]
    assert context["history"][2]["direction"] == "down"
    assert context["history"][2]["distance"] == "short"
    assert "private literal text" not in prompt


def test_planner_receives_target_point_and_accepts_one_refined_correction() -> None:
    failed_proposal = ActionProposal(
        proposal_id="proposal-failed",
        source_observation_id="obs_0",
        action=ActionKind.TAP,
        target_expression="the Edit Schedule item",
        expected_outcome="The schedule editor opens.",
        reason="The schedule must be edited.",
    )
    failed_entry = PlanningHistoryEntry(
        proposal=failed_proposal,
        outcome=ActionOutcome(
            outcome_id="outcome-failed",
            authorization_id="authorization-failed",
            receipt_id="receipt-failed",
            action=ActionKind.TAP,
            source_observation_id="obs_0",
            settled_observation_id="obs_1",
            disposition=ActionOutcomeDisposition.NO_OBSERVABLE_CHANGE,
            signals=frozenset(),
            reason="The screen did not visibly change.",
            sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
            retention=ArtifactRetention.RUN,
        ),
        grounded_target_point=GroundedTargetPoint(
            point=PixelPoint(x=546, y=1240),
            screenshot_width_px=1080,
            screenshot_height_px=2400,
        ),
    )
    model = StubPlannerModel(
        _proposal_output(
            grounding_correction_for_proposal_id="proposal-failed",
            target_expression=(
                "the Edit Schedule row directly below Trainings Schedule near the top of "
                "the open navigation drawer, not the blank lower drawer area"
            ),
            reason=(
                "The previous point at (546,1240) was below the visible menu rows; refine "
                "the target toward the upper drawer row."
            ),
        )
    )

    proposal = _planner(model).plan(_planning_request(history=(failed_entry,)))

    assert proposal.grounding_correction_for_proposal_id == "proposal-failed"
    assert failed_entry.grounded_target_point is not None
    assert failed_entry.grounded_target_point.correction().minimum_distance_px == 22
    prompt, _ = _user_context(model.requests[0])
    context = json.loads(prompt.split("\n", 1)[1])
    assert context["history"][0]["grounded_target_point"] == {
        "point": {"x": 546, "y": 1240},
        "screenshot_height_px": 2400,
        "screenshot_width_px": 1080,
    }


def test_planner_rejects_a_correction_of_a_correction() -> None:
    corrected_proposal = ActionProposal(
        proposal_id="proposal-correction",
        source_observation_id="obs_0",
        action=ActionKind.TAP,
        grounding_correction_for_proposal_id="proposal-original",
        target_expression="the upper Edit Schedule row",
        expected_outcome="The schedule editor opens.",
        reason="The prior point was below the visible row.",
    )
    corrected_entry = PlanningHistoryEntry(
        proposal=corrected_proposal,
        outcome=ActionOutcome(
            outcome_id="outcome-correction",
            authorization_id="authorization-correction",
            receipt_id="receipt-correction",
            action=ActionKind.TAP,
            source_observation_id="obs_0",
            settled_observation_id="obs_1",
            disposition=ActionOutcomeDisposition.NO_OBSERVABLE_CHANGE,
            signals=frozenset(),
            reason="The screen did not visibly change.",
            sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
            retention=ArtifactRetention.RUN,
        ),
        grounded_target_point=GroundedTargetPoint(
            point=PixelPoint(x=150, y=675),
            screenshot_width_px=1080,
            screenshot_height_px=2400,
        ),
    )
    model = StubPlannerModel(
        _proposal_output(
            grounding_correction_for_proposal_id="proposal-correction",
            target_expression="the Edit Schedule text row at the top of the drawer",
        )
    )

    with pytest.raises(TaskPlanningError, match="cannot correct another grounding correction"):
        _planner(model).plan(_planning_request(history=(corrected_entry,)))


def test_target_rejection_gives_planner_device_reason_and_grounded_point() -> None:
    failed_proposal = ActionProposal(
        proposal_id="proposal-rejected",
        source_observation_id="obs_0",
        action=ActionKind.FOCUS,
        target_expression="the Time input field below its label",
        expected_outcome="The Time field gains focus.",
        reason="The field must accept a duration.",
    )
    rejected_entry = PlanningHistoryEntry(
        proposal=failed_proposal,
        receipt=DispatchReceipt(
            receipt_id="receipt-rejected",
            authorization_id="authorization-rejected",
            proposal_id=failed_proposal.proposal_id,
            action=ActionKind.FOCUS,
            disposition=DispatchDisposition.TARGET_REJECTED,
            reason="The grounded target did not accept keyboard focus.",
            sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
            retention=ArtifactRetention.RUN,
        ),
        grounded_target_point=GroundedTargetPoint(
            point=PixelPoint(x=540, y=1437),
            screenshot_width_px=1080,
            screenshot_height_px=2400,
        ),
    )
    model = StubPlannerModel(_proposal_output())

    _planner(model).plan(_planning_request(history=(rejected_entry,)))

    prompt, _ = _user_context(model.requests[0])
    history = json.loads(prompt.split("\n", 1)[1])["history"][0]
    assert history["feedback"] == {
        "kind": "receipt",
        "disposition": "target_rejected",
        "reason": "The grounded target did not accept keyboard focus.",
    }
    assert history["grounded_target_point"]["point"] == {"x": 540, "y": 1437}


def test_history_accepts_full_recovery_adjusted_run_and_compacts_older_evidence() -> None:
    candidate = ActionProposal(
        proposal_id="proposal-refused",
        source_observation_id="obs_0",
        action=ActionKind.TAP,
        target_expression="the Save button",
        expected_outcome="The form saves.",
        reason="The form must be saved.",
    )
    entry = PlanningHistoryEntry(
        proposal=candidate,
        refusal=ActionRefusal(
            proposal_id=candidate.proposal_id,
            code=ActionRefusalCode.KEYBOARD_RULE,
            reason="Dismiss the visible keyboard before pointer actions.",
        ),
    )
    model = StubPlannerModel(_proposal_output())

    _planner(model).plan(_planning_request(history=(entry,) * 110))

    prompt, _ = _user_context(model.requests[0])
    context = json.loads(prompt.split("\n", 1)[1])
    assert len(context["history"]) == 14
    assert context["earlier_history_summary"] == {
        "entries": [
            {
                "action": "tap",
                "count": 96,
                "feedback": {"code": "keyboard_rule", "kind": "refusal"},
                "target_expression": "the Save button",
            }
        ],
        "omitted_groups": 0,
    }


def test_provider_failure_is_distinct_from_cannot_progress() -> None:
    failed_model = StubPlannerModel(
        None,
        disposition=ModelCallDisposition.PROVIDER_ERROR,
    )

    with pytest.raises(TaskPlanningError) as failure:
        _planner(failed_model).plan(_planning_request())

    assert failure.value.disposition is ModelCallDisposition.PROVIDER_ERROR

    stopped_model = StubPlannerModel(_proposal_output(ActionKind.CANNOT_PROGRESS))
    proposal = _planner(stopped_model).plan(_planning_request())
    assert proposal.action is ActionKind.CANNOT_PROGRESS


def test_rejects_an_undeclared_value_name() -> None:
    output = {
        "proposal_id": "proposal-1",
        "source_observation_id": "obs_1",
        "action": "type",
        "text": {"source": "declared_value", "value_name": "unknown_password"},
        "expected_outcome": "The password field contains the declared value.",
        "reason": "The password field must be completed.",
    }

    with pytest.raises(TaskPlanningError, match="unavailable declared value"):
        _planner(StubPlannerModel(output)).plan(_planning_request())


@pytest.mark.parametrize(
    "output",
    [
        _proposal_output(proposal_id="wrong-proposal"),
        _proposal_output(source_observation_id="obs_other"),
        _proposal_output(ActionKind.WAIT),
        {
            "proposal_id": "proposal-1",
            "source_observation_id": "obs_1",
            "action": "tap",
        },
    ],
)
def test_rejects_a_response_that_is_not_usable_for_the_request(
    output: dict[str, object],
) -> None:
    model = StubPlannerModel(output)
    request = _planning_request(available_actions=frozenset({ActionKind.TAP}))

    with pytest.raises(TaskPlanningError):
        _planner(model).plan(request)


@pytest.mark.parametrize("extra_field", ["workflow_mode", "declared_values"])
def test_request_rejects_workflow_switches_and_value_bodies(extra_field: str) -> None:
    values = _planning_request().model_dump()
    values[extra_field] = "not allowed"

    with pytest.raises(ValidationError):
        PlanningRequest.model_validate(values)


def test_history_rejects_an_outcome_from_another_observation() -> None:
    proposal = ActionProposal.model_validate(_proposal_output())
    outcome = ActionOutcome(
        outcome_id="outcome-1",
        authorization_id="authorization-1",
        receipt_id="receipt-1",
        action=ActionKind.TAP,
        source_observation_id="obs_other",
        settled_observation_id="obs_2",
        disposition=ActionOutcomeDisposition.OBSERVED,
        signals=frozenset({ActionOutcomeSignal.PIXELS_CHANGED}),
        reason="The screen changed.",
        sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
        retention=ArtifactRetention.RUN,
    )

    with pytest.raises(ValidationError, match="proposal observation"):
        PlanningHistoryEntry(proposal=proposal, outcome=outcome)
