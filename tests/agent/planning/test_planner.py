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
    ActionProposalDecision,
    ActionRefusal,
    ActionRefusalCode,
    DispatchDisposition,
    DispatchReceipt,
    LiteralText,
    ScrollDistance,
)
from yewo.agent.screen_analysis import (
    ScreenElementKind,
    ScreenSemanticRole,
    ScreenVisualKind,
)
from yewo.devices.interaction import ScrollDirection
from yewo.agent.planning.history import RECENT_PLANNING_HISTORY_LIMIT
from yewo.agent.planning.planner import (
    GroundedTargetPoint,
    PLANNER_MAX_COMPLETION_TOKENS,
    PlanningHistoryEntry,
    TASK_PLANNER_SYSTEM_PROMPT,
    TASK_PLANNER_USER_PROMPT,
)
from yewo.agent.planning.records import (
    PlanningRequest,
    TaskProgressContext,
)
from yewo.agent.screen_analysis import (
    CurrentScreenAnalysis,
    ScreenElement,
    ScreenElementConfidence,
)
from yewo.artifacts import ArtifactRetention
from yewo.grounding import PixelPoint
from yewo.models import (
    ImageModelContent,
    ModelMessageRole,
    ModelRequest,
    ModelRole,
    TextModelContent,
)
from agent.planning.support import (
    history_contexts as _history_contexts,
    StubPlannerModel,
    planner as _planner,
    planning_request as _planning_request,
    proposal_output as _proposal_output,
    user_context as _user_context,
)
from support import DISPATCHED_AT


def test_planner_returns_the_existing_action_proposal_contract() -> None:
    model = StubPlannerModel(_proposal_output())

    proposal = _planner(model).plan(_planning_request())

    assert proposal == ActionProposal.model_validate(
        {
            "proposal_id": "proposal-1",
            "source_observation_id": "obs_1",
            **_proposal_output(),
        }
    )
    assert model.response_models == [ActionProposalDecision]
    assert model.validation_contexts == [
        {
            "available_actions": ("cannot_progress", "done", "tap", "type", "wait"),
            "declared_value_names": ("declared_secret",),
            "correctable_grounding_proposals": {},
            "available_element_refs": (),
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
        ModelMessageRole.USER,
    )
    system_content = model_request.messages[0].content[0]
    assert system_content.kind == "text"
    assert system_content.text == TASK_PLANNER_SYSTEM_PROMPT
    assert system_content.text.strip()
    task_content = next(
        content
        for content in model_request.messages[2].content
        if isinstance(content, TextModelContent) and content.text.startswith("Task context:\n")
    )
    assert task_content.kind == "text"
    assert task_content.text.startswith("Task context:\n")
    task_context = json.loads(task_content.text.split("\n", 1)[1])
    assert task_context["objective"] == "Add one record."
    prompt, image = _user_context(model_request)
    stable_prompt = model_request.messages[1].content[0]
    assert isinstance(stable_prompt, TextModelContent)
    assert stable_prompt.text == TASK_PLANNER_USER_PROMPT
    assert model_request.messages[1].cache_prefix_end
    assert prompt.startswith("Current decision context:\n")
    context = json.loads(prompt.split("\n", 1)[1])
    assert "proposal_id" not in context
    assert "source_observation_id" not in context
    assert "objective" not in context
    offered = {entry["action"] for entry in context["available_actions"]}
    assert {"done", "cannot_progress"} <= offered
    tap = next(entry for entry in context["available_actions"] if entry["action"] == "tap")
    assert tap["blocked_while_keyboard_visible"] is True
    assert tap["supports_grounding_correction"] is True
    assert sorted(tap["required_fields"]) == ["expected_outcome", "reason"]
    assert image.screenshot.sha256 == "a" * 64


@pytest.mark.parametrize(
    "updates",
    [
        {"objective": "Open settings."},
        {
            "objective": "Learn the record-creation route.",
            "open_work": ("Find the entry point that opens the record form.",),
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


def test_reconciled_progress_preserves_completion_conditions() -> None:
    model = StubPlannerModel(_proposal_output())
    request = _planning_request(
        open_work=("Add one item.",),
        progress_context=TaskProgressContext(
            current_progress=("The record is visibly saved.",),
            remaining_work=("Add one item.",),
            historical_memory_facts=("The Add control previously opened the item editor.",),
            recovery_guidance=("The overflow menu showed no relevant command.",),
        ),
    )

    _planner(model).plan(request)

    model_request = model.requests[0]
    prompt, unused_image = _user_context(model_request)
    context = json.loads(prompt.split("\n", 1)[1])
    task_content = next(
        content
        for content in model_request.messages[2].content
        if isinstance(content, TextModelContent) and content.text.startswith("Task context:\n")
    )
    system_prompt = model_request.messages[0].content[0]
    assert isinstance(task_content, TextModelContent) and isinstance(
        system_prompt, TextModelContent
    )
    assert "progress_context.remaining_work" not in system_prompt.text
    task_context = json.loads(task_content.text.split("\n", 1)[1])
    assert task_context["completion_conditions"] == ["The new record appears in the list."]
    assert context["open_work"] == ["Add one item."]
    assert context["current_observation"]["platform"] == "android"
    assert context["progress_context"] == {
        "completed_work_items": [],
        "current_progress": ["The record is visibly saved."],
        "historical_memory_facts": ["The Add control previously opened the item editor."],
        "known_procedures": [],
        "recovery_guidance": ["The overflow menu showed no relevant command."],
        "route_attempts": [],
    }


def test_planner_receives_one_compact_current_screen_analysis() -> None:
    request = _planning_request()
    observation = request.observation
    analysis = CurrentScreenAnalysis(
        analysis_id="analysis-1",
        observation_id=observation.observation_id,
        screenshot_sha256=observation.screenshot.sha256,
        presentation=observation.presentation_key,
        screen_summary="Visible application form.",
        elements=(
            ScreenElement(
                element_ref="primary-field",
                name="Primary field",
                kind=ScreenElementKind(
                    visual=ScreenVisualKind.INPUT_SURFACE,
                    semantic=ScreenSemanticRole.TEXT_FIELD,
                ),
                state="empty",
                confidence=ScreenElementConfidence.HIGH,
                visual_cues=("below Section heading",),
            ),
        ),
    )
    model = StubPlannerModel(_proposal_output())

    _planner(model).plan(request.model_copy(update={"current_screen_analysis": analysis}))

    prompt, unused_image = _user_context(model.requests[0])
    context = json.loads(prompt.split("\n", 1)[1])
    assert context["current_screen_analysis"] == {
        "analysis_id": "analysis-1",
        "observation_id": "obs_1",
        "screen_summary": "Visible application form.",
        "elements": [
            {
                "element_ref": "primary-field",
                "name": "Primary field",
                "kind": {"visual": "input_surface", "semantic": "text_field"},
                "affordance": "interactive",
                "state": "empty",
                "confidence": "high",
                "visual_cues": ["below Section heading"],
            }
        ],
    }


def test_repair_reference_screenshot_is_a_second_planner_image() -> None:
    model = StubPlannerModel(_proposal_output())
    request = _planning_request()
    request = request.model_copy(update={"reference_screenshot": request.observation.screenshot})

    _planner(model).plan(request)

    content = model.requests[0].messages[-1].content
    images = tuple(item for item in content if isinstance(item, ImageModelContent))
    text = tuple(item.text for item in content if isinstance(item, TextModelContent))
    assert len(images) == 2
    assert any("Compiled target screenshot" in item for item in text)


def test_planning_rejects_analysis_for_another_observation() -> None:
    request = _planning_request()
    observation = request.observation
    analysis = CurrentScreenAnalysis(
        analysis_id="analysis-1",
        observation_id="obs-old",
        screenshot_sha256=observation.screenshot.sha256,
        presentation=observation.presentation_key,
        screen_summary="Visible application form.",
    )

    with pytest.raises(ValidationError, match="must match the planning observation"):
        PlanningRequest.model_validate(request.model_dump() | {"current_screen_analysis": analysis})


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
            retention=ArtifactRetention.RUN,
        ),
    )
    scrolled_proposal = ActionProposal(
        proposal_id="proposal-scrolled",
        source_observation_id="obs_0",
        action=ActionKind.SCROLL,
        direction=ScrollDirection.DOWN,
        distance=ScrollDistance.SHORT,
        expected_outcome="More records become visible.",
        reason="The requested record is not visible in the current viewport.",
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
            retention=ArtifactRetention.RUN,
        ),
    )
    model = StubPlannerModel(_proposal_output())

    _planner(model).plan(_planning_request(history=(refused, observed, scrolled)))

    model_request = model.requests[0]
    history = _history_contexts(model_request)
    assert all("expected_outcome" not in entry for entry in history)
    assert history[0]["feedback"]["code"] == "action_not_allowed"
    assert history[1]["feedback"]["signals"] == ["text_not_landed"]
    assert history[2]["direction"] == "down"
    assert history[2]["distance"] == "short"
    assert "private literal text" not in str(model_request.messages[1].content)


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
            dispatched_at=DISPATCHED_AT,
            receipt_id="receipt-rejected",
            authorization_id="authorization-rejected",
            proposal_id=failed_proposal.proposal_id,
            action=ActionKind.FOCUS,
            disposition=DispatchDisposition.TARGET_REJECTED,
            reason="The grounded target did not accept keyboard focus.",
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

    history = _history_contexts(model.requests[0])[0]
    assert history["feedback"] == {
        "kind": "receipt",
        "disposition": "target_rejected",
        "reason": "The grounded target did not accept keyboard focus.",
    }
    assert history["grounded_target_point"]["point"] == {"x": 540, "y": 1437}


def _refusal_entry(index: int) -> PlanningHistoryEntry:
    candidate = ActionProposal(
        proposal_id=f"proposal-refused-{index}",
        source_observation_id="obs_0",
        action=ActionKind.TAP,
        target_expression=f"the Save button {index}",
        expected_outcome="The form saves.",
        reason="The form must be saved.",
    )
    return PlanningHistoryEntry(
        proposal=candidate,
        refusal=ActionRefusal(
            proposal_id=candidate.proposal_id,
            code=ActionRefusalCode.KEYBOARD_RULE,
            reason="Dismiss the visible keyboard before pointer actions.",
        ),
    )


def _cached_prefix(request: ModelRequest) -> tuple[str, ...]:
    blocks: list[str] = []
    for message in request.messages:
        blocks.extend(
            content.text for content in message.content if isinstance(content, TextModelContent)
        )
        if message.cache_prefix_end:
            return tuple(blocks)
    raise AssertionError("The planner request declares no cache prefix.")


def test_history_stays_bounded_and_outside_the_cached_prefix() -> None:
    history = tuple(_refusal_entry(index) for index in range(110))
    first_model = StubPlannerModel(_proposal_output())
    second_model = StubPlannerModel(_proposal_output())

    _planner(first_model).plan(_planning_request(history=history))
    _planner(second_model).plan(_planning_request(history=(*history, _refusal_entry(110))))

    first, second = first_model.requests[0], second_model.requests[0]
    assert len(_history_contexts(first)) == RECENT_PLANNING_HISTORY_LIMIT + 1
    # History slides every step, so no part of it can sit inside the cached prefix.
    assert _history_contexts(first) != _history_contexts(second)
    assert _cached_prefix(first) == _cached_prefix(second)
