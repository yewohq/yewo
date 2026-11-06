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
    TaskPlanningError,
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
from yewo.artifacts import ArtifactRetention, ArtifactSensitivity
from yewo.grounding import PixelPoint
from yewo.models import (
    ModelCallDisposition,
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
    task_content = model_request.messages[0].content[1]
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
    assert "done" in context["available_actions"]
    assert "cannot_progress" in context["available_actions"]
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
            exhausted_branches=("The overflow menu showed no relevant command.",),
        ),
    )

    _planner(model).plan(request)

    model_request = model.requests[0]
    prompt, unused_image = _user_context(model_request)
    context = json.loads(prompt.split("\n", 1)[1])
    task_content = model_request.messages[0].content[1]
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
        "exhausted_branches": ["The overflow menu showed no relevant command."],
        "historical_memory_facts": ["The Add control previously opened the item editor."],
        "known_procedures": [],
        "recovery_guidance": [],
        "route_attempts": [],
    }


def test_planner_receives_one_compact_current_screen_analysis() -> None:
    request = _planning_request()
    observation = request.observation
    analysis = CurrentScreenAnalysis(
        analysis_id="analysis-1",
        observation_id=observation.observation_id,
        screen_fingerprint=observation.screen_fingerprint.value,
        screenshot_sha256=observation.redacted_screenshot.sha256,
        platform_context=observation.platform_context,
        keyboard_visible=observation.keyboard_visible,
        orientation=observation.orientation,
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
                evidence="An underlined input is visible.",
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


def test_planning_rejects_analysis_for_another_observation() -> None:
    request = _planning_request()
    observation = request.observation
    analysis = CurrentScreenAnalysis(
        analysis_id="analysis-1",
        observation_id="obs-old",
        screen_fingerprint=observation.screen_fingerprint.value,
        screenshot_sha256=observation.redacted_screenshot.sha256,
        platform_context=observation.platform_context,
        keyboard_visible=observation.keyboard_visible,
        orientation=observation.orientation,
    )

    with pytest.raises(ValidationError, match="must match the planning observation"):
        PlanningRequest.model_validate(request.model_dump() | {"current_screen_analysis": analysis})


def test_planner_keeps_runtime_actions_out_of_the_stable_task_prefix() -> None:
    first_model = StubPlannerModel(_proposal_output())
    second_model = StubPlannerModel(_proposal_output())
    first_request = _planning_request(open_work=("Create the record.",))
    second_request = _planning_request(
        open_work=("Reopen the record.",),
        available_actions=frozenset(
            {ActionKind.CANNOT_PROGRESS, ActionKind.DISMISS_KEYBOARD, ActionKind.TAP}
        ),
    )

    _planner(first_model).plan(first_request)
    _planner(second_model).plan(second_request)

    first_messages = first_model.requests[0].messages
    second_messages = second_model.requests[0].messages
    assert first_messages[0] == second_messages[0]
    assert first_messages[2] != second_messages[2]


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
            sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
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
        _proposal_output(ActionKind.WAIT),
        {
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
    proposal = ActionProposal.model_validate(
        {
            "proposal_id": "proposal-1",
            "source_observation_id": "obs_1",
            **_proposal_output(),
        }
    )
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
