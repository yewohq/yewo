"""Offline tests for the shared semantic task planner."""

from datetime import datetime, timezone
import json
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from yewo.agent.actuation import (
    ActionKind,
    ActionOutcome,
    ActionOutcomeDisposition,
    ActionOutcomeSignal,
    ActionProposal,
    ActionRefusal,
    ActionRefusalCode,
    LiteralText,
    ScrollDistance,
)
from yewo.devices.interaction import ScrollDirection
from yewo.agent.planner import (
    PlanningHistoryEntry,
    PlanningRequest,
    TASK_PLANNER_SYSTEM_PROMPT,
    TASK_PLANNER_USER_PROMPT,
    TaskPlanner,
    TaskPlanningError,
)
from yewo.artifacts import ArtifactRetention, ArtifactSensitivity
from yewo.models import (
    ImageModelContent,
    ModelCallDisposition,
    ModelCallRecord,
    ModelCallUsage,
    ModelMessageRole,
    ModelRequest,
    ModelResponse,
    ModelRole,
)

from observations.factories import build_observation


class StubPlannerModel:
    """Return one configured structured response without network access."""

    def __init__(
        self,
        output: dict[str, object] | None,
        *,
        disposition: ModelCallDisposition = ModelCallDisposition.COMPLETED,
    ) -> None:
        self.output = output
        self.disposition = disposition
        self.requests: list[ModelRequest] = []
        self.response_models: list[type[BaseModel] | None] = []

    def complete(
        self,
        request: ModelRequest,
        response_model: type[BaseModel] | None = None,
        validation_context: dict[str, object] | None = None,
    ) -> ModelResponse:
        del validation_context
        self.requests.append(request)
        self.response_models.append(response_model)
        completed = self.disposition is ModelCallDisposition.COMPLETED
        reason = None if completed else "The configured planner call failed."
        return ModelResponse(
            request_id=request.request_id,
            disposition=self.disposition,
            text=json.dumps(self.output) if completed else None,
            structured_output=self.output if completed else None,
            reason=reason,
            calls=(
                ModelCallRecord(
                    request_id=request.request_id,
                    profile_id=request.profile_id,
                    role=ModelRole.TASK_PLANNER,
                    requested_provider="offline",
                    requested_model="deterministic-planner",
                    attempt_number=1,
                    started_at=datetime(2026, 10, 30, tzinfo=timezone.utc),
                    disposition=self.disposition,
                    usage=ModelCallUsage(),
                ),
            ),
            sensitivity=request.sensitivity,
            retention=request.retention,
        )


def _planning_request(**updates: object) -> PlanningRequest:
    values: dict[str, object] = {
        "observation": build_observation(),
        "objective": "Add one account.",
        "known_facts": ("The account list is visible.",),
        "open_work": ("Open the account form.",),
        "completion_conditions": ("The new account appears in the list.",),
        "declared_value_names": ("account_password",),
        "available_actions": frozenset(
            {
                ActionKind.TAP,
                ActionKind.TYPE,
                ActionKind.WAIT,
            }
        ),
    }
    values.update(updates)
    return PlanningRequest.model_validate(values)


def _proposal_output(action: ActionKind = ActionKind.TAP, **updates: object) -> dict[str, object]:
    values: dict[str, object] = {
        "proposal_id": "proposal-1",
        "source_observation_id": "obs_1",
        "action": action.value,
    }
    if action is ActionKind.TAP:
        values.update(
            {
                "target_expression": "the Add account button",
                "expected_outcome": "The account form opens.",
            }
        )
    elif action is ActionKind.WAIT:
        values.update(
            {
                "duration_seconds": 0.5,
                "expected_outcome": "The loading indicator disappears.",
            }
        )
    else:
        values["reason"] = "No permitted useful action remains."
    values.update(updates)
    return values


def _planner(model: StubPlannerModel) -> TaskPlanner:
    identifiers = iter(("proposal-1", "model-request-1"))
    return TaskPlanner(
        model,
        "planner-primary",
        identifier_factory=lambda: next(identifiers),
    )


def _user_context(request: ModelRequest) -> tuple[str, ImageModelContent]:
    user_message = request.messages[1]
    text_part, image_part = user_message.content
    assert text_part.kind == "text"
    assert isinstance(image_part, ImageModelContent)
    return text_part.text, image_part


def test_planner_returns_the_existing_action_proposal_contract() -> None:
    model = StubPlannerModel(_proposal_output())

    proposal = _planner(model).plan(_planning_request())

    assert proposal == ActionProposal.model_validate(_proposal_output())
    assert model.response_models == [ActionProposal]
    model_request = model.requests[0]
    assert model_request.request_id == "model-request-1"
    assert model_request.profile_id == "planner-primary"
    assert model_request.role is ModelRole.TASK_PLANNER
    assert model_request.response_contract is not None
    assert model_request.response_contract.name == "action-proposal-v1"
    assert tuple(message.role for message in model_request.messages) == (
        ModelMessageRole.SYSTEM,
        ModelMessageRole.USER,
    )
    system_content = model_request.messages[0].content[0]
    assert system_content.kind == "text"
    assert system_content.text == TASK_PLANNER_SYSTEM_PROMPT
    assert "For a device action, omit reason." in system_content.text
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


def test_history_contains_trusted_feedback_without_repeating_literal_text() -> None:
    refused_proposal = ActionProposal(
        proposal_id="proposal-refused",
        source_observation_id="obs_0",
        action=ActionKind.TAP,
        target_expression="the disabled Save button",
        expected_outcome="The form saves.",
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
        target_expression="the password field",
        text=LiteralText(value="private literal text"),
        expected_outcome="The password field contains the text.",
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
    assert context["history"][0]["feedback"]["code"] == "action_not_allowed"
    assert context["history"][1]["feedback"]["signals"] == ["text_not_landed"]
    assert context["history"][2]["direction"] == "down"
    assert context["history"][2]["distance"] == "short"
    assert "private literal text" not in prompt


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
        "target_expression": "the password field",
        "text": {"source": "declared_value", "value_name": "unknown_password"},
        "expected_outcome": "The password field contains the declared value.",
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
        signals=frozenset({ActionOutcomeSignal.SCREEN_CHANGED}),
        reason="The screen changed.",
        sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
        retention=ArtifactRetention.RUN,
    )

    with pytest.raises(ValidationError, match="proposal observation"):
        PlanningHistoryEntry(proposal=proposal, outcome=outcome)
