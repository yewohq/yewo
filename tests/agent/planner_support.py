"""Deterministic task-planner test doubles and request builders."""

from datetime import datetime, timezone
import json

from pydantic import BaseModel

from yewo.agent.actuation import ActionKind
from yewo.agent.planner import PlanningRequest, TaskPlanner
from yewo.models import (
    ImageModelContent,
    ModelCallDisposition,
    ModelCallRecord,
    ModelCallUsage,
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
        return ModelResponse(
            request_id=request.request_id,
            disposition=self.disposition,
            text=json.dumps(self.output) if completed else None,
            structured_output=self.output if completed else None,
            reason=None if completed else "The configured planner call failed.",
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


def planning_request(**updates: object) -> PlanningRequest:
    values: dict[str, object] = {
        "observation": build_observation(),
        "objective": "Add one account.",
        "known_facts": ("The account list is visible.",),
        "open_work": ("Open the account form.",),
        "completion_conditions": ("The new account appears in the list.",),
        "declared_value_names": ("account_password",),
        "available_actions": frozenset({ActionKind.TAP, ActionKind.TYPE, ActionKind.WAIT}),
    }
    values.update(updates)
    return PlanningRequest.model_validate(values)


def proposal_output(action: ActionKind = ActionKind.TAP, **updates: object) -> dict[str, object]:
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
                "reason": "The visible Add account button opens the required form.",
            }
        )
    elif action is ActionKind.WAIT:
        values.update(
            {
                "duration_seconds": 0.5,
                "expected_outcome": "The loading indicator disappears.",
                "reason": "The current loading indicator may finish without intervention.",
            }
        )
    else:
        values["reason"] = "No permitted useful action remains."
    values.update(updates)
    return values


def planner(model: StubPlannerModel) -> TaskPlanner:
    identifiers = iter(("proposal-1", "model-request-1"))
    return TaskPlanner(model, "planner-primary", identifier_factory=lambda: next(identifiers))


def user_context(request: ModelRequest) -> tuple[str, ImageModelContent]:
    user_message = request.messages[1]
    text_part, image_part = user_message.content
    assert text_part.kind == "text"
    assert isinstance(image_part, ImageModelContent)
    return text_part.text, image_part
