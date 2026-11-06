"""Deterministic task-planner test doubles and request builders."""

from datetime import datetime, timezone
from itertools import chain, count
import json
from typing import Any

from pydantic import BaseModel

from yewo.agent.actuation import ActionKind
from yewo.agent.planning.planner import (
    TaskPlanner,
)
from yewo.agent.planning.records import (
    PlanningRequest,
)
from yewo.models import (
    ImageModelContent,
    ModelClient,
    ModelCallDisposition,
    ModelCallRecord,
    ModelCallUsage,
    ModelRequest,
    ModelResponse,
    TextModelContent,
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
        self.validation_contexts: list[dict[str, object] | None] = []

    def complete(
        self,
        request: ModelRequest,
        response_model: type[BaseModel] | None = None,
        validation_context: dict[str, object] | None = None,
    ) -> ModelResponse:
        self.requests.append(request)
        self.response_models.append(response_model)
        self.validation_contexts.append(validation_context)
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
                    role=request.role,
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
        "objective": "Add one record.",
        "known_facts": ("The record list is visible.",),
        "open_work": ("Open the record form.",),
        "completion_conditions": ("The new record appears in the list.",),
        "declared_value_names": ("declared_secret",),
        "available_actions": frozenset({ActionKind.TAP, ActionKind.TYPE, ActionKind.WAIT}),
    }
    values.update(updates)
    return PlanningRequest.model_validate(values)


def proposal_output(action: ActionKind = ActionKind.TAP, **updates: object) -> dict[str, object]:
    values: dict[str, object] = {
        "action": action.value,
    }
    if action is ActionKind.TAP:
        values.update(
            {
                "target_expression": "the Create record control",
                "expected_outcome": "The record form opens.",
                "reason": "The visible Create record control opens the required form.",
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


def planner(model: ModelClient) -> TaskPlanner:
    identifiers = chain.from_iterable(
        (f"proposal-{attempt}", f"model-request-{attempt}") for attempt in count(1)
    )
    return TaskPlanner(model, "planner-primary", identifier_factory=lambda: next(identifiers))


def user_context(request: ModelRequest) -> tuple[str, ImageModelContent]:
    user_message = request.messages[2]
    text_part, image_part = user_message.content
    assert text_part.kind == "text"
    assert isinstance(image_part, ImageModelContent)
    return text_part.text, image_part


def history_contexts(request: ModelRequest) -> list[dict[str, Any]]:
    return [
        json.loads(content.text.split("\n", 1)[1])
        for message in request.messages
        for content in message.content
        if isinstance(content, TextModelContent)
        and content.text.startswith(("Earlier history summary:", "Trusted history event:"))
    ]
