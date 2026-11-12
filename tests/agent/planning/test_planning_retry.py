"""Offline tests for bounded resampling of one rejected planner decision."""

from datetime import datetime, timezone
import json

import pytest
from pydantic import BaseModel

from agent.planning.support import planner as _planner
from agent.planning.support import planning_request as _planning_request
from agent.planning.support import proposal_output as _proposal_output
from yewo.agent.actuation import ActionKind
from yewo.agent.planning.planner import PLANNING_ATTEMPT_LIMIT, TaskPlanningError
from yewo.models import (
    ModelCallDisposition,
    ModelCallRecord,
    ModelCallUsage,
    ModelRequest,
    ModelResponse,
)


class SequencedPlannerModel:
    """Return one configured response per call."""

    def __init__(
        self,
        outputs: tuple[tuple[dict[str, object] | None, ModelCallDisposition], ...],
    ) -> None:
        self._outputs = outputs
        self.call_count = 0

    def complete(
        self,
        request: ModelRequest,
        response_model: type[BaseModel] | None = None,
        validation_context: dict[str, object] | None = None,
    ) -> ModelResponse:
        del response_model
        output, disposition = self._outputs[min(self.call_count, len(self._outputs) - 1)]
        self.call_count += 1
        completed = disposition is ModelCallDisposition.COMPLETED
        return ModelResponse(
            request_id=request.request_id,
            disposition=disposition,
            text=json.dumps(output) if completed else None,
            structured_output=output if completed else None,
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
                    disposition=disposition,
                    usage=ModelCallUsage(),
                ),
            ),
            retention=request.retention,
        )


def _unavailable_action() -> dict[str, object]:
    return _proposal_output(ActionKind.WAIT)


def _usable_proposal() -> dict[str, object]:
    return _proposal_output()


def test_resamples_one_rejected_decision_and_returns_the_recovered_proposal() -> None:
    model = SequencedPlannerModel(
        (
            (_unavailable_action(), ModelCallDisposition.COMPLETED),
            (_usable_proposal(), ModelCallDisposition.COMPLETED),
        )
    )
    request = _planning_request(available_actions=frozenset({ActionKind.TAP}))

    proposal = _planner(model).plan(request)

    assert proposal.action is ActionKind.TAP
    assert model.call_count == 2


def test_stops_after_the_planning_attempt_limit_and_reports_the_first_rejection() -> None:
    model = SequencedPlannerModel(((_unavailable_action(), ModelCallDisposition.COMPLETED),))
    request = _planning_request(available_actions=frozenset({ActionKind.TAP}))

    with pytest.raises(TaskPlanningError, match="not available for this task"):
        _planner(model).plan(request)

    assert model.call_count == PLANNING_ATTEMPT_LIMIT


def test_a_later_rejection_for_a_different_reason_does_not_replace_the_reported_cause() -> None:
    undeclared_value = _proposal_output(
        ActionKind.TYPE,
        text={"source": "declared_value", "value_name": "unknown_password"},
        expected_outcome="The password field contains the declared value.",
        reason="The password field must be completed.",
    )
    model = SequencedPlannerModel(
        (
            (undeclared_value, ModelCallDisposition.COMPLETED),
            (_unavailable_action(), ModelCallDisposition.COMPLETED),
        )
    )
    request = _planning_request(available_actions=frozenset({ActionKind.TAP, ActionKind.TYPE}))

    with pytest.raises(TaskPlanningError, match="unavailable declared value"):
        _planner(model).plan(request)

    assert model.call_count == PLANNING_ATTEMPT_LIMIT


def test_does_not_resample_a_transport_failure_the_model_client_already_retried() -> None:
    model = SequencedPlannerModel(((None, ModelCallDisposition.PROVIDER_ERROR),))

    with pytest.raises(TaskPlanningError):
        _planner(model).plan(_planning_request())

    assert model.call_count == 1


def test_reports_the_model_requests_one_plan_operation_actually_consumed() -> None:
    model = SequencedPlannerModel(((_usable_proposal(), ModelCallDisposition.COMPLETED),))
    task_planner = _planner(model)

    task_planner.plan(_planning_request())

    assert task_planner.required_model_requests(_planning_request()) == PLANNING_ATTEMPT_LIMIT
    assert task_planner.last_planning_model_request_count() == 1
