"""Safe Device Agent handling for unusable planner responses."""

from yewo.agent import AgentTask, AgentTerminalReason
from yewo.agent.actuation import ActionKind
from yewo.agent.planning.planner import TaskPlanner
from yewo.agent.planning.evidence import PlannerDecisionMetadata, PlannerDecisionSource
from yewo.models import ModelCallDisposition

from agent.planning.support import StubPlannerModel
from agent.actuation.support import proposal
from agent.runtime.test_agent import (
    NeverInterrupted,
    ObservationSequence,
    RuntimeDevice,
    _mode_policy,
    _observation,
    _runtime,
)


class InvalidRetainedPlanner:
    """Return one reused device action without its transition evidence."""

    def required_model_requests(self, request) -> int:
        del request
        return 0

    def plan(self, request):
        return proposal(
            ActionKind.TAP,
            source_observation_id=request.observation.observation_id,
            target_expression="the Save button",
        )

    def decision_metadata(self) -> PlannerDecisionMetadata:
        return PlannerDecisionMetadata(
            source=PlannerDecisionSource.COMPILED_PROCEDURE,
            reused_flow_id="procedure_1",
        )


def test_runtime_returns_incomplete_agent_result_for_truncated_planning(tmp_path) -> None:
    planner = TaskPlanner(
        StubPlannerModel(None, disposition=ModelCallDisposition.TRUNCATED),
        "planner-primary",
    )
    result = _runtime(
        tmp_path,
        planner,
        ObservationSequence([_observation("obs_1", "0000000000000000")]),
        RuntimeDevice(),
    ).run(
        AgentTask(
            objective="Add one record.",
            completion_conditions=("The record is visible.",),
        ),
        _mode_policy(),
        NeverInterrupted(),
    )

    assert result.terminal_reason is AgentTerminalReason.CANNOT_PROGRESS
    assert result.usage.planning_requests == 1
    assert result.usage.dispatched_actions == 0
    assert len(result.steps) == 1
    assert result.steps[0].terminal_reason is AgentTerminalReason.CANNOT_PROGRESS
    assert result.steps[0].decision.source is PlannerDecisionSource.SYSTEM
    assert result.reason == "Planning stopped: The configured planner call failed."


def test_runtime_stops_invalid_reused_provenance_before_dispatch(tmp_path) -> None:
    device = RuntimeDevice()
    result = _runtime(
        tmp_path,
        InvalidRetainedPlanner(),
        ObservationSequence([_observation("obs_1", "0000000000000000")]),
        device,
    ).run(
        AgentTask(
            objective="Save the form.",
            completion_conditions=("The form is saved.",),
        ),
        _mode_policy(),
        NeverInterrupted(),
    )

    assert result.terminal_reason is AgentTerminalReason.CANNOT_PROGRESS
    assert result.usage.dispatched_actions == 0
    assert device.calls == []
    assert result.steps[0].decision.source is PlannerDecisionSource.SYSTEM
    assert "before dispatch" in result.reason
