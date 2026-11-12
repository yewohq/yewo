"""Safe Device Agent handling for unusable planner responses."""

import pytest

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


def test_a_planner_that_returns_nothing_costs_a_step_not_the_run(tmp_path) -> None:
    # A provider that answered nothing has said nothing about the task. Concluding from its
    # silence threw away a run that was one action from finishing.
    planner = TaskPlanner(
        StubPlannerModel(None, disposition=ModelCallDisposition.TRUNCATED),
        "planner-primary",
    )
    result = _runtime(
        tmp_path,
        planner,
        ObservationSequence(
            [_observation(f"obs_{index}", "0000000000000000") for index in range(1, 20)]
        ),
        RuntimeDevice(),
    ).run(
        AgentTask(
            objective="Add one record.",
            completion_conditions=("The record is visible.",),
        ),
        _mode_policy(),
        NeverInterrupted(),
    )

    assert result.terminal_reason is not AgentTerminalReason.CANNOT_PROGRESS
    assert result.usage.planning_requests > 1
    assert len(result.steps) > 1
    assert all(step.terminal_reason is None for step in result.steps)
    assert result.steps[0].decision.source is PlannerDecisionSource.SYSTEM
    assert "used all available steps" in result.reason


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


def test_an_answered_but_unusable_plan_still_stops_the_run(tmp_path) -> None:
    # A planner that answered and chose something invalid has described the task, unlike one that
    # returned nothing. Only the second is the provider's silence.
    planner = TaskPlanner(
        StubPlannerModel({"not": "an action proposal"}),
        "planner-primary",
    )

    result = _runtime(
        tmp_path,
        planner,
        ObservationSequence(
            [_observation(f"obs_{index}", "0000000000000000") for index in range(1, 6)]
        ),
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
    assert result.reason.startswith("Planning stopped:")


@pytest.mark.parametrize(
    "disposition",
    [
        ModelCallDisposition.EMPTY,
        ModelCallDisposition.PROVIDER_ERROR,
        ModelCallDisposition.MALFORMED,
        ModelCallDisposition.REASONING_EXHAUSTED,
        ModelCallDisposition.TRUNCATED,
    ],
)
def test_every_shape_of_provider_silence_costs_a_step(disposition, tmp_path) -> None:
    # Each of these leaves the step without a decision. None of them says the task is finished.
    planner = TaskPlanner(StubPlannerModel(None, disposition=disposition), "planner-primary")

    result = _runtime(
        tmp_path,
        planner,
        ObservationSequence(
            [_observation(f"obs_{index}", "0000000000000000") for index in range(1, 20)]
        ),
        RuntimeDevice(),
    ).run(
        AgentTask(
            objective="Add one record.",
            completion_conditions=("The record is visible.",),
        ),
        _mode_policy(),
        NeverInterrupted(),
    )

    assert result.terminal_reason is not AgentTerminalReason.CANNOT_PROGRESS


def test_a_completed_call_with_no_structured_output_costs_a_step(tmp_path) -> None:
    # The provider reported success and returned nothing usable, which is silence wearing a
    # completed disposition.
    planner = TaskPlanner(StubPlannerModel(None), "planner-primary")

    result = _runtime(
        tmp_path,
        planner,
        ObservationSequence(
            [_observation(f"obs_{index}", "0000000000000000") for index in range(1, 20)]
        ),
        RuntimeDevice(),
    ).run(
        AgentTask(
            objective="Add one record.",
            completion_conditions=("The record is visible.",),
        ),
        _mode_policy(),
        NeverInterrupted(),
    )

    assert result.terminal_reason is not AgentTerminalReason.CANNOT_PROGRESS
