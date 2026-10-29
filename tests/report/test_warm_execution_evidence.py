"""Warm route failures and costs remain explicit scenario evidence."""

from yewo.agent.run.record import CallRow
from yewo.testing.models import GeneratedTestCase, ScenarioDraft, ScenarioReport
from yewo.testing.run_usage import record_scenario_execution_cost
from yewo.testing.steps import ActionStep
from yewo.testing.taxonomy import Category, CoverageCell, Priority, ScenarioType
from yewo.testing.warm_execution import (
    WarmExecutionTerminal,
    unavailable_warm_execution,
)
from yewo.testing.warm_execution_reporting import (
    failed_warm_execution_report,
    prepare_scenario_route,
)


def _scenario() -> GeneratedTestCase:
    return GeneratedTestCase.of(
        CoverageCell(
            Category.FUNCTIONAL,
            ScenarioType.HAPPY_PATH,
            Priority.P0,
        ),
        ScenarioDraft(
            scenario_name="Create a schedule",
            route_memory_ids=["route-form"],
            steps=[ActionStep(instruction="Open the schedule editor.")],
            expected_outcomes=["The schedule exists."],
        ),
    )


def _call(generation_id: str, cost: float | None) -> CallRow:
    return CallRow(
        component="planner",
        requested_model="model",
        served_model="model",
        generation_id=generation_id,
        attempt=1,
        finish_reason="stop",
        prompt_tokens=1,
        completion_tokens=1,
        cached_tokens=0,
        reasoning_tokens=0,
        cost=cost,
        latency_ms=10,
    )


def test_missing_live_route_boundary_is_unverified_evidence():
    scenario = _scenario()

    warm_execution = prepare_scenario_route(scenario, None, object(), object())
    report = failed_warm_execution_report(
        scenario,
        warm_execution,
        duration=0.25,
        measurements=(),
        lifecycle_operations=(),
    )

    assert warm_execution.terminal is WarmExecutionTerminal.UNVERIFIED
    assert report.status == "UNVERIFIED"
    assert report.cause == "evidence"
    assert report.warm_execution == warm_execution


def test_semantic_scenario_cost_is_separate_from_route_cost():
    scenario = _scenario()
    warm_execution = unavailable_warm_execution(scenario, "No live route.")
    report = ScenarioReport(
        scenario_uuid=scenario.scenario_uuid,
        test_id=scenario.test_id,
        scenario_name=scenario.scenario_name,
        cell=scenario.cell,
        status="UNVERIFIED",
        duration=1.25,
        warm_execution=warm_execution,
    )

    record_scenario_execution_cost(
        report,
        [_call("known", 0.0015), _call("unknown", None)],
    )

    scenario_cost = report.warm_execution.costs.scenario
    assert scenario_cost.duration_ms == 1250
    assert scenario_cost.model_calls == 2
    assert scenario_cost.cost_micro_usd == 1500
    assert scenario_cost.cost_unknown_calls == 1
    assert report.warm_execution.costs.route_reuse.model_calls == 0
