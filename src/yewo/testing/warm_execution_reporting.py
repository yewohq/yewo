"""QA-runner reporting around scenario-local warm execution."""

from collections.abc import Sequence

from yewo.errors import Cause
from yewo.llm.failures import reportable
from yewo.metrics import Measurement
from yewo.testing.lifecycle import LifecycleOperation
from yewo.testing.models import GeneratedTestCase, ReportStatus, ScenarioReport
from yewo.testing.warm_execution import (
    ScenarioRoutePreparer,
    ScenarioWarmExecution,
    WarmExecutionTerminal,
    unavailable_warm_execution,
)


def prepare_scenario_route(
    test_case: GeneratedTestCase,
    route_preparer: ScenarioRoutePreparer | None,
    device_control: object,
    perception: object,
) -> ScenarioWarmExecution:
    if route_preparer is None:
        return unavailable_warm_execution(
            test_case,
            "The scenario requires RouteMemory, but the live runner "
            "has no compatible route execution boundary.",
        )
    try:
        return route_preparer.prepare(test_case, device_control, perception)
    except Exception as exc:  # noqa: BLE001 - live boundary
        return unavailable_warm_execution(
            test_case,
            "Warm route execution failed at its live boundary: " + reportable(exc),
        )


def failed_warm_execution_report(
    test_case: GeneratedTestCase,
    warm_execution: ScenarioWarmExecution,
    *,
    duration: float,
    measurements: Sequence[Measurement],
    lifecycle_operations: Sequence[LifecycleOperation],
) -> ScenarioReport:
    return ScenarioReport(
        scenario_uuid=test_case.scenario_uuid,
        test_id=test_case.test_id,
        scenario_name=test_case.scenario_name,
        cell=test_case.cell,
        status=(
            ReportStatus.ABORTED
            if warm_execution.terminal is WarmExecutionTerminal.INTERRUPTED
            else ReportStatus.UNVERIFIED
        ),
        cause=Cause.EVIDENCE.value,
        duration=round(duration, 2),
        failure_reason=warm_execution.reason,
        measurements=list(measurements),
        lifecycle_operations=list(lifecycle_operations),
        warm_execution=warm_execution,
    )
