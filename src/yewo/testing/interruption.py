"""Preserve active scenario evidence when suite execution is interrupted."""

from typing import Optional, Sequence

from yewo.testing.models import GeneratedTestCase, ScenarioReport
from yewo.testing.report import infrastructure
from yewo.testing import obligation_auditing


def record_active_scenario(
    manifest,
    scenario: Optional[GeneratedTestCase],
    qa_agent,
    error: BaseException,
    scenario_reports: list[ScenarioReport],
    usage_tracker,
    scenario_components: Sequence[str],
) -> None:
    """Commit the active walk before the remaining scenarios are aborted."""
    run_result = qa_agent._device_agent.interrupted_result
    if scenario is None or run_result is None:
        return
    report = infrastructure.interrupted_scenario(
        scenario,
        run_result,
        error,
        obligation_evidence=obligation_auditing.snapshot_evidence(
            qa_agent._active_obligation_controller
        ),
    )
    scenario_reports.append(report)
    manifest.record(
        scenario.test_id,
        report,
        calls=usage_tracker.take_calls_outside_steps(scenario_components),
    )
