"""The rows a run writes about itself, rather than about the app.

A suite reports one row per scenario. Three things that are not scenarios
also have to reach the report. A reader of a red build would otherwise look
for them in the app. They are:

- a run that was interrupted, whose remaining scenarios never ran
- a scenario whose model service would not answer
- a report Yewo could not write

Each carries `cell="infrastructure"`, which is what says the row is about
Yewo. Each also carries a cause, so the reader is pointed at the provider,
the device or Yewo rather than at the app.
"""

import logging
from typing import List, Optional

from yewo.errors import Cause, cause_of
from yewo.llm.failures import reportable
from yewo.testing.models import (
    GeneratedTestCase,
    JourneyReport,
    ReportStatus,
    ScenarioReport,
)

logger = logging.getLogger(__name__)


def write_failure(
    report: JourneyReport, artifact_name: str, error: Exception
) -> None:
    """Fails the run for a report that could not be written.

    A missing artifact is not a passing suite. The run produced verdicts and
    no record of them, and nothing downstream can tell that from a run that
    was never started.
    """
    report.status = ReportStatus.FAILED
    report.failed_count += 1
    report.scenarios.append(
        ScenarioReport(
            test_id=artifact_name,
            scenario_name=artifact_name.replace("_", " ").title(),
            cell="infrastructure",
            status=ReportStatus.ERROR,
            failure_reason=f"Failed to write {artifact_name}: {error}",
        )
    )


def record_aborted(
    scenarios: List[GeneratedTestCase],
    scenario_reports: List[ScenarioReport],
    error: BaseException,
) -> int:
    """Records a failure for everything that never produced a result.

    A run can die before any scenario exists, if generation returns nothing.
    It would otherwise report zero tests and zero failures, which CI reads as
    a pass.
    """
    reason = f"Aborted due to run failure/interrupt: {reportable(error)}"
    if not scenarios:
        scenario_reports.append(
            ScenarioReport(
                test_id="scenario_generation",
                scenario_name="Scenario generation",
                cell="infrastructure",
                status=ReportStatus.ERROR,
                # Generation is a model call, so its failure is usually the
                # provider's, and a reader should not look in the app.
                cause=cause_of(error).value,
                failure_reason=reason,
            )
        )
        return 1

    recorded = 0
    for scenario in scenarios:
        if any(report.test_id == scenario.test_id for report in scenario_reports):
            continue
        scenario_reports.append(
            ScenarioReport(
                scenario_uuid=scenario.scenario_uuid,
                test_id=scenario.test_id,
                scenario_name=scenario.scenario_name,
                cell=scenario.cell,
                status=ReportStatus.ABORTED,
                cause=cause_of(error).value,
                failure_reason=reason,
            )
        )
        recorded += 1
    return recorded


def provider_error(
    scenario: GeneratedTestCase, error: BaseException, scrubber
) -> ScenarioReport:
    """The report for a scenario a model service would not answer for.

    ERROR rather than FAILED, because nothing was learned about the app. The
    cause says whose problem it is, so a reader of a red build does not go
    looking in the app for it.
    """
    reason = scrubber.scrub(reportable(error))
    logger.error("Scenario %s ended on the provider: %s", scenario.test_id, reason)
    return ScenarioReport(
        scenario_uuid=scenario.scenario_uuid,
        test_id=scenario.test_id,
        scenario_name=scenario.scenario_name,
        cell=scenario.cell,
        status=ReportStatus.ERROR,
        cause=Cause.PROVIDER.value,
        failure_reason=reason,
    )


def uncovered_cell(gap) -> ScenarioReport:
    """The row for a cell nothing could produce a scenario for.

    Written before the first scenario runs, so a reader sees the gap even if
    the run dies next.
    """
    return ScenarioReport(
        test_id=gap.cell.test_id,
        scenario_name=f"Uncovered cell {gap.cell.key}",
        cell=gap.cell.key,
        status=gap.status,
        cause=gap.cause.value,
        failure_reason=gap.reason,
    )


def usage_failure(error: Exception) -> ScenarioReport:
    """The row for usage telemetry that could not be serialized."""
    return ScenarioReport(
        test_id="usage_report",
        scenario_name="LLM usage report",
        cell="infrastructure",
        status=ReportStatus.ERROR,
        failure_reason=f"Failed to serialize LLM usage: {error}",
    )


def optional_cause(error: Optional[BaseException]) -> Optional[str]:
    """The cause of a failure, where there was one."""
    return None if error is None else cause_of(error).value
