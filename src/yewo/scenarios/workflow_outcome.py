"""Terminal outcome derivation for the scenario workflow."""

from yewo.reporting import (
    ScenarioAuditRecord,
    ScenarioAuditVerdict,
    ScenarioExecutionClassification,
)
from yewo.runs import RunTerminalCategory, RunWorkflowOutcome
from yewo.runs.errors import RunWorkflowInfrastructureError
from yewo.runs.records import RUN_TERMINAL_REASON_MAX_LENGTH
from yewo.scenarios.coordinator import ScenarioCoordinationResult


def derive_workflow_outcome(
    executions: tuple[ScenarioCoordinationResult, ...],
    audits: dict[str, ScenarioAuditRecord],
) -> RunWorkflowOutcome:
    """Derive the run outcome from scenario execution and audit results."""
    scenario_ids = {item.scenario_id for item in executions}
    if scenario_ids != set(audits):
        raise RunWorkflowInfrastructureError(
            "Scenario execution and audit results do not describe the same scenarios."
        )
    classifications = {item.execution.classification for item in executions}
    restoration_failures = tuple(
        item.execution.restoration_failure_reason
        for item in executions
        if item.execution.restoration_failure_reason is not None
    )
    application_failed = bool(
        classifications
        & {
            ScenarioExecutionClassification.APPLICATION_FAILURE,
            ScenarioExecutionClassification.APPLICATION_CRASH,
        }
    )
    if restoration_failures and application_failed:
        return RunWorkflowOutcome(
            RunTerminalCategory.APPLICATION_FAILURE,
            _reason_with_details(
                "A scenario exposed an application failure. Device-state restoration also failed:",
                restoration_failures,
            ),
        )
    if restoration_failures:
        raise RunWorkflowInfrastructureError(
            _reason_with_details("Device-state restoration failed:", restoration_failures)
        )
    infrastructure_failures = tuple(
        item.execution.reason
        for item in executions
        if item.execution.classification
        in {
            ScenarioExecutionClassification.CAPABILITY_FAILURE,
            ScenarioExecutionClassification.INFRASTRUCTURE_FAILURE,
        }
    )
    if classifications & {
        ScenarioExecutionClassification.CAPABILITY_FAILURE,
        ScenarioExecutionClassification.INFRASTRUCTURE_FAILURE,
    }:
        raise RunWorkflowInfrastructureError(
            _reason_with_details("Scenario infrastructure failed:", infrastructure_failures)
        )
    if ScenarioExecutionClassification.BUDGET_EXHAUSTED in classifications:
        return RunWorkflowOutcome(
            RunTerminalCategory.BUDGET_EXHAUSTED,
            "Scenario execution exhausted its shared agent budget.",
        )
    if ScenarioExecutionClassification.BLOCKED_BY_POLICY in classifications:
        return RunWorkflowOutcome(
            RunTerminalCategory.BLOCKED_BY_POLICY,
            "Scenario execution was blocked by policy.",
        )
    if application_failed:
        return RunWorkflowOutcome(
            RunTerminalCategory.APPLICATION_FAILURE,
            "A scenario exposed an application failure.",
        )
    verdicts = {audit.verdict for audit in audits.values()}
    if ScenarioAuditVerdict.FAILED in verdicts:
        return RunWorkflowOutcome(
            RunTerminalCategory.APPLICATION_FAILURE,
            "Independent scenario audit found an unmet QA obligation.",
        )
    if ScenarioAuditVerdict.ERROR in verdicts:
        audit_failures = tuple(
            audit.summary
            for audit in audits.values()
            if audit.verdict is ScenarioAuditVerdict.ERROR
        )
        raise RunWorkflowInfrastructureError(
            _reason_with_details("Independent scenario auditing failed:", audit_failures)
        )
    incomplete = classifications & {
        ScenarioExecutionClassification.INSUFFICIENT_DISCOVERY,
        ScenarioExecutionClassification.UNSUPPORTED_CAPABILITY,
    }
    if incomplete or not executions:
        return RunWorkflowOutcome(
            RunTerminalCategory.INCOMPLETE,
            "Scenario generation completed, but not every scenario was executable.",
        )
    if verdicts - {ScenarioAuditVerdict.PASSED}:
        return RunWorkflowOutcome(
            RunTerminalCategory.INCOMPLETE,
            "Scenario execution completed without a conclusive independent audit.",
        )
    return RunWorkflowOutcome(
        RunTerminalCategory.COMPLETED,
        "Scenario execution and independent audit completed.",
    )


def _reason_with_details(prefix: str, details: tuple[str, ...]) -> str:
    distinct_details = tuple(dict.fromkeys(details))
    reason = f"{prefix} {'; '.join(distinct_details)}"
    return reason[:RUN_TERMINAL_REASON_MAX_LENGTH]
