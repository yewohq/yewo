"""Top-level scenario verdict and completeness aggregation."""

import pytest

from yewo.reporting import (
    AuditObligationRecord,
    AuditObligationStatus,
    ScenarioAuditRecord,
    ScenarioAuditVerdict,
    ScenarioExecutionClassification,
    ScenarioExecutionRecord,
)
from yewo.runs import RunTerminalCategory
from yewo.runs.errors import RunWorkflowInfrastructureError
from yewo.scenarios.coordinator import ScenarioCoordinationResult
from yewo.scenarios.workflow_outcome import derive_workflow_outcome


def _execution(
    scenario_id: str,
    classification: ScenarioExecutionClassification,
    *,
    restoration_failure_reason: str | None = None,
) -> ScenarioCoordinationResult:
    return ScenarioCoordinationResult(
        scenario_id=scenario_id,
        execution=ScenarioExecutionRecord(
            classification=classification,
            reason="The configured execution classification was observed.",
            restoration_failure_reason=restoration_failure_reason,
        ),
    )


def _audit(
    verdict: ScenarioAuditVerdict,
    status: AuditObligationStatus,
) -> ScenarioAuditRecord:
    return ScenarioAuditRecord(
        verdict=verdict,
        summary="The configured audit verdict was observed.",
        obligations=(
            AuditObligationRecord(
                obligation_id="outcome_1",
                description="The visible layout remains usable.",
                status=status,
                evidence_ids=("execution_1",)
                if status is not AuditObligationStatus.UNVERIFIED
                else (),
                reason="The configured evidence produced this result.",
            ),
        ),
    )


def test_proven_application_failure_is_not_hidden_by_incomplete_work() -> None:
    executions = (
        _execution("failed", ScenarioExecutionClassification.COMPLETED),
        _execution(
            "incomplete",
            ScenarioExecutionClassification.INSUFFICIENT_DISCOVERY,
        ),
    )
    audits = {
        "failed": _audit(ScenarioAuditVerdict.FAILED, AuditObligationStatus.UNMET),
        "incomplete": _audit(
            ScenarioAuditVerdict.UNVERIFIED,
            AuditObligationStatus.UNVERIFIED,
        ),
    }

    outcome = derive_workflow_outcome(executions, audits)

    assert outcome.category is RunTerminalCategory.APPLICATION_FAILURE


def test_application_failure_survives_a_restoration_failure() -> None:
    executions = (
        _execution(
            "failed",
            ScenarioExecutionClassification.APPLICATION_FAILURE,
            restoration_failure_reason="Device-state restoration failed.",
        ),
    )
    audits = {
        "failed": _audit(ScenarioAuditVerdict.FAILED, AuditObligationStatus.UNMET),
    }

    outcome = derive_workflow_outcome(executions, audits)

    assert outcome.category is RunTerminalCategory.APPLICATION_FAILURE
    assert "restoration also failed" in outcome.reason
    assert "Device-state restoration failed." in outcome.reason


def test_restoration_failure_preserves_its_reason() -> None:
    executions = (
        _execution(
            "restoration",
            ScenarioExecutionClassification.COMPLETED,
            restoration_failure_reason="The original network state was not restored.",
        ),
    )
    audits = {
        "restoration": _audit(
            ScenarioAuditVerdict.PASSED,
            AuditObligationStatus.SATISFIED,
        ),
    }

    with pytest.raises(
        RunWorkflowInfrastructureError,
        match="The original network state was not restored",
    ):
        derive_workflow_outcome(executions, audits)


def test_audit_error_preserves_its_summary() -> None:
    executions = (_execution("audit", ScenarioExecutionClassification.COMPLETED),)
    audit = _audit(ScenarioAuditVerdict.ERROR, AuditObligationStatus.UNVERIFIED).model_copy(
        update={"summary": "The audit response could not be validated."}
    )

    with pytest.raises(
        RunWorkflowInfrastructureError,
        match="The audit response could not be validated",
    ):
        derive_workflow_outcome(executions, {"audit": audit})


def test_execution_and_audit_scenario_ids_must_match() -> None:
    executions = (_execution("execution", ScenarioExecutionClassification.COMPLETED),)
    audits = {
        "different": _audit(
            ScenarioAuditVerdict.PASSED,
            AuditObligationStatus.SATISFIED,
        )
    }

    with pytest.raises(RunWorkflowInfrastructureError, match="same scenarios"):
        derive_workflow_outcome(executions, audits)
