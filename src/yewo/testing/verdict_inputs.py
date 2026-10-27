"""The replayable inputs passed to the deterministic scenario judge."""

from yewo.testing.models import ReceiptRecord, VerdictInputs


def build_verdict_inputs(
    run_result,
    stop_reason,
    qa_outputs,
    assessments,
    test_case,
    security_rules,
    all_findings,
    measurement_samples=(),
    crash_diagnostic=None,
) -> VerdictInputs:
    """Keep every fact needed to reach the scenario verdict again."""
    last_output = qa_outputs[-1] if qa_outputs else None
    return VerdictInputs(
        run_status=run_result.status,
        stop_reason=stop_reason,
        is_scenario_feasible=(
            last_output.is_scenario_feasible if last_output else True
        ),
        infeasibility_reason=(
            last_output.infeasibility_reason if last_output else None
        ),
        screens_judged=len(assessments),
        expected_outcomes=list(test_case.expected_outcomes),
        security_rules=list(security_rules),
        sensitive_findings=list(all_findings),
        required_capabilities=list(test_case.required_capabilities),
        measurements=list(measurement_samples),
        receipts=[
            ReceiptRecord(capability=receipt.capability, confirmed=receipt.confirmed)
            for receipt in run_result.capability_receipts
        ],
        postcondition_verifications=list(run_result.postcondition_verifications),
        crash_diagnostic=crash_diagnostic,
    )
