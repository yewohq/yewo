"""Canonical reporting for audited scenario execution."""

from yewo.reporting import (
    AuditObligationRecord,
    AuditObligationStatus,
    ScenarioAuditRecord,
    ScenarioAuditVerdict,
)
from yewo.scenarios.auditor import scenario_execution_artifact_id
from yewo.scenarios.reporting import _canonical_scenario

from scenarios.test_coordinator import _coordinator, _generated_suite


def _audit(scenario, status=AuditObligationStatus.SATISFIED):
    outcomes = tuple(
        outcome for obligation in scenario.obligations for outcome in obligation.expected_outcomes
    )
    verdict = (
        ScenarioAuditVerdict.PASSED
        if status is AuditObligationStatus.SATISFIED
        else ScenarioAuditVerdict.UNVERIFIED
    )
    return ScenarioAuditRecord(
        verdict=verdict,
        summary="Independent evidence produced the configured verdict.",
        obligations=tuple(
            AuditObligationRecord(
                obligation_id=outcome.outcome_id,
                description=outcome.statement,
                status=status,
                evidence_ids=(
                    (scenario_execution_artifact_id(scenario.scenario_id),)
                    if status is AuditObligationStatus.SATISFIED
                    else ()
                ),
                reason="Current-run evidence produced the configured status.",
            )
            for outcome in outcomes
        ),
    )


def test_completed_execution_uses_the_independent_audit(tmp_path) -> None:
    suite = _generated_suite()
    scenario = suite.scenarios[0]
    coordinator, _ = _coordinator(tmp_path, [])

    execution = coordinator.execute(suite, scenario)
    record = _canonical_scenario(scenario, execution, _audit(scenario))

    assert record.audit.verdict is ScenarioAuditVerdict.PASSED
    assert {item.status for item in record.audit.obligations} == {AuditObligationStatus.SATISFIED}
    assert record.execution.evidence_ids == execution.execution.evidence_ids


def test_audited_report_preserves_a_long_journey_outcome(tmp_path) -> None:
    suite = _generated_suite()
    scenario = suite.scenarios[0]
    long_outcome = " ".join(
        "The complete food-delivery journey remains verifiable." for _ in range(30)
    )
    first_obligation = scenario.obligations[0]
    first_outcome = first_obligation.expected_outcomes[0].model_copy(
        update={"statement": long_outcome}
    )
    scenario = scenario.model_copy(
        update={
            "objective": long_outcome,
            "obligations": (
                first_obligation.model_copy(update={"expected_outcomes": (first_outcome,)}),
            ),
        }
    )
    coordinator, _ = _coordinator(tmp_path, [])

    execution = coordinator.execute(suite.model_copy(update={"scenarios": (scenario,)}), scenario)
    record = _canonical_scenario(scenario, execution, _audit(scenario))

    assert record.objective == long_outcome
    assert record.expected_outcomes == (long_outcome,)
    assert record.audit.obligations[0].description == long_outcome
