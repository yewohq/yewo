"""A model writes these reasons, so their length is not ours to bound."""

from yewo.reporting import (
    ScenarioExecutionClassification,
    ScenarioExecutionRecord,
)
from yewo.reporting.audit_records import (
    AuditObligationRecord,
    AuditObligationStatus,
    ScenarioAuditRecord,
    ScenarioAuditVerdict,
)
from yewo.runs.records import RunStage, RunTerminal, RunTerminalCategory

VERBOSE = ("The app never settled. " * 400).strip()


def test_verbose_execution_reason_is_kept() -> None:
    """Capping this failed a run for being wordy, which says nothing about correctness.

    A provider's completion limit already bounds how much a model can return, so a second
    bound here only converted verbosity into a lost verdict.
    """
    record = ScenarioExecutionRecord(
        classification=ScenarioExecutionClassification.APPLICATION_FAILURE,
        reason=VERBOSE,
    )

    assert record.reason == VERBOSE


def test_verbose_audit_summary_is_kept() -> None:
    """The audit record carries the same model prose one layer on."""
    audit = ScenarioAuditRecord(
        verdict=ScenarioAuditVerdict.FAILED,
        summary=VERBOSE,
        obligations=(
            AuditObligationRecord(
                obligation_id="outcome_1",
                description="The requested state change is visible.",
                status=AuditObligationStatus.UNMET,
                reason=VERBOSE,
            ),
        ),
    )

    assert audit.summary == VERBOSE
    assert audit.obligations[0].reason == VERBOSE


def test_verbose_terminal_reason_is_kept() -> None:
    """The run terminal is the last place this prose lands."""
    terminal = RunTerminal(
        run_id="a" * 32,
        category=RunTerminalCategory.INFRASTRUCTURE_FAILURE,
        stage=RunStage.EXECUTION,
        reason=VERBOSE,
        completed_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        partial_evidence_available=False,
    )

    assert terminal.reason == VERBOSE
