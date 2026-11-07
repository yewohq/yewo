"""Tests for the canonical report source shared by every renderer."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from pydantic import ValidationError
import pytest

from yewo.reporting import (
    AuditObligationRecord,
    AuditObligationStatus,
    CanonicalEvidenceKind,
    CanonicalEvidenceReference,
    CanonicalRunRecord,
    CanonicalScenarioRecord,
    ScenarioAuditRecord,
    ScenarioAuditVerdict,
    ScenarioExecutionClassification,
    ScenarioExecutionRecord,
)
from yewo.runs import RunMode, RunStage, RunTerminalCategory


STARTED_AT = datetime(2026, 11, 6, 9, tzinfo=timezone.utc)
COMPLETED_AT = STARTED_AT + timedelta(seconds=20)


def _evidence(
    *,
    artifact_id: str = "screenshot_1",
    scenario_id: str | None = "scenario_1",
) -> CanonicalEvidenceReference:
    return CanonicalEvidenceReference(
        artifact_id=artifact_id,
        kind=CanonicalEvidenceKind.SCREENSHOT,
        path=Path(f"screenshots/{artifact_id}.png"),
        label="Final screen",
        scenario_id=scenario_id,
    )


def _scenario(
    *,
    execution_classification: ScenarioExecutionClassification = (
        ScenarioExecutionClassification.COMPLETED
    ),
    verdict: ScenarioAuditVerdict = ScenarioAuditVerdict.PASSED,
    obligation_status: AuditObligationStatus = AuditObligationStatus.SATISFIED,
) -> CanonicalScenarioRecord:
    evidence_ids = ("screenshot_1",) if obligation_status is AuditObligationStatus.SATISFIED else ()
    return CanonicalScenarioRecord(
        scenario_id="scenario_1",
        journey_id="journey_1",
        coverage_cell_id="functional.happy_path.0",
        name="Valid sign-in",
        objective="Sign in with valid credentials.",
        expected_outcomes=("The account home screen is visible.",),
        execution=ScenarioExecutionRecord(
            classification=execution_classification,
            reason="Scenario execution reached a terminal state.",
            started_at=STARTED_AT,
            completed_at=COMPLETED_AT,
            action_count=3,
            model_call_count=2,
            evidence_ids=("screenshot_1",),
        ),
        audit=ScenarioAuditRecord(
            verdict=verdict,
            summary="The final screen proves the expected outcome.",
            obligations=(
                AuditObligationRecord(
                    obligation_id="home_visible",
                    description="The account home screen is visible.",
                    status=obligation_status,
                    evidence_ids=evidence_ids,
                    reason="The accepted final screenshot shows the account home screen.",
                ),
            ),
        ),
    )


def _record(
    *,
    scenarios: tuple[CanonicalScenarioRecord, ...] | None = None,
    evidence: tuple[CanonicalEvidenceReference, ...] | None = None,
) -> CanonicalRunRecord:
    return CanonicalRunRecord(
        run_id="run_1",
        mode=RunMode.SCENARIO,
        started_at=STARTED_AT,
        completed_at=COMPLETED_AT,
        terminal_category=RunTerminalCategory.COMPLETED,
        terminal_stage=RunStage.FINISHED,
        terminal_reason="The scenario workflow completed.",
        partial_evidence_available=True,
        scenarios=scenarios if scenarios is not None else (_scenario(),),
        evidence=evidence if evidence is not None else (_evidence(),),
    )


def test_record_accepts_one_evidence_grounded_scenario() -> None:
    record = _record()

    assert record.schema_version == 1
    assert record.scenarios[0].audit.verdict is ScenarioAuditVerdict.PASSED
    assert record.evidence[0].path == Path("screenshots/screenshot_1.png")


def test_completed_scenario_run_rejects_zero_scenarios() -> None:
    with pytest.raises(ValidationError, match="must contain an executable scenario"):
        _record(scenarios=())


def test_passed_scenario_requires_evidence_for_every_obligation() -> None:
    with pytest.raises(ValidationError, match="requires evidence"):
        AuditObligationRecord(
            obligation_id="home_visible",
            description="The home screen is visible.",
            status=AuditObligationStatus.SATISFIED,
            evidence_ids=(),
            reason="No evidence was retained.",
        )


def test_scenario_rejects_evidence_outside_the_canonical_registry() -> None:
    with pytest.raises(ValidationError, match="outside the canonical registry"):
        _record(evidence=())


def test_interrupted_execution_requires_an_aborted_verdict() -> None:
    with pytest.raises(ValidationError, match="must agree"):
        _scenario(
            execution_classification=ScenarioExecutionClassification.INTERRUPTED,
            verdict=ScenarioAuditVerdict.UNVERIFIED,
            obligation_status=AuditObligationStatus.UNVERIFIED,
        )


def test_evidence_path_rejects_parent_traversal() -> None:
    with pytest.raises(ValidationError, match="safe relative path"):
        CanonicalEvidenceReference(
            artifact_id="screenshot_1",
            kind=CanonicalEvidenceKind.SCREENSHOT,
            path=Path("../screenshot.png"),
            label="Unsafe screenshot",
        )
