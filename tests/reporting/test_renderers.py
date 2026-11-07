"""Cross-format contracts for canonical reporting."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from yewo.reporting import (
    AuditFindingRecord,
    AuditFindingRegionRecord,
    AuditFindingVerificationKind,
    AuditObligationRecord,
    AuditObligationStatus,
    CanonicalEvidenceKind,
    CanonicalEvidenceReference,
    CanonicalInteractionRecord,
    CanonicalPoint,
    CanonicalRegion,
    CanonicalRunRecord,
    CanonicalScenarioRecord,
    InteractionGeometrySource,
    ScenarioAuditRecord,
    ScenarioAuditVerdict,
    ScenarioExecutionClassification,
    ScenarioExecutionRecord,
)
from yewo.reporting.bundle import write_report_bundle
from yewo.reporting import ReportOutputDisposition, ReportOutputFormat
from yewo.reporting import bundle as report_bundle
from yewo.reporting.html import render_html
from yewo.reporting.junit import render_junit
from yewo.reporting.markdown import render_markdown
from yewo.runs import RunMode, RunStage, RunTerminalCategory


STARTED_AT = datetime(2026, 11, 6, 9, tzinfo=timezone.utc)


def _embedded_report_data(html: str) -> dict[str, Any]:
    marker = '<script id="report-data" type="application/json">'
    serialized = html.split(marker, maxsplit=1)[1].split("</script>", maxsplit=1)[0]
    return json.loads(serialized)


def _record(verdict: ScenarioAuditVerdict = ScenarioAuditVerdict.FAILED) -> CanonicalRunRecord:
    completed_at = STARTED_AT + timedelta(seconds=5)
    status = (
        AuditObligationStatus.UNMET
        if verdict is ScenarioAuditVerdict.FAILED
        else AuditObligationStatus.SATISFIED
    )
    findings = (
        (
            AuditFindingRecord(
                finding_id="finding_layout",
                claim="The basket visibly overflows.",
                reason="The auditor located a directly visible defect.",
                verification_kind=AuditFindingVerificationKind.VISUAL,
                regions=(
                    AuditFindingRegionRecord(
                        observation_id="observation_1",
                        label="Basket overflow",
                        left=100,
                        top=100,
                        right=900,
                        bottom=900,
                        screenshot_evidence_id="screen_1",
                    ),
                ),
            ),
        )
        if status is AuditObligationStatus.UNMET
        else ()
    )
    screenshot = CanonicalEvidenceReference(
        artifact_id="screen_1",
        kind=CanonicalEvidenceKind.SCREENSHOT,
        path=Path("scenarios/scenario_1/screenshots/screen.png"),
        label="Before tap",
        scenario_id="scenario_1",
    )
    execution = CanonicalEvidenceReference(
        artifact_id="execution_1",
        kind=CanonicalEvidenceKind.EXECUTION_TRACE,
        path=Path("scenarios/scenario_1/execution.json"),
        label="Execution",
        scenario_id="scenario_1",
    )
    scenario = CanonicalScenarioRecord(
        scenario_id="scenario_1",
        journey_id="journey_1",
        coverage_cell_id="ui.visual.0",
        name="Basket <layout>",
        objective="Check the basket.",
        expected_outcomes=("The basket has no overflow.",),
        execution=ScenarioExecutionRecord(
            classification=ScenarioExecutionClassification.COMPLETED,
            reason="Execution completed.",
            started_at=STARTED_AT,
            completed_at=completed_at,
            action_count=1,
            evidence_ids=("execution_1",),
        ),
        audit=ScenarioAuditRecord(
            verdict=verdict,
            summary="A visible RenderFlex warning was found.",
            obligations=(
                AuditObligationRecord(
                    obligation_id="layout",
                    description="The basket has no overflow.",
                    status=status,
                    evidence_ids=("execution_1",),
                    reason="The screen shows a RenderFlex warning.",
                    findings=findings,
                ),
            ),
        ),
        interactions=(
            CanonicalInteractionRecord(
                interaction_id="step_1",
                action="tap",
                source_observation_id="observation_1",
                screenshot_evidence_id="screen_1",
                screenshot_width_px=400,
                screenshot_height_px=800,
                point=CanonicalPoint(x=200, y=400),
                region=CanonicalRegion(left=150, top=350, right=250, bottom=450),
                geometry_source=InteractionGeometrySource.SCREEN_ANALYSIS,
                target="Add <cake>",
                expected_outcome="The item enters the basket.",
                dispatch_disposition="dispatched",
                outcome_disposition="observed",
                settled_observation_id="observation_2",
            ),
        ),
    )
    return CanonicalRunRecord(
        run_id="run_1",
        mode=RunMode.SCENARIO,
        started_at=STARTED_AT,
        completed_at=completed_at,
        terminal_category=RunTerminalCategory.APPLICATION_FAILURE,
        terminal_stage=RunStage.FINISHED,
        terminal_reason="An application defect was found.",
        partial_evidence_available=True,
        scenarios=(scenario,),
        evidence=(screenshot, execution),
    )


def test_every_format_preserves_the_same_failure() -> None:
    record = _record()

    markdown = render_markdown(record)
    html = render_html(record)
    junit = ET.fromstring(render_junit(record))
    report_data = _embedded_report_data(html)
    canonical_record = report_data["record"]

    assert "FAILED — Basket <layout>" in markdown
    assert '<div id="report-root">' in html
    assert '<header class="top">' in html
    assert '<section class="tiles" aria-label="Run summary">' in html
    assert '<template id="scenario-template">' in html
    assert '<template id="interaction-template">' in html
    assert '<template id="issue-template">' in html
    assert '<script id="report-data" type="application/json">' in html
    assert canonical_record["scenarios"][0]["name"] == "Basket <layout>"
    assert canonical_record["scenarios"][0]["interactions"][0]["target"] == "Add <cake>"
    assert (
        canonical_record["scenarios"][0]["audit"]["obligations"][0]["findings"][0]["regions"][0][
            "screenshot_evidence_id"
        ]
        == "screen_1"
    )
    assert canonical_record["evidence"][0]["path"] == (
        "scenarios/scenario_1/screenshots/screen.png"
    )
    assert "width:30px;height:30px;border:2px solid var(--clay)" in html
    assert "background:rgba(201,120,74,.3)" in html
    assert "rgba(255,255,255" not in html
    assert "Targeting" not in html
    assert "YOLO" not in html
    assert junit.attrib["failures"] == "1"
    assert junit.find(".//failure") is not None


def test_html_escapes_embedded_data_that_could_close_the_script() -> None:
    record = _record()
    scenario = record.scenarios[0].model_copy(update={"objective": "Check </script> safely."})

    html = render_html(record.model_copy(update={"scenarios": (scenario,)}))
    report_data = _embedded_report_data(html)

    assert report_data["record"]["scenarios"][0]["objective"] == "Check </script> safely."
    serialized_data = html.split('<script id="report-data"', maxsplit=1)[1].split(
        "</script>", maxsplit=1
    )[0]
    assert "</script>" not in serialized_data.lower()


def test_report_bundle_writes_stable_paths(tmp_path) -> None:
    outputs = write_report_bundle(tmp_path, _record())

    assert {output.path.as_posix() for output in outputs} == {
        "reports/report.md",
        "reports/report.html",
        "reports/junit.xml",
    }
    assert all((tmp_path / output.path).is_file() for output in outputs)


def test_zero_scenarios_is_a_junit_failure() -> None:
    record = _record().model_copy(
        update={
            "terminal_category": RunTerminalCategory.INCOMPLETE,
            "scenarios": (),
            "evidence": (),
        }
    )

    junit = ET.fromstring(render_junit(record))

    assert junit.attrib["tests"] == "1"
    assert junit.attrib["failures"] == "1"
    assert junit.find(".//failure") is not None


def test_one_renderer_failure_does_not_discard_other_reports(tmp_path, monkeypatch) -> None:
    def fail_html(unused_record):
        raise ValueError("The configured renderer failed.")

    monkeypatch.setattr(report_bundle, "render_html", fail_html)

    outputs = write_report_bundle(tmp_path, _record())

    by_format = {output.format: output for output in outputs}
    assert by_format[ReportOutputFormat.HTML].disposition is ReportOutputDisposition.FAILED
    assert by_format[ReportOutputFormat.MARKDOWN].disposition is ReportOutputDisposition.WRITTEN
    assert by_format[ReportOutputFormat.JUNIT].disposition is ReportOutputDisposition.WRITTEN
    assert not (tmp_path / "reports/report.html").exists()
    assert (tmp_path / "reports/report.md").is_file()
    assert (tmp_path / "reports/junit.xml").is_file()
