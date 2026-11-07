"""Render one canonical run record as JUnit XML."""

import xml.etree.ElementTree as ET

from yewo.reporting.records import (
    CanonicalRunRecord,
    CanonicalScenarioRecord,
    ScenarioAuditVerdict,
)


def render_junit(record: CanonicalRunRecord) -> bytes:
    """Return CI output with truthful failure and error semantics."""
    scenarios = record.scenarios
    synthetic = not scenarios
    failures = sum(
        scenario.audit.verdict in {ScenarioAuditVerdict.FAILED, ScenarioAuditVerdict.UNVERIFIED}
        for scenario in scenarios
    ) + int(synthetic)
    errors = sum(
        scenario.audit.verdict in {ScenarioAuditVerdict.ERROR, ScenarioAuditVerdict.ABORTED}
        for scenario in scenarios
    )
    duration = (record.completed_at - record.started_at).total_seconds()
    attributes = {
        "name": "Yewo scenarios",
        "tests": str(len(scenarios) or 1),
        "failures": str(failures),
        "errors": str(errors),
        "skipped": "0",
        "time": f"{duration:.6f}",
    }
    suites = ET.Element("testsuites", attributes)
    suite = ET.SubElement(suites, "testsuite", attributes)
    properties = ET.SubElement(suite, "properties")
    _property(properties, "yewo.run_id", record.run_id)
    _property(properties, "yewo.qa_verdict", record.qa_verdict.value)
    _property(properties, "yewo.completeness", record.completeness.value)
    _property(properties, "yewo.terminal_category", record.terminal_category.value)
    if synthetic:
        case = ET.SubElement(
            suite,
            "testcase",
            {"name": "No executable scenarios", "classname": "yewo.run", "time": "0"},
        )
        failure = ET.SubElement(
            case,
            "failure",
            {"type": "YewoUnverified", "message": "No executable scenarios were audited."},
        )
        failure.text = record.terminal_reason
    for scenario in scenarios:
        elapsed = 0.0
        if (
            scenario.execution.started_at is not None
            and scenario.execution.completed_at is not None
        ):
            elapsed = (
                scenario.execution.completed_at - scenario.execution.started_at
            ).total_seconds()
        case = ET.SubElement(
            suite,
            "testcase",
            {
                "name": scenario.name,
                "classname": f"yewo.{scenario.journey_id}.{scenario.coverage_cell_id}",
                "time": f"{elapsed:.6f}",
            },
        )
        _scenario_result(case, scenario.audit.verdict, scenario.audit.summary)
        output = ET.SubElement(case, "system-out")
        output.text = _scenario_output(scenario)
    return ET.tostring(suites, encoding="utf-8", xml_declaration=True)


def _property(parent: ET.Element, name: str, value: str) -> None:
    ET.SubElement(parent, "property", {"name": name, "value": value})


def _scenario_result(parent: ET.Element, verdict: ScenarioAuditVerdict, summary: str) -> None:
    if verdict in {ScenarioAuditVerdict.FAILED, ScenarioAuditVerdict.UNVERIFIED}:
        failure = ET.SubElement(
            parent,
            "failure",
            {"type": f"Yewo{verdict.value.title()}", "message": summary},
        )
        failure.text = summary
    elif verdict in {ScenarioAuditVerdict.ERROR, ScenarioAuditVerdict.ABORTED}:
        error = ET.SubElement(
            parent,
            "error",
            {"type": f"Yewo{verdict.value.title()}", "message": summary},
        )
        error.text = summary


def _scenario_output(scenario: CanonicalScenarioRecord) -> str:
    lines = [
        f"execution={scenario.execution.classification.value}",
        f"execution_reason={scenario.execution.reason}",
    ]
    lines.extend(
        f"outcome={item.obligation_id} status={item.status.value} reason={item.reason}"
        for item in scenario.audit.obligations
    )
    lines.extend(
        f"interaction={item.interaction_id} action={item.action} "
        f"point=({item.point.x},{item.point.y}) source={item.geometry_source.value} "
        f"dispatch={item.dispatch_disposition} outcome={item.outcome_disposition or 'unavailable'}"
        for item in scenario.interactions
    )
    return "\n".join(lines)
