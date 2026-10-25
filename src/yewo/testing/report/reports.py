"""Human-readable and CI report serialization for journey runs."""

import logging
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from yewo.errors import ArtifactWriteFailed
from yewo.metrics import MeasurementRelation
from yewo.testing.models import (
    JourneyReport,
    PolicyCheckReport,
    PolicyCheckStatus,
    ReportStatus,
    ScenarioReport,
)
from yewo.testing.judge import verdicts
from yewo.utils.secrets import SecretScrubber

logger = logging.getLogger(__name__)


def safe_artifact_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return stem or "journey"


def _junit_type(scenario) -> str:
    """The type a CI reader groups on, naming the cause when there is one."""
    kind = f"Yewo{scenario.status.title()}"
    return f"{kind}.{scenario.cause}" if scenario.cause else kind


def _junit_classname(report: JourneyReport, suffix: str) -> str:
    """Qualify a test with its target without changing legacy class names."""
    parts = []
    if report.environment and report.environment.target_id:
        parts.append(safe_artifact_stem(report.environment.target_id))
    parts.extend((safe_artifact_stem(report.journey_id), suffix))
    return ".".join(parts)


def _scrub(scrubber: SecretScrubber | None, value: str | None) -> str:
    return scrubber.scrub(value) if scrubber else value or ""


def write_atomic(path: Path, content: str | bytes) -> None:
    """Writes via a temporary file, so a half-written report never appears."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        if isinstance(content, bytes):
            temporary.write_bytes(content)
        else:
            temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    except OSError as exc:
        raise ArtifactWriteFailed(f"Could not write {path}: {exc}") from exc


def _environment_lines(environment) -> list[str]:
    """Renders the run environment as report bullets."""
    if environment is None:
        return []
    model = f"`{environment.model}`"
    if environment.fast_model:
        model += f" (fast: `{environment.fast_model}`)"
    lines = [
        f"* **Yewo**: `{environment.yewo_version}` on `{environment.platform}`",
        f"* **Model**: {model}",
        f"* **Targeting**: `{environment.grounding}`",
    ]
    if environment.target_id:
        lines.append(f"* **Matrix target**: `{environment.target_id}`")
    if environment.run_id:
        lines.append(f"* **Run ID**: `{environment.run_id}`")
    if environment.execution_id:
        lines.append(f"* **Execution ID**: `{environment.execution_id}`")
    if environment.detector:
        lines.append(f"* **Detector**: `{environment.detector}`")
    return lines


def save_markdown_report(
    report: JourneyReport,
    output_dir: Path,
    session=None,
    scrubber: SecretScrubber | None = None,
    label: str | None = None,
) -> Path:
    stamp = label or time.strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / (
        f"report_{safe_artifact_stem(report.journey_id)}_{stamp}.md"
    )
    report_status = {
        ReportStatus.PASSED: "🟢 PASSED",
        ReportStatus.FAILED: "🔴 FAILED",
        ReportStatus.RUNNING: "🟠 RUNNING (checkpoint)",
        ReportStatus.ABORTED: "🟠 ABORTED",
    }[report.status]
    lines = [
        f"# Yewo QA Test Report: {_scrub(scrubber, report.journey_name)}",
        "",
        f"* **Journey ID**: `{_scrub(scrubber, report.journey_id)}`",
        f"* **Status**: {report_status}",
        f"* **Scenarios Run**: {len(report.scenarios)} "
        f"({report.passed_count} passed, {report.failed_count} unsuccessful, "
        f"{report.skipped_count} skipped)",
        f"* **Duration**: {report.total_duration}s",
    ]
    lines.extend(_environment_lines(report.environment))
    if session:
        lines.extend(
            [
                f"* **Execution Log**: `{session.root_log_file.name}`",
                f"* **Appium Server Log**: `{session.appium_log_path.name}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Test Flow",
            _scrub(scrubber, report.description),
            "",
            "---",
            "",
        ]
    )
    for index, scenario in enumerate(report.scenarios, start=1):
        _append_scenario(lines, index, scenario, scrubber)

    _append_policy_checks(lines, report.policy_checks, scrubber)

    if report.unevidenced_security_rules:
        lines.append("")
        lines.append("## Security Rules Not Evaluated")
        lines.append("")
        lines.append(
            "No scenario reached a screen where Yewo could evaluate these rules:"
        )
        for rule in report.unevidenced_security_rules:
            lines.append(f"* ⚠️ {_scrub(scrubber, rule)}")

    write_atomic(filepath, "\n".join(lines))
    logger.info("Markdown test report successfully written to %s", filepath)
    return filepath


def _append_policy_checks(
    lines: list[str],
    checks: list[PolicyCheckReport],
    scrubber: SecretScrubber | None,
) -> None:
    """Render run-level checks separately from scenario verification."""
    if not checks:
        return
    icons = {
        PolicyCheckStatus.PASSED: "✅",
        PolicyCheckStatus.FAILED: "❌",
        PolicyCheckStatus.WAIVED: "⚪",
        PolicyCheckStatus.UNSUPPORTED: "⚠️",
        PolicyCheckStatus.EVIDENCE_MISSING: "⚠️",
    }
    lines.extend(["", "## Policy Checks", ""])
    for check in checks:
        lines.extend(
            [
                f"### `{check.check_id}` — {icons[check.status]} {check.status.value}",
                f"* **Platform**: `{check.platform.value}`",
                f"* **Source**: {_scrub(scrubber, check.source)}",
                f"* **Evidence**: {_scrub(scrubber, check.evidence)}",
            ]
        )
        if check.waiver_reason:
            lines.append(
                f"* **Waiver reason**: {_scrub(scrubber, check.waiver_reason)}"
            )
        lines.append("")


def _append_scenario(
    lines: list[str],
    index: int,
    scenario: ScenarioReport,
    scrubber: SecretScrubber | None,
) -> None:
    icon = {
        ReportStatus.PASSED: "🟢",
        ReportStatus.FAILED: "🔴",
        ReportStatus.SKIPPED: "🟡",
    }.get(scenario.status, "🟠")
    lines.extend(
        [
            f"## Scenario {index}: {_scrub(scrubber, scenario.scenario_name)} "
            f"({scenario.test_id})",
            f"* **Cell**: `{scenario.cell}`",
            f"* **Status**: {icon} {scenario.status}"
            + (f" ({scenario.cause})" if scenario.cause else ""),
            f"* **Steps Executed**: {scenario.steps_executed}",
            f"* **Duration**: {scenario.duration}s",
            "",
        ]
    )
    if scenario.failure_reason:
        lines.append(
            f"* ❌ **Failure Reason**: `{_scrub(scrubber, scenario.failure_reason)}`"
        )
    if scenario.lifecycle_operations:
        lines.append("### Lifecycle:")
        for operation in scenario.lifecycle_operations:
            detail = operation.failure_reason or operation.observed or ""
            lines.append(
                f"  * **{operation.phase.value.title()}** "
                f"`{operation.requested}`: {operation.status.value}"
                + (f" — {_scrub(scrubber, detail)}" if detail else "")
            )
        lines.append("")
    if scenario.measurements:
        lines.append("### Measurements:")
        relation = {
            MeasurementRelation.EXACT: "",
            MeasurementRelation.AT_LEAST: "at least ",
            MeasurementRelation.AT_MOST: "at most ",
        }
        for measurement in scenario.measurements:
            lines.append(
                f"  * **`{measurement.metric.value}`**: "
                f"{relation[measurement.relation]}{measurement.value:g} "
                f"{measurement.unit.value} — `{measurement.purpose.value}` via "
                f"{_scrub(scrubber, measurement.collector)}"
            )
        lines.append("")
    rule_verdicts = verdicts.per_rule(list(scenario.rule_verifications))
    if rule_verdicts:
        lines.append("### Verification Results:")
        for verification in rule_verdicts.values():
            status = "[x] Passed" if verification.status == "passed" else "[ ] Failed"
            screen = (
                f" on `{_scrub(scrubber, verification.screen_name)}`"
                if verification.screen_name
                else ""
            )
            lines.append(
                f"  * {status}: {_scrub(scrubber, verification.rule)}{screen} "
                f"(Reasoning: {_scrub(scrubber, verification.reasoning)})"
            )
        lines.append("")
    if scenario.detected_issues:
        lines.append("### Observed Issues:")
        for issue in scenario.detected_issues:
            lines.append(
                f"  * ⚠️ **{issue.severity.upper()} ({issue.issue_type.value})**: "
                f"{_scrub(scrubber, issue.description)} on screen "
                f"`{_scrub(scrubber, issue.current_screen)}`"
            )
        lines.append("")
    lines.append("---")


def save_junit_report(
    report: JourneyReport,
    output_dir: Path,
    scrubber: SecretScrubber | None = None,
    label: str | None = None,
) -> Path:
    stamp = label or time.strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / (
        f"junit_{safe_artifact_stem(report.journey_id)}_{stamp}.xml"
    )
    errors = sum(
        s.status in (ReportStatus.ERROR, ReportStatus.ABORTED)
        for s in report.scenarios
    )
    failures = sum(
        s.status
        in (ReportStatus.FAILED, ReportStatus.UNVERIFIED, ReportStatus.UNSUPPORTED)
        for s in report.scenarios
    )
    policy_failures = sum(
        check.status
        in (
            PolicyCheckStatus.FAILED,
            PolicyCheckStatus.UNSUPPORTED,
            PolicyCheckStatus.EVIDENCE_MISSING,
        )
        for check in report.policy_checks
    )
    policy_skipped = sum(
        check.status is PolicyCheckStatus.WAIVED for check in report.policy_checks
    )
    attributes = {
        "tests": str(len(report.scenarios) + len(report.policy_checks)),
        "failures": str(failures + policy_failures),
        "errors": str(errors),
        "skipped": str(report.skipped_count + policy_skipped),
        "time": str(report.total_duration),
    }
    root = ET.Element("testsuites", {"name": "Yewo User Journeys", **attributes})
    suite = ET.SubElement(
        root,
        "testsuite",
        {
            "name": _scrub(scrubber, report.journey_name),
            "id": _scrub(scrubber, report.journey_id),
            **attributes,
        },
    )
    environment = report.environment
    if environment and environment.target_id:
        properties = ET.SubElement(suite, "properties")
        ET.SubElement(
            properties, "property", name="yewo.target_id", value=environment.target_id
        )
        if environment.run_id:
            ET.SubElement(
                properties, "property", name="yewo.run_id", value=environment.run_id
            )
        if environment.execution_id:
            ET.SubElement(
                properties,
                "property",
                name="yewo.execution_id",
                value=environment.execution_id,
            )
    for scenario in report.scenarios:
        _append_junit_case(suite, report, scenario, scrubber)
    for check in report.policy_checks:
        _append_junit_policy_check(suite, report, check, scrubber)

    content = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    write_atomic(filepath, content)
    logger.info("JUnit XML test report successfully written to %s", filepath)
    return filepath


def _append_junit_case(
    suite,
    report: JourneyReport,
    scenario: ScenarioReport,
    scrubber: SecretScrubber | None,
) -> None:
    testcase = ET.SubElement(
        suite,
        "testcase",
        {
            "name": _scrub(scrubber, scenario.scenario_name),
            "classname": _junit_classname(report, scenario.test_id),
            "time": str(scenario.duration),
        },
    )
    message = _scrub(scrubber, scenario.failure_reason) or "Verification failed"
    if scenario.status in (
        ReportStatus.FAILED,
        ReportStatus.UNVERIFIED,
        ReportStatus.UNSUPPORTED,
    ):
        element = ET.SubElement(
            testcase,
            "failure",
            {"message": message, "type": _junit_type(scenario)},
        )
        details = [
            f"[{issue.severity.upper()}] {issue.issue_type.value}: "
            f"{_scrub(scrubber, issue.description)} on "
            f"{_scrub(scrubber, issue.current_screen)}"
            for issue in scenario.detected_issues
        ]
        element.text = "\n".join(details) or message
    elif scenario.status in (ReportStatus.ERROR, ReportStatus.ABORTED):
        element = ET.SubElement(
            testcase,
            "error",
            {"message": message, "type": _junit_type(scenario)},
        )
        element.text = message
    elif scenario.status == ReportStatus.SKIPPED:
        ET.SubElement(testcase, "skipped", {"message": message})

    verification_lines = [
        " ".join(
            (
                f"rule_id={verification.rule_id or ''}",
                f"status={verification.status}",
                f"rule={_scrub(scrubber, verification.rule)}",
                f"reasoning={_scrub(scrubber, verification.reasoning)}",
            )
        )
        for verification in scenario.rule_verifications
    ]
    lifecycle_lines = [
        " ".join(
            (
                f"lifecycle_id={operation.operation_id}",
                f"phase={operation.phase.value}",
                f"requested={_scrub(scrubber, operation.requested)}",
                f"status={operation.status.value}",
                f"reason={_scrub(scrubber, operation.failure_reason)}",
            )
        )
        for operation in scenario.lifecycle_operations
    ]
    measurement_lines = [
        " ".join(
            (
                f"metric={measurement.metric.value}",
                f"value={measurement.value:g}",
                f"relation={measurement.relation.value}",
                f"unit={measurement.unit.value}",
                f"purpose={measurement.purpose.value}",
                f"collector={_scrub(scrubber, measurement.collector)}",
            )
        )
        for measurement in scenario.measurements
    ]
    if verification_lines or lifecycle_lines or measurement_lines:
        output = ET.SubElement(testcase, "system-out")
        output.text = "\n".join(
            [*lifecycle_lines, *measurement_lines, *verification_lines]
        )


def _append_junit_policy_check(
    suite,
    report: JourneyReport,
    check: PolicyCheckReport,
    scrubber: SecretScrubber | None,
) -> None:
    """Write one run-level check as a visible CI test case."""
    testcase = ET.SubElement(
        suite,
        "testcase",
        {
            "name": check.check_id,
            "classname": _junit_classname(report, "policy"),
            "time": "0",
        },
    )
    details = "\n".join(
        (
            f"status={check.status.value}",
            f"platform={check.platform.value}",
            f"source={_scrub(scrubber, check.source)}",
            f"evidence={_scrub(scrubber, check.evidence)}",
        )
    )
    if check.status in (
        PolicyCheckStatus.FAILED,
        PolicyCheckStatus.UNSUPPORTED,
        PolicyCheckStatus.EVIDENCE_MISSING,
    ):
        failure = ET.SubElement(
            testcase,
            "failure",
            {
                "message": _scrub(scrubber, check.evidence),
                "type": f"YewoPolicyCheck.{check.status.value}",
            },
        )
        failure.text = details
    elif check.status is PolicyCheckStatus.WAIVED:
        ET.SubElement(
            testcase,
            "skipped",
            {"message": f"Waived: {_scrub(scrubber, check.waiver_reason)}"},
        )
    output = ET.SubElement(testcase, "system-out")
    output.text = details + (
        f"\nwaiver_reason={_scrub(scrubber, check.waiver_reason)}"
        if check.waiver_reason
        else ""
    )
