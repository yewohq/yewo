"""Human-readable and CI report serialization for journey runs."""

import logging
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from yewo.errors import ArtifactWriteFailed
from yewo.testing.models import JourneyReport, ScenarioReport
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
        "PASSED": "🟢 PASSED",
        "FAILED": "🔴 FAILED",
        "RUNNING": "🟠 RUNNING (checkpoint)",
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


def _append_scenario(
    lines: list[str],
    index: int,
    scenario: ScenarioReport,
    scrubber: SecretScrubber | None,
) -> None:
    icon = "🟢" if scenario.status == "PASSED" else "🟡" if scenario.status == "SKIPPED" else "🔴"
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
    settled = verdicts.per_rule(list(scenario.rule_verifications))
    if settled:
        lines.append("### Verification Results:")
        for rule in settled.values():
            status = "[x] Passed" if rule.status == "passed" else "[ ] Failed"
            screen = f" on `{_scrub(scrubber, rule.screen_name)}`" if rule.screen_name else ""
            lines.append(
                f"  * {status}: {_scrub(scrubber, rule.rule)}{screen} "
                f"(Reasoning: {_scrub(scrubber, rule.reasoning)})"
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
    errors = sum(s.status in ("ERROR", "ABORTED") for s in report.scenarios)
    failures = sum(
        s.status in ("FAILED", "UNVERIFIED", "UNSUPPORTED")
        for s in report.scenarios
    )
    attributes = {
        "tests": str(len(report.scenarios)),
        "failures": str(failures),
        "errors": str(errors),
        "skipped": str(report.skipped_count),
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
    for scenario in report.scenarios:
        _append_junit_case(suite, report, scenario, scrubber)

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
            "classname": f"{safe_artifact_stem(report.journey_id)}.{scenario.test_id}",
            "time": str(scenario.duration),
        },
    )
    message = _scrub(scrubber, scenario.failure_reason) or "Verification failed"
    if scenario.status in ("FAILED", "UNVERIFIED", "UNSUPPORTED"):
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
    elif scenario.status in ("ERROR", "ABORTED"):
        element = ET.SubElement(
            testcase,
            "error",
            {"message": message, "type": _junit_type(scenario)},
        )
        element.text = message
    elif scenario.status == "SKIPPED":
        ET.SubElement(testcase, "skipped", {"message": message})
