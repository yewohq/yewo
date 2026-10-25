"""Lifecycle evidence stays canonical across every report view."""

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from pydantic import ValidationError

from yewo.testing.lifecycle import (
    LifecycleOperation,
    LifecyclePhase,
    LifecycleStatus,
)
from yewo.testing.models import JourneyReport, ReportStatus, ScenarioReport
from yewo.testing.report.page import document
from yewo.testing.report.page.view import page_of
from yewo.testing.report.reports import save_junit_report, save_markdown_report


def operation(status=LifecycleStatus.CONFIRMED, reason=None):
    return LifecycleOperation(
        operation_id="app_state.setup",
        phase=LifecyclePhase.SETUP,
        kind="app_state",
        requested="clear_data",
        status=status,
        source="adb shell pm clear",
        observed="clear_data (default)" if reason is None else None,
        failure_reason=reason,
    )


def scenario(status=ReportStatus.PASSED, lifecycle_operations=None):
    return ScenarioReport(
        scenario_uuid="scenario-uuid",
        test_id="TS_001",
        scenario_name="Checkout",
        cell="functional.happy_path.0",
        status=status,
        lifecycle_operations=list(lifecycle_operations or []),
    )


def report(scenario_report):
    passed = int(scenario_report.status is ReportStatus.PASSED)
    return JourneyReport(
        journey_id="checkout",
        journey_name="Checkout",
        description="Let the user check out.",
        status=(ReportStatus.PASSED if passed else ReportStatus.FAILED),
        scenarios=[scenario_report],
        passed_count=passed,
        failed_count=1 - passed,
    )


class LifecycleVerdictTests(unittest.TestCase):
    def test_a_failed_lifecycle_operation_cannot_hide_in_a_passing_run(self):
        failed = operation(LifecycleStatus.FAILED, "device offline")

        with self.assertRaises(ValidationError):
            report(scenario(lifecycle_operations=[failed]))


class LifecycleExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.report = report(scenario(lifecycle_operations=[operation()]))

    def test_markdown_separates_lifecycle_from_user_steps(self):
        path = save_markdown_report(self.report, self.directory, label="lifecycle")
        rendered = path.read_text(encoding="utf-8")

        self.assertIn("### Lifecycle", rendered)
        self.assertIn("`clear_data`: confirmed", rendered)

    def test_junit_keeps_lifecycle_evidence_in_system_output(self):
        path = save_junit_report(self.report, self.directory, label="lifecycle")
        root = ET.parse(path).getroot()

        output = root.find("testsuite/testcase/system-out")
        self.assertIsNotNone(output)
        self.assertIn("lifecycle_id=app_state.setup", output.text)

    def test_html_labels_lifecycle_as_state_outside_the_user_journey(self):
        rendered = document(
            page_of(self.report, generated="now", version="test")
        )

        self.assertIn("Lifecycle", rendered)
        self.assertIn("State outside the user journey", rendered)

    def test_run_json_is_the_source_of_the_lifecycle_evidence(self):
        from fakes import suite_run, walk
        from yewo.testing.models import GeneratedTestCase

        case = GeneratedTestCase(
            test_id="TS_001",
            scenario_name="Checkout",
            cell="functional.happy_path.0",
            category="functional",
            scenario_type="happy_path",
            priority="P0",
            steps=walk("Open the app."),
            expected_outcomes=["Checkout is visible"],
        )

        def report_for(test_case):
            return ScenarioReport(
                scenario_uuid=test_case.scenario_uuid,
                test_id=test_case.test_id,
                scenario_name=test_case.scenario_name,
                cell=test_case.cell,
                status=ReportStatus.PASSED,
                lifecycle_operations=[operation()],
            )

        _, run_record = suite_run(
            [case], self.directory, report_for=report_for
        )

        recorded = run_record["scenarios"][0]["lifecycle_operations"][0]
        self.assertEqual(recorded["status"], "confirmed")
        self.assertEqual(recorded["requested"], "clear_data")


if __name__ == "__main__":
    unittest.main()
