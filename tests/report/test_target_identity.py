"""A matrix run remains identifiable in every detailed report format."""

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from yewo.testing.models import JourneyReport, RunEnvironment, ScenarioReport
from yewo.testing.report.manifest import RunInputs, RunManifest, RunSubject
from yewo.testing.report.page import save_html_report
from yewo.testing.report.reports import save_junit_report, save_markdown_report


def matrix_report() -> JourneyReport:
    scenario = ScenarioReport(
        test_id="TS_1",
        scenario_name="Checkout",
        cell="functional.happy_path.0",
        status="PASSED",
    )
    return JourneyReport(
        journey_id="guest_checkout",
        journey_name="Guest checkout",
        description="Complete checkout.",
        status="PASSED",
        scenarios=[scenario],
        passed_count=1,
        environment=RunEnvironment(
            yewo_version="1.0",
            model="model",
            platform="ios",
            grounding="marks",
            execution_id="execution-1",
            run_id="run-1",
            target_id="iphone",
        ),
    )


class DetailedReportIdentityTests(unittest.TestCase):
    def test_target_and_run_identity_reach_human_and_ci_reports(self):
        report = matrix_report()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            markdown = save_markdown_report(report, root, label="now")
            junit = save_junit_report(report, root, label="now")
            page = save_html_report(report, root, label="now")
            markdown_text = markdown.read_text(encoding="utf-8")
            page_text = page.read_text(encoding="utf-8")
            junit_root = ET.parse(junit).getroot()

        properties = {
            item.attrib["name"]: item.attrib["value"]
            for item in junit_root.findall("./testsuite/properties/property")
        }
        case = junit_root.find("./testsuite/testcase")
        self.assertIn("iphone", markdown_text)
        self.assertIn("run-1", markdown_text)
        self.assertIn("execution-1", markdown_text)
        self.assertIn("iphone", page_text)
        self.assertEqual(properties["yewo.target_id"], "iphone")
        self.assertEqual(properties["yewo.run_id"], "run-1")
        self.assertEqual(properties["yewo.execution_id"], "execution-1")
        self.assertEqual(case.attrib["classname"], "iphone.guest_checkout.TS_1")

    def test_the_declared_run_id_is_the_manifest_run_uuid(self):
        report = matrix_report()
        with tempfile.TemporaryDirectory() as directory:
            manifest = RunManifest(
                Path(directory) / "run.json",
                RunSubject(kind="journey", id="guest_checkout", name="Guest checkout"),
                [],
                environment=report.environment,
                inputs=RunInputs(subject="guest_checkout", subject_sha256="a" * 64),
                run_uuid="run-1",
            )

        self.assertEqual(manifest.run_uuid, "run-1")
