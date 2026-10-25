"""Target-matrix selection, isolation, and execution-level reporting."""

import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

from yewo.errors import UsageError
from yewo.testing.models import JourneyReport, RunEnvironment
from yewo.testing.lifecycle import LifecycleOperation
from yewo.testing.models import ScenarioReport
from yewo.testing.suite import Target, Workflow
from yewo.testing.target_matrix import (
    TargetRunResult,
    isolated_run_dir,
    select_workflow_targets,
    restoration_failed,
    validate_matrix_arguments,
    write_execution_summary,
)


def target(target_id: str, platform: str = "android") -> Target:
    return Target(
        target_id,
        {"platform": platform, "device": f"{target_id}-device"},
    )


def workflow(name: str, *targets: Target) -> Workflow:
    return Workflow(name=name, targets=targets)


class SelectionTests(unittest.TestCase):
    def test_declaration_order_is_preserved(self):
        selected = select_workflow_targets(
            [workflow("checkout", target("pixel"), target("iphone", "ios"))],
            None,
        )

        self.assertEqual(
            [item.id for item in selected[0].targets], ["pixel", "iphone"]
        )

    def test_a_global_filter_keeps_matching_targets_in_each_workflow(self):
        selected = select_workflow_targets(
            [
                workflow("checkout", target("pixel"), target("iphone", "ios")),
                workflow("profile", target("pixel")),
            ],
            "pixel",
        )

        self.assertEqual([item.workflow.name for item in selected], ["checkout", "profile"])
        self.assertEqual([item.targets[0].id for item in selected], ["pixel", "pixel"])

    def test_an_unknown_target_is_refused(self):
        with self.assertRaises(UsageError) as caught:
            select_workflow_targets([workflow("checkout", target("pixel"))], "tablet")

        self.assertIn("tablet", str(caught.exception))

    def test_every_selected_workflow_must_declare_its_matrix(self):
        with self.assertRaises(UsageError) as caught:
            select_workflow_targets(
                [workflow("checkout", target("pixel")), workflow("profile")], None
            )

        self.assertIn("profile", str(caught.exception))


class ArgumentTests(unittest.TestCase):
    def _args(self, **changes):
        values = {
            "locked": True,
            "device": None,
            "platform": None,
            "app_path": None,
        }
        values.update(changes)
        return SimpleNamespace(**values)

    def test_a_matrix_requires_the_locked_suite(self):
        with self.assertRaises(UsageError) as caught:
            validate_matrix_arguments(self._args(locked=False))

        self.assertIn("--locked", str(caught.exception))

    def test_a_device_override_cannot_collapse_the_matrix(self):
        with self.assertRaises(UsageError) as caught:
            validate_matrix_arguments(self._args(device="Pixel_9"))

        self.assertIn("--target", str(caught.exception))


class ArtifactIsolationTests(unittest.TestCase):
    def test_each_target_and_run_owns_a_distinct_directory(self):
        root = Path("assets") / "executions" / "execution-1"

        first = isolated_run_dir(root, "checkout", "pixel", "guest", "run-1")
        second = isolated_run_dir(root, "checkout", "iphone", "guest", "run-2")

        self.assertEqual(
            first,
            root / "checkout" / "pixel" / "guest" / "run-1",
        )
        self.assertNotEqual(first, second)

    def test_the_execution_index_links_separate_run_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_report = root / "checkout" / "pixel" / "run-1" / "report.html"
            first_report.parent.mkdir(parents=True)
            first_report.write_text("one", encoding="utf-8")
            results = [
                TargetRunResult(
                    workflow="checkout",
                    target_id="pixel",
                    journey_id="guest",
                    journey_name="Guest checkout",
                    run_id="run-1",
                    status="PASSED",
                    platform="android",
                    device="Pixel 8",
                    app_sha256="a" * 64,
                    report_path=first_report,
                ),
                TargetRunResult(
                    workflow="checkout",
                    target_id="iphone",
                    journey_id="guest",
                    journey_name="Guest checkout",
                    run_id="run-2",
                    status="UNSUPPORTED",
                    platform="ios",
                    device="iPhone 16",
                ),
            ]

            html_path, json_path, junit_path = write_execution_summary(
                root, "execution-1", 2, results
            )
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            page = html_path.read_text(encoding="utf-8")
            junit = ET.parse(junit_path).getroot()

        self.assertEqual(payload["status"], "FAILED")
        self.assertEqual(payload["runs"][0]["run_id"], "run-1")
        self.assertEqual(
            payload["runs"][0]["report"],
            "checkout/pixel/run-1/report.html",
        )
        self.assertIn("iphone", page)
        self.assertIn("Execution report", page)
        self.assertIn("Yewo Sans", page)
        self.assertIn('class="theme"', page)
        self.assertIn('class="run-card"', page)
        self.assertIn('class="badge b-pass"', page)
        self.assertIn("Open run report", page)
        self.assertEqual(junit.attrib["tests"], "2")
        self.assertEqual(junit.attrib["failures"], "1")

    def test_a_failed_restoration_quarantines_its_target(self):
        report = JourneyReport(
            journey_id="guest",
            journey_name="Guest checkout",
            description="d",
            status="FAILED",
            scenarios=[
                ScenarioReport(
                    test_id="TS_1",
                    scenario_name="Checkout",
                    cell="functional.happy_path.0",
                    status="FAILED",
                    lifecycle_operations=[
                        LifecycleOperation(
                            operation_id="restore-1",
                            phase="restoration",
                            kind="device_state",
                            requested="portrait",
                            status="failed",
                            source="orientation",
                        )
                    ],
                )
            ],
        )

        self.assertTrue(restoration_failed(report))

    def test_a_partial_index_says_running_instead_of_passing(self):
        result = TargetRunResult(
            workflow="checkout",
            target_id="pixel",
            journey_id="guest",
            journey_name="Guest checkout",
            run_id="run-1",
            status="PASSED",
            platform="android",
            device="Pixel",
        )
        with tempfile.TemporaryDirectory() as directory:
            _, json_path, _ = write_execution_summary(
                Path(directory), "execution-1", 2, [result]
            )
            payload = json.loads(json_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "RUNNING")

    def test_an_interrupted_partial_index_says_aborted(self):
        result = TargetRunResult(
            workflow="checkout",
            target_id="pixel",
            journey_id="guest",
            journey_name="Guest checkout",
            run_id="run-1",
            status="ABORTED",
            platform="android",
            device="Pixel",
        )
        with tempfile.TemporaryDirectory() as directory:
            _, json_path, _ = write_execution_summary(
                Path(directory), "execution-1", 2, [result]
            )
            payload = json.loads(json_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "ABORTED")

    def test_a_target_result_reads_identity_from_its_own_report(self):
        declared = target("pixel")
        report = JourneyReport(
            journey_id="guest",
            journey_name="Guest checkout",
            description="d",
            status="PASSED",
            environment=RunEnvironment(
                yewo_version="1",
                model="m",
                platform="android",
                grounding="marks",
            ),
        )

        result = TargetRunResult.from_report(
            "checkout", declared, "run-1", report, None
        )

        self.assertEqual(result.target_id, "pixel")
        self.assertEqual(result.run_id, "run-1")
