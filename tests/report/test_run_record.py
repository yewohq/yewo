"""The record a real run leaves, written by the runner rather than by hand.

`test_manifest.py` holds the writer to its own contract. These hold the
runner to using it: every row that reaches a report reaches the record, the
lock it ran against is named, and an interrupted run is not recorded as a
failed one.
"""

import contextlib
import json
import tempfile
import unittest
from pathlib import Path

from fakes import walk

from yewo.testing.models import GeneratedTestCase
from yewo.testing.report.manifest import SCHEMA


def scenario(number: int, **extra) -> GeneratedTestCase:
    fields = {
        "test_id": f"TS_00{number}",
        "scenario_name": f"Scenario {number}",
        "cell": "functional.happy_path.0",
        "category": "functional",
        "scenario_type": "happy_path",
        "priority": "P0",
        "steps": walk("Open the app."),
        "expected_outcomes": ["ok"],
    }
    fields.update(extra)
    return GeneratedTestCase(**fields)


class RunnerTests(unittest.TestCase):
    """The runner's own writing of the record, end to end.

    Everything but the device and the paid calls takes part, so a wiring
    change that stops the record being written fails here rather than on a
    device.
    """

    def _run(self, **kwargs):
        from fakes import suite_run

        with tempfile.TemporaryDirectory() as directory:
            return suite_run([scenario(1), scenario(2)], directory, **kwargs)

    def test_a_run_leaves_a_record_of_every_scenario(self):
        _, data = self._run()

        self.assertEqual(data["schema"], SCHEMA)
        self.assertEqual(data["mode"], "test")
        self.assertEqual(data["total"], 2)
        self.assertEqual(data["counts"], {"PASSED": 2})
        self.assertEqual(data["status"], "PASSED")
        self.assertIsNotNone(data["ended_at"])

    def test_the_record_says_what_produced_the_run(self):
        _, data = self._run()
        identity = data["identity"]

        self.assertEqual(identity["platform"], "android")
        self.assertEqual(identity["device"]["model"], "Pixel 8")
        self.assertIn("yewo_version", identity)
        self.assertIn("git", identity)
        self.assertIn("ci", identity)

    def test_the_record_says_what_the_run_was_given(self):
        _, data = self._run()

        self.assertEqual(data["subject"]["kind"], "journey")
        self.assertEqual(data["inputs"]["subject"], "j")
        self.assertEqual(len(data["inputs"]["subject_sha256"]), 64)
        self.assertEqual(data["inputs"]["scenarios_generated"], 2)
        self.assertEqual(data["inputs"]["scenarios_from_lock"], 0)

    def test_what_a_run_spent_is_recorded_with_it(self):
        _, data = self._run()

        self.assertIn("total_calls", data["usage"])
        self.assertNotIn("components", data["usage"])

    def test_the_artifact_classes_are_declared_for_the_journey_that_ran(self):
        _, data = self._run()
        classes = {row["kind"] for row in data["artifacts"]}

        self.assertIn("recordings", classes)
        self.assertIn("screenshots", classes)

    def test_a_failing_scenario_is_pointed_at(self):
        from yewo.testing.models import ScenarioReport

        def failing(test_case):
            return ScenarioReport(
                scenario_uuid=test_case.scenario_uuid,
                test_id=test_case.test_id,
                scenario_name=test_case.scenario_name,
                cell=test_case.cell,
                status="FAILED" if test_case.test_id == "TS_002" else "PASSED",
                failure_reason="the sign-in screen did not stay",
            )

        report, data = self._run(report_for=failing)

        self.assertEqual(report.status, "FAILED")
        self.assertEqual(data["status"], "FAILED")
        self.assertEqual(data["first_failure"]["test_id"], "TS_002")


class LockedRunnerTests(unittest.TestCase):
    """The lock and the record, written by one run."""

    def _run(self, directory):
        from fakes import suite_run

        journey_file = Path(directory) / "yewo.yaml"
        journey_file.write_text("device: Pixel_8\n")
        report, data = suite_run(
            [scenario(1)], directory, journey_file=journey_file
        )
        return report, data, Path(directory) / "yewo.lock"

    def test_a_run_writes_the_lock_beside_the_journey_file(self):
        with tempfile.TemporaryDirectory() as directory:
            _, data, lock_path = self._run(directory)

            self.assertTrue(lock_path.is_file())
            entries = json.loads(lock_path.read_text())["entries"]

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["cell"], "functional.happy_path.0")
        self.assertEqual(len(data["inputs"]["lock_sha256"]), 64)

    def test_the_record_says_the_lock_it_ran_against(self):
        """So two runs can be told apart by the test and not only the result."""
        with tempfile.TemporaryDirectory() as directory:
            _, data, lock_path = self._run(directory)
            from yewo.testing import lock_file

            held = lock_file.read(lock_path)

        self.assertEqual(data["inputs"]["lock_sha256"], held.sha256())

    def test_the_locked_scenario_keeps_the_identity_the_record_names(self):
        with tempfile.TemporaryDirectory() as directory:
            _, data, lock_path = self._run(directory)
            entries = json.loads(lock_path.read_text())["entries"]

        self.assertEqual(
            entries[0]["scenario_uuid"], data["scenarios"][0]["scenario_uuid"]
        )


class EverythingReachesTheRecordTests(unittest.TestCase):
    """The record is the artifact every report is a view over.

    So a row that reaches the markdown report has to reach this file. A cell
    nothing covered, a report that would not write, and usage that would not
    serialize are all rows about the run rather than about a scenario, and a
    reader told to trust one file has to find them in it.
    """

    def _gap(self, status="UNSUPPORTED", reason="no second account was declared"):
        from yewo.errors import Cause
        from yewo.testing.generator import UncoveredCell
        from yewo.testing.taxonomy import Category, CoverageCell, Priority, ScenarioType

        return UncoveredCell(
            CoverageCell(Category.SECURITY, ScenarioType.ROLE, Priority.P0),
            status,
            Cause.SCENARIO,
            reason,
        )

    def _run(self, **kwargs):
        from fakes import suite_run

        with tempfile.TemporaryDirectory() as directory:
            return suite_run([scenario(1)], directory, **kwargs)

    def test_a_cell_nothing_covered_is_a_row_in_the_record(self):
        report, data = self._run(uncovered=[self._gap()])

        rows = {row["test_id"]: row for row in data["scenarios"]}
        self.assertIn("TS_security_role_0", rows)
        self.assertEqual(rows["TS_security_role_0"]["status"], "UNSUPPORTED")
        self.assertEqual(rows["TS_security_role_0"]["cause"], "scenario")
        self.assertEqual(data["total"], 2)

    def test_the_gap_is_listed_where_a_reader_looks_for_gaps(self):
        _, data = self._run(uncovered=[self._gap()])

        self.assertEqual(
            [row["cell"] for row in data["uncovered"]], ["security.role.0"]
        )
        self.assertIn("declared", data["uncovered"][0]["reason"])

    def test_an_uncollectable_metric_is_also_listed_as_an_uncovered_cell(self):
        gap = self._gap(
            status="UNVERIFIED",
            reason="No live collector is available for app_launch_ttid_ms.",
        )

        _, data = self._run(uncovered=[gap])

        self.assertEqual(data["uncovered"][0]["status"], "UNVERIFIED")
        self.assertIn("app_launch_ttid_ms", data["uncovered"][0]["reason"])

    def test_the_record_and_the_report_hold_the_same_rows(self):
        """Two artifacts of one run that disagree are worse than one."""
        report, data = self._run(uncovered=[self._gap()])

        self.assertEqual(
            sorted(r.test_id for r in report.scenarios),
            sorted(row["test_id"] for row in data["scenarios"]),
        )

    def test_a_report_that_would_not_write_fails_the_run_in_the_record(self):
        from unittest.mock import patch

        from yewo.errors import ArtifactWriteFailed
        from fakes import suite_run

        with tempfile.TemporaryDirectory() as directory, patch(
            "yewo.testing.report.writing.save_junit_report",
            side_effect=ArtifactWriteFailed("the disk is full"),
        ):
            report, data = suite_run([scenario(1)], directory)

        self.assertEqual(report.status, "FAILED")
        self.assertEqual(data["status"], "FAILED")
        rows = {row["test_id"]: row for row in data["scenarios"]}
        self.assertIn("junit_report", rows)
        self.assertEqual(rows["junit_report"]["status"], "ERROR")


class GenerationCostTests(unittest.TestCase):
    """What the suite cost to write is part of what the run cost."""

    def test_the_generator_is_not_charged_to_a_scenario(self):
        """It runs once, before any scenario, and no scenario owns it."""
        import tempfile as tmp
        from pathlib import Path as P

        from yewo.llm.usage import CallRecord
        from yewo.testing.report.usage_tracker import LLMUsageTracker
        from yewo.testing.runner import GENERATION, PLANNER

        with tmp.TemporaryDirectory() as directory:
            tracker = LLMUsageTracker(output_dir=P(directory))
            tracker.add_call(
                GENERATION,
                CallRecord(
                    model="m", served_model="m", generation_id="g", attempt=1,
                    finish_reason="stop", latency_ms=1.0, cost=0.01,
                ),
            )

            self.assertEqual(len(tracker.calls_outside_steps(GENERATION)), 1)
            self.assertEqual(tracker.calls_outside_steps(PLANNER), [])


class AbortedRunTests(unittest.TestCase):
    """An interrupted run is not a failed one, and must not read as one."""

    def _interrupted(self, failure):
        from fakes import suite_run

        def boom(_test_case):
            raise failure

        with tempfile.TemporaryDirectory() as directory:
            # The interruption is the point. It is re-raised by the runner.
            with contextlib.suppress(BaseException):
                suite_run([scenario(1), scenario(2)], directory, report_for=boom)
            return json.loads((Path(directory) / "run.json").read_text())

    def test_an_interrupted_run_keeps_that_status(self):
        """The scenarios aggregate to FAILED, which is true of them and not of the run.

        The run reached no verdict. It was stopped. A record that called that
        FAILED would say the suite judged the app and found it wanting.
        """
        data = self._interrupted(KeyboardInterrupt("the user stopped it"))

        self.assertEqual(data["status"], "ABORTED")

    def test_every_report_keeps_the_interrupted_status(self):
        from fakes import suite_run

        def stop(_test_case):
            raise KeyboardInterrupt("the user stopped it")

        with tempfile.TemporaryDirectory() as directory:
            with contextlib.suppress(KeyboardInterrupt):
                suite_run([scenario(1)], directory, report_for=stop)
            root = Path(directory)
            record = json.loads((root / "run.json").read_text())
            markdown = next(root.rglob("*.md")).read_text()
            junit = next(root.rglob("*.xml")).read_text()
            html = next(root.rglob("*.html")).read_text()

        self.assertEqual(record["status"], "ABORTED")
        self.assertIn("**Status**: 🟠 ABORTED", markdown)
        self.assertNotIn("🔴 ABORTED", markdown)
        self.assertIn('type="YewoAborted"', junit)
        self.assertIn("Aborted", html)
        self.assertNotIn("YewoAborted.unknown", junit)

    def test_every_unfinished_scenario_is_aborted_in_the_record(self):
        data = self._interrupted(KeyboardInterrupt("the user stopped it"))

        self.assertEqual(data["counts"], {"ABORTED": 2})

    def test_an_interrupted_device_start_still_leaves_the_canonical_record(self):
        from fakes import suite_run

        with tempfile.TemporaryDirectory() as directory:
            journey_file = Path(directory) / "yewo.yaml"
            journey_file.write_text("device: Pixel_8\n")
            suite_run([scenario(1)], directory, journey_file=journey_file)
            with contextlib.suppress(KeyboardInterrupt):
                suite_run(
                    [scenario(1)], directory, journey_file=journey_file,
                    locked=True, session_start_error=KeyboardInterrupt(),
                )
            data = json.loads((Path(directory) / "run.json").read_text())

        self.assertEqual(data["status"], "ABORTED")
        self.assertEqual(data["counts"], {"ABORTED": 1})
        self.assertIsNotNone(data["ended_at"])

    def test_a_crash_is_recorded_the_same_way(self):
        data = self._interrupted(RuntimeError("the driver died"))

        self.assertEqual(data["status"], "ABORTED")
