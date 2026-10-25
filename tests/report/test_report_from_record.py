"""The page a reader builds from `run.json`, with no run in memory.

This is what makes the record canonical rather than one output among four.
A CI job downloads one file and renders the report from it. Nothing here
opens a device, a model or the run that wrote the file.

The test that matters is the first one: the same run, rendered both ways,
produces the same page. A field the record does not carry shows up there as
a difference, which is a gap in the record rather than in the page.
"""

import json
import tempfile
import unittest
from pathlib import Path

from fakes import suite_run, walk

from yewo.agent.run.record import ActionRecord, DecisionRecord, OutcomeRecord, StepRecord
from yewo.errors import UsageError
from yewo.testing.models import (
    GeneratedTestCase,
    QAIssue,
    RuleVerification,
    ScenarioReport,
)
from yewo.testing.report.page import RUN_FILES, document, page_of_record
from yewo.testing.report.page.view import page_of
from yewo.testing.report.manifest import SCHEMA

GENERATED = "22 Oct 2026, 07:46 UTC"
VERSION = "0.0.2"


def scenario(number: int, **extra) -> GeneratedTestCase:
    fields = {
        "test_id": f"TS_00{number}",
        "scenario_name": f"Scenario {number}",
        "cell": "functional.happy_path.0",
        "category": "functional",
        "scenario_type": "happy_path",
        "priority": "P0",
        "steps": walk("Open the app."),
        "expected_outcomes": ["An order confirmation is shown."],
    }
    fields.update(extra)
    return GeneratedTestCase(**fields)


def step(n: int, success: bool = True) -> StepRecord:
    return StepRecord(
        n=n,
        screen_name="Basket",
        source="replay" if n == 1 else "model",
        decision=DecisionRecord(current_screen="Basket", reasoning="The control is here."),
        action=ActionRecord(type="tap", description=f"Tap the control on step {n}"),
        outcome=OutcomeRecord(
            success=success, observation="Done" if success else "Nothing changed"
        ),
        screenshots={"raw": f"step{n:03d}_raw.png"},
        perceive_ms=400.0,
        plan_ms=1500.0,
        act_ms=100.0,
    )


def failing(case: GeneratedTestCase) -> ScenarioReport:
    """A report with something in every part of the page a reader reads."""
    walked = [step(1), step(2), step(3, success=case.test_id.endswith("2"))]
    return ScenarioReport(
        scenario_uuid=case.scenario_uuid,
        test_id=case.test_id,
        scenario_name=case.scenario_name,
        cell=case.cell,
        status="FAILED" if case.test_id.endswith("2") else "PASSED",
        duration=31.0,
        steps_executed=len(walked),
        steps=walked,
        failure_reason="The app stayed on the basket after Pay was tapped.",
        rule_verifications=[
            RuleVerification(
                rule="An order confirmation screen shows an order number.",
                rule_id="expected:0",
                status="failed",
                reasoning="The basket was still on screen.",
                screen_name="Basket",
            )
        ],
        detected_issues=[
            QAIssue(
                issue_type="sensitive_data_exposure",
                severity="high",
                description="The full card number is legible before the field masks it.",
                current_screen="Payment",
            )
        ],
    )


def both_pages(directory: str):
    """One run, rendered from the report it held and from the file it wrote."""
    report, data = suite_run(
        [scenario(1), scenario(2)], directory, report_for=failing
    )
    from_report = page_of(
        report,
        generated=GENERATED,
        version=report.environment.yewo_version if report.environment else VERSION,
        usage=data.get("usage"),
        screenshots_href="screenshots",
        artifacts=RUN_FILES,
    )
    from_record = page_of_record(data, generated=GENERATED)
    return from_report, from_record


class WhatTheRecordRendersTests(unittest.TestCase):
    def test_the_record_renders_the_same_page_as_the_run_that_wrote_it(self):
        with tempfile.TemporaryDirectory() as directory:
            from_report, from_record = both_pages(directory)

        self.assertEqual(document(from_record), document(from_report))

    def test_the_verdict_and_the_walk_survive_the_file(self):
        """Named separately, so a whole-page failure says which part moved."""
        with tempfile.TemporaryDirectory() as directory:
            from_report, from_record = both_pages(directory)

        self.assertEqual(from_record.verdict, from_report.verdict)
        self.assertEqual(from_record.scenarios, from_report.scenarios)
        self.assertEqual(from_record.findings, from_report.findings)
        self.assertEqual(from_record.tiles, from_report.tiles)

    def test_the_record_says_what_the_run_was_asked_to_do(self):
        """The journey's own words. Without them the page has no description."""
        with tempfile.TemporaryDirectory() as directory:
            _, data = suite_run([scenario(1)], directory)

        view = page_of_record(data, generated=GENERATED)

        self.assertTrue(data["subject"]["description"])
        self.assertEqual(view.description, data["subject"]["description"])

    def test_a_page_is_written_from_a_file_and_nothing_else(self):
        with tempfile.TemporaryDirectory() as directory:
            _, data = suite_run([scenario(1)], directory)
            out = Path(directory) / "elsewhere"
            out.mkdir()
            record = out / "run.json"
            record.write_text(json.dumps(data), encoding="utf-8")

            from yewo.testing.report.page import save_page

            written = save_page(
                page_of_record(
                    json.loads(record.read_text(encoding="utf-8")),
                    generated=GENERATED,
                ),
                out,
                label="record",
            )

            self.assertTrue(written.exists())
            self.assertIn("Scenario 1", written.read_text(encoding="utf-8"))


class WhenTheRecordIsNotReadableTests(unittest.TestCase):
    """A file somebody edited is refused, rather than reported as an empty run."""

    def test_a_file_that_is_not_a_record_is_refused(self):
        with self.assertRaises(UsageError):
            page_of_record({"hello": "world"}, generated=GENERATED)

    def test_an_unreadable_step_fails_the_page(self):
        """A walk that silently lost a step reads as a shorter walk."""
        payload = {
            "schema": SCHEMA,
            "subject": {"kind": "journey", "id": "j", "name": "J"},
            "status": "PASSED",
            "counts": {"PASSED": 1},
            "scenarios": [{"test_id": "TS_001", "name": "S", "steps": [{"n": "late"}]}],
        }

        with self.assertRaises(UsageError):
            page_of_record(payload, generated=GENERATED)

    def test_an_unreadable_identity_fails_the_page(self):
        payload = {
            "schema": SCHEMA,
            "subject": {"kind": "journey", "id": "j", "name": "J"},
            "status": "PASSED",
            "counts": {},
            "scenarios": [],
            "identity": {"model": "m"},
        }

        with self.assertRaises(UsageError):
            page_of_record(payload, generated=GENERATED)


class WhatTheRecordIsAllowedToSayTests(unittest.TestCase):
    """A record is a file, and a file can say anything."""

    def _record(self, **extra) -> dict:
        base = {
            "schema": SCHEMA,
            "subject": {"kind": "journey", "id": "j", "name": "J"},
            "status": "PASSED",
            "counts": {"PASSED": 1},
            "scenarios": [{"test_id": "T", "name": "S", "status": "PASSED"}],
        }
        base.update(extra)
        return base

    def test_a_status_yewo_never_writes_is_refused(self):
        """Otherwise the renderer titles it and reports a verdict nothing reached."""
        record = self._record(
            scenarios=[{"test_id": "T", "name": "S", "status": "teleported"}]
        )

        with self.assertRaises(UsageError) as refused:
            page_of_record(record, generated=GENERATED)

        self.assertIn("teleported", str(refused.exception))

    def test_a_rule_verification_status_is_not_a_scenario_status(self):
        record = self._record(
            scenarios=[{"test_id": "T", "name": "S", "status": "untested"}]
        )

        with self.assertRaises(UsageError):
            page_of_record(record, generated=GENERATED)

    def test_a_run_status_yewo_never_writes_is_refused(self):
        with self.assertRaises(UsageError):
            page_of_record(self._record(status="vibes"), generated=GENERATED)

    def test_status_text_is_normalized_to_the_shared_enum(self):
        from yewo.testing.models import ReportStatus
        from yewo.testing.report.page.record import run_of

        record = self._record(
            status="pass",
            scenarios=[{"test_id": "T", "name": "S", "status": "fail"}],
        )

        loaded = run_of(record)

        self.assertIs(loaded.status, ReportStatus.PASSED)
        self.assertIs(loaded.scenarios[0].status, ReportStatus.FAILED)

    def test_a_duration_that_is_a_word_is_refused_as_a_record_problem(self):
        """Named as the field it is about, not as whatever float raises."""
        record = self._record(
            scenarios=[
                {"test_id": "T", "name": "S", "status": "PASSED",
                 "duration_seconds": "abc"}
            ]
        )

        with self.assertRaises(UsageError) as refused:
            page_of_record(record, generated=GENERATED)

        self.assertIn("duration_seconds", str(refused.exception))

    def test_a_field_the_page_does_not_render_is_ignored(self):
        """The record holds more than a page shows, and that is not an error."""
        view = page_of_record(
            self._record(policy_checks=[], first_failure=None, mode="test"),
            generated=GENERATED,
        )

        self.assertEqual(view.scenarios[0].status, "Passed")

    def test_a_schema_this_yewo_does_not_read_is_refused(self):
        """Nothing translates between formats while the format is moving."""
        from yewo.testing.report.manifest import SCHEMA

        with self.assertRaises(UsageError) as refused:
            page_of_record(self._record(schema=SCHEMA + 997), generated=GENERATED)

        self.assertIn("schema", str(refused.exception))

    def test_a_record_with_no_schema_is_refused(self):
        bare = self._record()
        del bare["schema"]

        with self.assertRaises(UsageError):
            page_of_record(bare, generated=GENERATED)

    def test_a_cost_that_is_a_word_is_refused(self):
        """It would otherwise escape as whatever formatting a float raises."""
        record = self._record(usage={"total_estimated_cost_usd": "money"})

        with self.assertRaises(UsageError):
            page_of_record(record, generated=GENERATED)

    def test_a_number_written_as_a_string_is_refused(self):
        """Read rather than coerced. A quoted number was not written by Yewo."""
        record = self._record(
            scenarios=[{"test_id": "T", "name": "S", "status": "PASSED",
                        "duration_seconds": "12.5"}]
        )

        with self.assertRaises(UsageError):
            page_of_record(record, generated=GENERATED)

    def test_the_summary_is_counted_from_the_rows_it_summarises(self):
        """A corrupted count cannot put a green headline over failed rows."""
        record = self._record(
            counts={"PASSED": 9},
            scenarios=[{"test_id": "T", "name": "S", "status": "FAILED"}],
        )

        view = page_of_record(record, generated=GENERATED)

        self.assertEqual(view.verdict.counts, "0 of 1 scenarios passed")

    def test_the_version_shown_is_the_one_that_ran(self):
        """A page rendered today from an old run is about that run."""
        record = self._record(
            identity={"yewo_version": "0.0.1", "model": "m", "platform": "android",
                      "grounding": "coordinates"}
        )

        view = page_of_record(record, generated=GENERATED)

        self.assertEqual(view.version, "0.0.1")


class WhatAPassingRunStillReportsTests(unittest.TestCase):
    def test_a_green_run_names_the_findings_it_recorded(self):
        """A run can reach every outcome and still have watched a card leak."""
        with tempfile.TemporaryDirectory() as directory:
            _, data = suite_run([scenario(1)], directory, report_for=failing)

        view = page_of_record(data, generated=GENERATED)

        self.assertEqual(len(view.findings), 1)
        self.assertIn("1 issue observed", view.verdict.detail)
        self.assertNotIn("No issues observed", view.verdict.detail)

    def test_one_row_per_rule_rather_than_one_per_answer(self):
        """The auditor answers every rule on every screen. Only one decides it."""
        with tempfile.TemporaryDirectory() as directory:
            _, data = suite_run([scenario(1)], directory, report_for=failing)

        view = page_of_record(data, generated=GENERATED)
        answers = len(data["scenarios"][0]["rule_verifications"])

        self.assertEqual(len(view.scenarios[0].checks), 1)
        self.assertEqual(view.scenarios[0].checks[0].observation_count, answers)

    def test_the_scenario_names_its_cell_and_test_id(self):
        """What matches a failure to its lock entry and to the next run."""
        with tempfile.TemporaryDirectory() as directory:
            _, data = suite_run([scenario(1)], directory)

        written = document(page_of_record(data, generated=GENERATED))

        self.assertIn("functional.happy_path.0", written)
        self.assertIn("TS_001", written)


class HowASecurityRuleSettlesTests(unittest.TestCase):
    """A password exposed at step 4 stays exposed, whatever step 9 shows."""

    def _answers(self):
        return [
            RuleVerification(
                rule="The card number is never legible.",
                rule_id="security:0",
                status="failed",
                reasoning="The full number was legible.",
                screen_name="Payment",
            ),
            RuleVerification(
                rule="The card number is never legible.",
                rule_id="security:0",
                status="passed",
                reasoning="Masked here.",
                screen_name="Receipt",
            ),
        ]

    def test_a_later_pass_does_not_take_back_a_violation(self):
        """Otherwise the page shows Passed on a run the judge failed for it."""
        from yewo.testing.judge import verdicts

        decided = verdicts.per_rule(self._answers())

        self.assertEqual(decided["security:0"].status, "failed")
        self.assertEqual(decided["security:0"].screen_name, "Payment")

    def test_the_page_says_what_the_verdict_said(self):
        from yewo.testing.judge import verdicts

        violated = verdicts._violated_security_rule(self._answers())
        decided = verdicts.per_rule(self._answers())

        self.assertIsNotNone(violated)
        self.assertEqual(decided["security:0"].status, "failed")

    def test_an_expected_outcome_still_settles_on_a_pass(self):
        """The two kinds settle differently, and this is the other one."""
        from yewo.testing.judge import verdicts

        answers = [
            RuleVerification(rule="r", rule_id="expected:0", status="failed",
                             reasoning="not here", screen_name="Cart"),
            RuleVerification(rule="r", rule_id="expected:0", status="passed",
                             reasoning="here", screen_name="Receipt"),
        ]

        self.assertEqual(verdicts.per_rule(answers)["expected:0"].status, "passed")


class WhatAnUnfinishedRecordRendersTests(unittest.TestCase):
    """The record is written before the first scenario and tailed while it runs."""

    def _running(self, directory: str) -> dict:
        from yewo.identity import DeviceIdentity
        from yewo.testing.models import RunEnvironment
        from yewo.testing.report.manifest import RunInputs, RunManifest, RunSubject

        manifest = RunManifest(
            Path(directory) / "run.json",
            RunSubject(kind="journey", id="j", name="J", description="d"),
            [scenario(1)],
            environment=RunEnvironment(
                yewo_version="0.0.2", model="m", platform="android",
                grounding="coordinates", device=DeviceIdentity(),
            ),
            inputs=RunInputs(subject="j", subject_sha256="x" * 64),
        )
        return manifest.payload()

    def test_a_record_with_no_duration_yet_still_renders(self):
        with tempfile.TemporaryDirectory() as directory:
            view = page_of_record(self._running(directory), generated=GENERATED)

        self.assertEqual(view.verdict.badge, "Running")

    def test_a_scenario_that_has_not_started_is_not_a_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            view = page_of_record(self._running(directory), generated=GENERATED)

        self.assertEqual(view.scenarios[0].tone, "muted")
        self.assertNotIn("failed", view.verdict.headline)
        self.assertIn("still running", view.verdict.headline)


class WhatAStepWasForTests(unittest.TestCase):
    """A row says what the step was for. The coordinates are in the detail."""

    def test_the_intent_is_recorded_beside_the_mechanics(self):
        from types import SimpleNamespace

        from yewo.agent.run.record import StepLog

        log = StepLog()
        log.open(1)
        log.record_action(
            SimpleNamespace(type="tap", reason="Tap Login to submit the credentials."),
            "tap at (540, 1644)",
            SimpleNamespace(success=True, observation="Done", error=None),
            None,
        )

        self.assertEqual(log.current.action.intent, "Tap Login to submit the credentials.")
        self.assertEqual(log.current.action.description, "tap at (540, 1644)")

    def test_the_row_reads_as_the_intent(self):
        from yewo.agent.run.record import ActionRecord
        from yewo.testing.report.page.scenario_steps import _title

        titled = StepRecord(
            n=1,
            action=ActionRecord(
                type="tap", description="tap at (540, 1644)", intent="Tap Login."
            ),
        )

        self.assertEqual(_title(titled), "Tap Login.")

    def test_a_record_without_an_intent_still_reads(self):
        from yewo.agent.run.record import ActionRecord
        from yewo.testing.report.page.scenario_steps import _title

        record = StepRecord(
            n=1, action=ActionRecord(type="tap", description="tap at (1, 2)")
        )

        self.assertEqual(_title(record), "tap at (1, 2)")


class WhatTheOfferSaysTests(unittest.TestCase):
    """The strip at the foot offers replay, and replay does not persist yet."""

    def test_the_page_offers_nothing_it_cannot_do(self):
        with tempfile.TemporaryDirectory() as directory:
            from_report, _ = both_pages(directory)

        written = document(from_report)

        self.assertNotIn("See Yewo Cloud", written)
        self.assertNotIn("could replay without AI", written)

    def test_the_offer_is_one_switch_away(self):
        """Turned on with durable replay, which is A9, and not before."""
        from dataclasses import replace
        from unittest import mock

        from yewo.testing.report.page import render

        with tempfile.TemporaryDirectory() as directory:
            from_report, _ = both_pages(directory)

        offered = replace(from_report, replayable=19)
        with mock.patch.object(render, "OFFER_CLOUD", True):
            written = document(offered)

        self.assertIn("See Yewo Cloud", written)
        self.assertIn("19 steps", written)


if __name__ == "__main__":
    unittest.main()
