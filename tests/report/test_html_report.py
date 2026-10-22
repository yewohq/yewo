"""The run report a person opens: what it says, and what must never reach it."""

import tempfile
import unittest
from pathlib import Path

from yewo.agent.run.record import ActionRecord, DecisionRecord, OutcomeRecord, StepRecord
from yewo.errors import ArtifactWriteFailed
from yewo.identity import AppIdentity, CiIdentity, DeviceIdentity, GitIdentity
from yewo.testing.models import (
    JourneyReport,
    QAIssue,
    RunEnvironment,
    RuleVerification,
    ScenarioReport,
)
from yewo.testing.report.page import document, save_html_report
from yewo.testing.report.page.view import page_of
from yewo.utils.secrets import SecretScrubber


def step(n: int, described: str, success: bool = True, **written) -> StepRecord:
    return StepRecord(
        n=n,
        screen_name="Basket",
        decision=DecisionRecord(current_screen="Basket", reasoning="The control is here."),
        action=ActionRecord(type="tap", described=described),
        outcome=OutcomeRecord(
            success=success, observation="Done" if success else "Nothing changed"
        ),
        screenshots=written,
        perceive_ms=400.0,
        plan_ms=1500.0,
        act_ms=100.0,
    )


def scenario(
    test_id: str = "TS_1",
    name: str = "Pay with saved card",
    status: str = "PASSED",
    steps=(),
    **extra,
) -> ScenarioReport:
    return ScenarioReport(
        test_id=test_id,
        scenario_name=name,
        cell="functional.happy_path.0",
        status=status,
        duration=31.4,
        steps_executed=len(steps),
        steps=list(steps),
        **extra,
    )


def journey(status: str = "FAILED", scenarios=(), **extra) -> JourneyReport:
    return JourneyReport(
        journey_id="checkout",
        journey_name="Checkout",
        description="Pay with the saved card.",
        status=status,
        total_duration=94.0,
        scenarios=list(scenarios),
        **extra,
    )


def page(report: JourneyReport, scrubber=None, **kwargs) -> str:
    return document(
        page_of(report, generated="1 Jan 2026, 00:00 UTC", version="0.1.0", **kwargs),
        scrubber,
    )


class WhatTheReportSaysTests(unittest.TestCase):
    def test_the_verdict_names_the_scenario_and_the_step_that_failed(self):
        """A reader of a red build must not have to scroll to find the failure."""
        report = journey(
            scenarios=[
                scenario("TS_1", "Happy path", "PASSED", [step(1, "Open the basket")]),
                scenario(
                    "TS_2",
                    "Pay with saved card",
                    "FAILED",
                    [step(1, "Tap Pay"), step(2, "Check the confirmation", success=False)],
                    failure_reason="No order confirmation appeared.",
                ),
            ],
            passed_count=1,
            failed_count=1,
        )

        written = page(report)

        self.assertIn("“Pay with saved card” failed at step 2.", written)
        self.assertIn("No order confirmation appeared.", written)
        self.assertIn('href="#ts_2-step-2"', written)
        self.assertIn('id="ts_2-step-2"', written)

    def test_a_passing_run_leads_with_the_pass_and_offers_nothing_to_jump_to(self):
        report = journey(
            status="PASSED",
            scenarios=[scenario(steps=[step(1, "Open the basket")])],
            passed_count=1,
        )

        written = page(report)

        self.assertIn("Every scenario in Checkout passed.", written)
        self.assertIn('class="verdict t-pass"', written)
        self.assertNotIn("Jump to it", written)

    def test_a_provider_failure_is_not_reported_as_the_app_failing(self):
        """ERROR means nothing was learned about the app, and it is not red."""
        report = journey(
            scenarios=[
                scenario(
                    "TS_9",
                    "Card is masked",
                    "ERROR",
                    cause="provider",
                    failure_reason="The model provider returned 503.",
                )
            ],
            failed_count=1,
        )

        written = page(report)

        self.assertIn('class="verdict t-warn"', written)
        self.assertIn("“Card is masked” could not run.", written)
        self.assertIn("provider", written)

    def test_the_findings_are_ordered_worst_first(self):
        report = journey(
            scenarios=[
                scenario(
                    detected_issues=[
                        QAIssue(
                            issue_type="accessibility_violation",
                            description="The remove button is too small.",
                            severity="low",
                        ),
                        QAIssue(
                            issue_type="sensitive_data_exposure",
                            description="The card number is legible.",
                            severity="critical",
                        ),
                    ]
                )
            ]
        )

        written = page(report)

        self.assertLess(
            written.index("The card number is legible."),
            written.index("The remove button is too small."),
        )

    def test_a_rule_nothing_evidenced_is_reported_rather_than_dropped(self):
        report = journey(
            scenarios=[scenario()],
            unevidenced_security_rules=["A signed-out user must not reach orders."],
        )

        written = page(report)

        self.assertIn("Security Rules Not Evaluated", written)
        self.assertIn("A signed-out user must not reach orders.", written)

    def test_the_rules_a_scenario_answered_are_shown_with_the_reasoning(self):
        report = journey(
            scenarios=[
                scenario(
                    rule_verifications=[
                        RuleVerification(
                            rule="An order number is shown.",
                            rule_id="expected:0",
                            status="failed",
                            reasoning="Nothing on the basket carried one.",
                            screen_name="Basket",
                        ),
                        RuleVerification(
                            rule="Nothing here is checked yet.",
                            rule_id="expected:1",
                            status="untested",
                            reasoning="",
                        ),
                    ]
                )
            ]
        )

        written = page(report)

        self.assertIn("An order number is shown.", written)
        self.assertIn("Nothing on the basket carried one.", written)
        self.assertNotIn("Nothing here is checked yet.", written)

    def test_what_the_run_cost_and_what_produced_it_are_both_on_the_page(self):
        report = journey(
            scenarios=[scenario(steps=[step(1, "Open the basket")])],
            environment=RunEnvironment(
                yewo_version="0.1.0",
                model="openrouter/some/model",
                fast_model="openrouter/some/fast",
                platform="android",
                grounding="marks",
                detector="orasul/deki-yolo@287f12d1",
                app=AppIdentity(version="1.4.0", build="212"),
                device=DeviceIdentity(model="Pixel 8", os_version="14"),
                git=GitIdentity(sha="a3f9c1e4d2b", branch="main"),
                ci=CiIdentity(provider="github"),
            ),
        )

        written = page(
            report, usage={"total_calls": 12, "total_estimated_cost_usd": 0.19}
        )

        self.assertIn("$0.19", written)
        self.assertIn("12 model calls", written)
        self.assertIn("openrouter/some/model", written)
        self.assertIn("1.4.0 (212)", written)
        self.assertIn("Pixel 8 · Android 14", written)
        self.assertIn("a3f9c1e", written)
        self.assertIn("orasul/deki-yolo@287f12d1", written)
        self.assertIn("github", written)
        self.assertIn("marks", written)

    def test_a_run_without_usage_telemetry_says_so_rather_than_nothing_spent(self):
        written = page(journey(scenarios=[scenario()]))

        self.assertIn("Not recorded", written)
        self.assertNotIn("$0.00", written)

    def test_a_scenario_that_recorded_no_steps_still_renders(self):
        written = page(journey(scenarios=[scenario(status="ABORTED")]))

        self.assertIn("This scenario recorded no steps.", written)


class WhatMustNotReachTheReportTests(unittest.TestCase):
    def test_app_content_cannot_become_markup(self):
        """Every string on the page is the app's or a model's, and neither is trusted."""
        report = journey(
            scenarios=[
                scenario(
                    name="<script>alert(1)</script>",
                    steps=[step(1, 'Tap "<img src=x onerror=alert(1)>"')],
                    failure_reason="<b>bold</b>",
                )
            ]
        )

        written = page(report)

        self.assertNotIn("<script>alert", written)
        self.assertNotIn("<img src=x", written)
        self.assertNotIn("<b>bold</b>", written)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", written)

    def test_a_declared_secret_is_masked_wherever_it_appears(self):
        scrubber = SecretScrubber()
        scrubber.learn_secrets({"TEST_CARD": "4111222233334444"})
        report = journey(
            scenarios=[
                scenario(
                    steps=[step(1, "Type 4111222233334444 into Card number")],
                    failure_reason="4111222233334444 was refused",
                )
            ]
        )

        written = page(report, scrubber)

        self.assertNotIn("4111222233334444", written)
        self.assertIn("{{TEST_CARD}}", written)

    def test_the_page_counts_the_values_it_is_hiding(self):
        report = journey(
            scenarios=[scenario(failure_reason="{{TEST_CARD}} and {{TEST_CARD}} again")]
        )
        report.description = "Log in with {{TEST_USER_EMAIL}}."

        written = page(report)

        self.assertIn("2 values hidden in this report", written)

    def test_the_page_fetches_nothing(self):
        """A CI artifact is opened offline, and behind policies that block hosts."""
        written = page(journey(scenarios=[scenario(steps=[step(1, "Open the basket")])]))

        self.assertNotIn("//fonts.", written)
        self.assertNotIn("<link", written)
        self.assertNotIn("@import", written)


class WhereTheReportIsWrittenTests(unittest.TestCase):
    def test_screenshots_are_linked_relative_to_the_report(self):
        """The report travels with the run directory, so nothing is absolute."""
        report = journey(
            scenarios=[
                scenario(steps=[step(1, "Open the basket", raw="step001_raw.png")])
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            telemetry = root / "telemetry"
            telemetry.mkdir()
            path = save_html_report(
                report, telemetry, screenshots=root / "screenshots", label="now"
            )
            written = path.read_text(encoding="utf-8")

            self.assertEqual(path.name, "report_checkout_now.html")
            self.assertIn('src="../screenshots/step001_raw.png"', written)
            self.assertNotIn(str(root), written)

    def test_a_journey_id_cannot_choose_the_filename(self):
        report = journey(scenarios=[scenario()])
        report.journey_id = "../../etc/passwd"

        with tempfile.TemporaryDirectory() as directory:
            path = save_html_report(report, Path(directory), label="now")

            self.assertEqual(path.parent, Path(directory))
            self.assertNotIn("..", path.name)

    def test_a_report_that_cannot_be_written_fails_the_run(self):
        report = journey(scenarios=[scenario()])

        with self.assertRaises(ArtifactWriteFailed):
            save_html_report(report, Path("/nonexistent-directory-for-yewo"), label="now")


class WhatEachStepIsLabelledTests(unittest.TestCase):
    def test_a_replayed_step_says_so(self):
        """Replayed steps cost nothing, and the page is where a reader sees that."""
        replayed = step(1, "Open the basket")
        replayed.source = "replay"

        written = page(journey(scenarios=[scenario(steps=[replayed])]))

        self.assertIn("Replayed", written)

    def test_a_target_the_screen_did_not_have_is_distinguished_from_one_not_sent(self):
        from yewo.agent.run.record import GroundingRecord

        unmatched = step(1, "Tap Pay", success=False)
        unmatched.grounding = GroundingRecord(mode="marks", resolved=False, unmatched=True)
        refused = step(2, "Tap Pay", success=False)
        refused.grounding = GroundingRecord(
            mode="marks", resolved=False, refusal="Outside the screen"
        )

        written = page(journey(scenarios=[scenario(steps=[unmatched, refused])]))

        self.assertIn("Target not on screen", written)
        self.assertIn("Action Refused", written)
        self.assertIn("Outside the screen", written)

    def test_a_step_with_no_action_is_still_named(self):
        """A run that stopped before acting still has to show what it was doing."""
        early = step(1, "")
        early.action = None
        bare = step(2, "")
        bare.action = None
        bare.decision = None
        bare.outcome = None

        written = page(journey(scenarios=[scenario(steps=[early, bare])]))

        self.assertIn("The control is here.", written)
        self.assertIn("Step 2", written)


class WhatTheFactsBandSaysTests(unittest.TestCase):
    def test_calls_nothing_knew_the_price_of_are_counted_separately(self):
        """A total that silently treats an unpriced call as free reads as complete."""
        usage = {
            "total_calls": 10,
            "total_estimated_cost_usd": 0.05,
            "cost_unknown_calls": 3,
        }

        written = page(journey(scenarios=[scenario()]), usage=usage)

        self.assertIn("3 without a price", written)

    def test_a_build_this_run_did_not_install_says_so(self):
        report = journey(
            scenarios=[scenario()],
            environment=RunEnvironment(
                yewo_version="0.1.0",
                model="m",
                platform="android",
                grounding="marks",
                app=AppIdentity(version="1.4.0", build="212", installed=False),
            ),
        )

        written = page(report)

        self.assertIn("already installed", written)


class WhenAReportWillNotWriteTests(unittest.TestCase):
    def test_a_checkpoint_that_fails_does_not_stop_the_run(self):
        """The run is still going, and a checkpoint is not what it is judged on."""
        from yewo.testing.report import writing

        with self.assertLogs("yewo.testing.report.writing", level="ERROR") as logged:
            writing.checkpoint(
                journey(scenarios=[scenario()]),
                Path("/nonexistent-directory-for-yewo"),
                None,
                SecretScrubber(),
            )

        self.assertTrue(any("markdown_report" in line for line in logged.output))


    def test_a_record_that_will_not_write_fails_the_run(self):
        """A run nobody can read is not a run that passed."""
        from yewo.testing.report import writing
        from yewo.testing.report.manifest import Inputs, RunManifest, Subject

        report = journey(status="PASSED", scenarios=[scenario()], passed_count=1)
        with tempfile.TemporaryDirectory() as directory:
            manifest = RunManifest(
                Path(directory) / "run.json",
                Subject(kind="journey", id="checkout", name="Checkout"),
                [],
                environment=RunEnvironment(
                    yewo_version="0.1.0", model="m", platform="android", grounding="marks"
                ),
                inputs=Inputs(subject="checkout", subject_sha256="abc"),
            )
            # A directory that was removed mid-run is the ordinary way this
            # happens, and the writer records it rather than raising.
            manifest.path = Path(directory) / "gone" / "run.json"

            writing.persist(
                report, Path(directory), None, SecretScrubber(), manifest, [], {}
            )

        self.assertEqual(report.status, "FAILED")
        self.assertIn("run_manifest", [one.test_id for one in report.scenarios])

    def test_screenshots_with_no_relative_path_are_linked_absolutely(self):
        """On Windows a second drive has no relative path, and the link still opens."""
        from unittest.mock import patch

        from yewo.testing.report.page import render

        with tempfile.TemporaryDirectory() as directory, patch.object(
            render.os.path, "relpath", side_effect=ValueError("different drive")
        ):
            path = save_html_report(
                journey(scenarios=[scenario(steps=[step(1, "Tap", raw="s.png")])]),
                Path(directory),
                screenshots=Path(directory) / "screenshots",
                label="now",
            )

            self.assertIn("file://", path.read_text(encoding="utf-8"))


class WhenTheRecordWillNotWriteTests(unittest.TestCase):
    """The reports are already on disk saying the run passed when this is found."""

    class ManifestThatCannotWrite:
        def __init__(self):
            self.unwritten = None

        def add(self, _report):
            pass

        def finish(self, status, unevidenced=None, usage=None, duration=None):
            self.unwritten = OSError("disk full")

    def test_every_report_is_rewritten_when_the_record_fails(self):
        from yewo.testing.report import writing
        from yewo.utils.secrets import SecretScrubber

        report = journey(status="PASSED", scenarios=[scenario(status="PASSED")])
        with tempfile.TemporaryDirectory() as directory:
            writing.persist(
                report, Path(directory), None, SecretScrubber(),
                self.ManifestThatCannotWrite(), [], {},
            )
            written = {
                path.suffix: path.read_text(encoding="utf-8")
                for path in Path(directory).rglob("*")
                if path.suffix in (".html", ".md", ".xml")
            }

        self.assertEqual(report.status, "FAILED")
        self.assertIn("FAILED", written[".md"])
        self.assertNotIn("passed.", written[".html"])
        self.assertIn("run_manifest", written[".xml"])


class WhatARunLeavesBehindTests(unittest.TestCase):
    def test_a_run_writes_the_page_beside_the_reports_a_machine_reads(self):
        """Three readers, three files, written by the run itself."""
        from fakes import suite_run, walk

        from yewo.testing.models import GeneratedTestCase

        generated = GeneratedTestCase(
            test_id="TS_001",
            scenario_name="Scenario 1",
            cell="functional.happy_path.0",
            category="functional",
            scenario_type="happy_path",
            priority="P0",
            steps=walk("Open the app."),
            expected_outcomes=["ok"],
        )
        with tempfile.TemporaryDirectory() as directory:
            suite_run([generated], directory)
            written = sorted(
                path.name for path in Path(directory).rglob("report_*") if path.is_file()
            )
            machine = sorted(path.name for path in Path(directory).rglob("junit_*.xml"))

        self.assertTrue(any(name.endswith(".html") for name in written), written)
        self.assertTrue(any(name.endswith(".md") for name in written), written)
        self.assertTrue(machine)


if __name__ == "__main__":
    unittest.main()
