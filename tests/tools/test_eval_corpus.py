"""What a corpus fixture is allowed to say, and what it builds."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fakes import solid_png

from tools.eval import corpus


def _written(directory: Path, name: str, fixture: dict) -> Path:
    path = directory / f"{name}.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")
    return path


class VerdictFixtureTests(unittest.TestCase):
    """A fixture holds evidence, and loading it builds the real objects."""

    def _fixture(self, **overrides) -> dict:
        fixture = {
            "id": "an-outcome-with-no-verdict",
            "why": "The outcome exists and nothing answered it.",
            "run": {"status": "goal_reached"},
            "outcomes": ["Order is confirmed"],
            "expect": {"status": "UNVERIFIED", "cause": "evidence"},
        }
        fixture.update(overrides)
        return fixture

    def _loaded(self, **overrides):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _written(root, "one", self._fixture(**overrides))
            return corpus.verdict_fixtures(root)[0]

    def test_a_fixture_builds_the_arguments_the_judge_takes(self):
        """Every value reaches the judge as the type production hands it.

        A fixture that built loose namespaces would keep passing after a
        field was renamed, which is the failure a corpus exists to catch.
        """
        from yewo.agent.run.outcomes import DeviceRunResult
        from yewo.testing.models import ExpectedOutcome, QAIssue, RuleVerification

        fixture = self._loaded(
            issues=[{"issue_type": "crash_screen", "severity": "critical",
                     "description": "A crash dialog is on screen."}],
            verifications=[{"rule": "Order is confirmed", "rule_id": "expected:0",
                            "status": "passed", "reasoning": "It says confirmed."}],
        )
        arguments = fixture.arguments

        self.assertIsInstance(arguments["run_result"], DeviceRunResult)
        self.assertIsInstance(arguments["all_issues"][0], QAIssue)
        self.assertIsInstance(arguments["all_verifications"][0], RuleVerification)
        self.assertIsInstance(arguments["required_outcomes"][0], ExpectedOutcome)

    def test_a_scoped_outcome_keeps_its_scope(self):
        from yewo.testing.models import AssertionScope

        fixture = self._loaded(
            outcomes=[{
                "statement": "The screen opens quickly",
                "scope": "measured",
                "metric": "app_launch_ttid_ms",
                "threshold": 800,
            }]
        )

        self.assertIs(
            fixture.arguments["required_outcomes"][0].scope, AssertionScope.MEASURED
        )

    def test_the_number_of_screens_judged_is_what_the_fixture_says(self):
        """The judge reads whether any screen was audited, and nothing else of them."""
        self.assertEqual(len(self._loaded(screens_judged=0).arguments["assessments"]), 0)
        self.assertEqual(len(self._loaded(screens_judged=2).arguments["assessments"]), 2)

    def test_measurements_are_built_as_the_judge_receives_them(self):
        from yewo.testing.models import Measurement

        fixture = self._loaded(measurements=[{
            "metric": "app_launch_ttid_ms",
            "value": 700,
            "unit": "ms",
            "source": "device",
            "platform": "ios",
            "collected_at": "2026-10-25T12:00:00Z",
        }])

        self.assertIsInstance(fixture.arguments["measurement_samples"][0], Measurement)

    def test_a_planner_answer_carries_feasibility_and_its_reason(self):
        fixture = self._loaded(
            planner=[{"feasible": False, "infeasibility_reason": "No guest checkout."}]
        )
        answer = fixture.arguments["qa_outputs"][0]

        self.assertFalse(answer.is_scenario_feasible)
        self.assertEqual(answer.infeasibility_reason, "No guest checkout.")

    def test_a_receipt_is_confirmed_or_unreadable_as_the_fixture_asks(self):
        """A fixture names the state of a receipt. It does not spell out four fields."""
        confirmed = self._loaded(
            run={"status": "goal_reached",
                 "receipts": [{"capability": "network", "requested": "offline"}]}
        )
        unreadable = self._loaded(
            run={"status": "goal_reached",
                 "receipts": [{"capability": "network", "requested": "offline",
                               "confirmed": False}]}
        )

        self.assertTrue(confirmed.arguments["run_result"].capability_receipts[0].confirmed)
        self.assertFalse(unreadable.arguments["run_result"].capability_receipts[0].confirmed)

    def test_a_capability_no_such_kind_names_is_refused(self):
        with self.assertRaises(corpus.CorpusError) as refused:
            self._loaded(required_capabilities=["teleport"])

        self.assertIn("required_capabilities.0", str(refused.exception))

    def test_a_fixture_missing_a_required_key_names_the_file(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture()
            del fixture["why"]
            _written(root, "unnamed", fixture)

            with self.assertRaises(corpus.CorpusError) as refused:
                corpus.verdict_fixtures(root)

        self.assertIn("unnamed.json", str(refused.exception))
        self.assertIn("why: Field required", str(refused.exception))

    def test_a_fixture_that_cannot_be_parsed_is_not_an_empty_corpus(self):
        """A corpus that reads as empty passes every gate over it.

        The refusal carries what the decoder said. Read as an empty object
        the file is refused too, for a missing key, and a maintainer is then
        told the wrong thing about a file that is not valid JSON at all.
        """
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "broken.json").write_text("{not json", encoding="utf-8")

            with self.assertRaises(corpus.CorpusError) as refused:
                corpus.verdict_fixtures(root)

        self.assertIn("broken.json", str(refused.exception))
        self.assertIn("JSON", str(refused.exception))

    def test_a_directory_with_no_fixtures_is_refused(self):
        """An empty corpus answers every question correctly and reports 0 of 0."""
        with TemporaryDirectory() as directory, \
             self.assertRaises(corpus.CorpusError) as refused:
            corpus.verdict_fixtures(Path(directory))

        self.assertIn("never empty", str(refused.exception))

    def test_a_run_status_no_walk_can_end_on_is_refused(self):
        """`DeviceRunResult` is a dataclass, so it would accept the string."""
        with self.assertRaises(corpus.CorpusError) as refused:
            self._loaded(run={"status": "teleported"})

        self.assertIn("run.status", str(refused.exception))

    def test_a_verdict_word_no_scenario_can_end_on_is_refused(self):
        with self.assertRaises(corpus.CorpusError) as refused:
            self._loaded(expect={"status": "PASSD"})

        self.assertIn("expect.status", str(refused.exception))

    def test_a_cause_no_failure_carries_is_refused(self):
        with self.assertRaises(corpus.CorpusError) as refused:
            self._loaded(expect={"status": "UNVERIFIED", "cause": "weather"})

        self.assertIn("expect.cause", str(refused.exception))

    def test_a_field_the_fixture_format_does_not_have_is_refused(self):
        """A misspelled key would otherwise be read as a default."""
        with self.assertRaises(corpus.CorpusError) as refused:
            self._loaded(verification=[])

        self.assertIn("Extra inputs are not permitted", str(refused.exception))

    def test_an_id_holding_a_slash_is_refused(self):
        """A screen rule is reported as `fixture/expected:0`."""
        with self.assertRaises(corpus.CorpusError) as refused:
            self._loaded(id="a/b")

        self.assertIn("no '/'", str(refused.exception))

    def test_a_feasibility_written_as_a_word_is_read_as_the_boolean(self):
        fixture = self._loaded(planner=[{"feasible": "false"}])

        self.assertIs(fixture.arguments["qa_outputs"][0].is_scenario_feasible, False)

    def test_a_feasibility_that_is_neither_is_refused(self):
        with self.assertRaises(corpus.CorpusError):
            self._loaded(planner=[{"feasible": "maybe"}])

    def test_a_value_the_real_model_refuses_is_named_as_this_fixtures(self):
        """Left to propagate, a validation error is a traceback where the gate
        documents an exit code."""
        with self.assertRaises(corpus.CorpusError) as refused:
            self._loaded(
                verifications=[{"rule": "r", "rule_id": "expected:0",
                                "status": "nonsense", "reasoning": "z"}]
            )

        self.assertIn("one.json", str(refused.exception))
        self.assertIn("verifications.0.status", str(refused.exception))

    def test_a_fixture_carries_the_hash_of_what_it_says(self):
        """A baseline entry is bound to this, so an edited fixture lapses it."""
        first = self._loaded()
        again = self._loaded()
        edited = self._loaded(why="a different reason")

        self.assertEqual(first.digest, again.digest)
        self.assertNotEqual(first.digest, edited.digest)

    def test_fixtures_load_in_a_fixed_order(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _written(root, "second", self._fixture(id="second"))
            _written(root, "first", self._fixture(id="first"))

            self.assertEqual(
                [fixture.id for fixture in corpus.verdict_fixtures(root)],
                ["first", "second"],
            )


class ExpectationTests(unittest.TestCase):
    """An expectation holds the status exactly and the reason loosely."""

    def test_a_named_cause_must_match(self):
        expectation = corpus.Expectation(status="FAILED", cause="agent", checks_cause=True)

        self.assertIsNone(expectation.disagreement("FAILED", None, "agent"))
        self.assertIn("expected agent", expectation.disagreement("FAILED", None, None))

    def test_a_null_cause_means_the_verdict_must_carry_none(self):
        """A crash is the app's doing. A cause on it would blame the run."""
        expectation = corpus.Expectation(status="FAILED", cause=None, checks_cause=True)

        self.assertIsNone(expectation.disagreement("FAILED", None, None))
        self.assertIn("cause agent", expectation.disagreement("FAILED", None, "agent"))

    def test_an_absent_cause_checks_nothing(self):
        expectation = corpus.Expectation(status="FAILED")

        self.assertIsNone(expectation.disagreement("FAILED", None, "agent"))

    def test_the_reason_is_held_as_a_fragment(self):
        """The sentence is prose and gets rewritten. The rule id is the meaning."""
        expectation = corpus.Expectation(status="UNVERIFIED", reason_holds="expected:0")

        self.assertIsNone(
            expectation.disagreement("UNVERIFIED", "Missing: expected:0, expected:1", None)
        )
        self.assertIn(
            "does not hold", expectation.disagreement("UNVERIFIED", "Missing: expected:4", None)
        )

    def test_a_wrong_status_is_reported_before_anything_else(self):
        expectation = corpus.Expectation(status="PASSED", reason_holds="never read")

        self.assertEqual(
            expectation.disagreement("FAILED", "", None), "status FAILED, expected PASSED"
        )


class ScreenFixtureTests(unittest.TestCase):
    """A screen fixture carries its screenshot, and one rule that inverts another."""

    def _fixture(self, **overrides) -> dict:
        fixture = {
            "id": "cart-one-burger",
            "why": "One item at $9.99 and a subtotal to match.",
            "screenshot": "cart.png",
            "rules": [
                {"kind": "expected", "expect": "passed", "text": "The subtotal is $9.99"},
                {"kind": "expected", "expect": "failed", "inverted": True,
                 "text": "The subtotal is $19.98"},
            ],
        }
        fixture.update(overrides)
        return fixture

    def _loaded(self, **overrides):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cart.png").write_bytes(solid_png())
            _written(root, "cart", self._fixture(**overrides))
            return corpus.screen_fixtures(root)[0]

    def test_the_rules_divide_by_kind_and_keep_their_order(self):
        fixture = self._loaded(
            rules=[
                {"kind": "expected", "expect": "passed", "text": "The cart is open"},
                {"kind": "security", "expect": "passed", "text": "No card number is legible"},
                {"kind": "expected", "expect": "failed", "inverted": True,
                 "text": "The cart is empty"},
            ]
        )

        self.assertEqual(fixture.outcomes, ["The cart is open", "The cart is empty"])
        self.assertEqual(fixture.security_rules, ["No card number is legible"])

    def test_a_fixture_with_no_inverted_rule_is_refused(self):
        """Without one, a reader that agrees with everything scores full marks."""
        with self.assertRaises(corpus.CorpusError) as refused:
            self._loaded(
                rules=[{"kind": "expected", "expect": "passed", "text": "The cart is open"}]
            )

        self.assertIn("no inverted rule", str(refused.exception))

    def test_a_screen_fixture_hashes_its_screenshot_as_well_as_its_rules(self):
        """Swapping the image under a fixture changes what the rules are about."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cart.png").write_bytes(solid_png())
            _written(root, "cart", self._fixture())
            first = corpus.screen_fixtures(root)[0].digest
            (root / "cart.png").write_bytes(solid_png(fill="black"))
            second = corpus.screen_fixtures(root)[0].digest

        self.assertNotEqual(first, second)

    def test_a_directory_with_no_screen_fixtures_is_refused(self):
        with TemporaryDirectory() as directory, \
             self.assertRaises(corpus.CorpusError) as refused:
            corpus.screen_fixtures(Path(directory))

        self.assertIn("never empty", str(refused.exception))

    def test_a_screenshot_outside_the_corpus_is_refused(self):
        """The bytes are sent to a model. A fixture is not a way off the disk."""
        with TemporaryDirectory() as directory:
            root = Path(directory) / "screens"
            root.mkdir()
            (Path(directory) / "private.png").write_bytes(solid_png())
            _written(root, "cart", self._fixture(screenshot="../private.png"))

            with self.assertRaises(corpus.CorpusError) as refused:
                corpus.screen_fixtures(root)

        self.assertIn("plain file name", str(refused.exception))

    def test_a_link_pointing_out_of_the_corpus_is_refused(self):
        """A plain name inside the corpus can still point out of it."""
        with TemporaryDirectory() as directory:
            root = Path(directory) / "screens"
            root.mkdir()
            elsewhere = Path(directory) / "private.png"
            elsewhere.write_bytes(solid_png())
            (root / "cart.png").symlink_to(elsewhere)
            _written(root, "cart", self._fixture())

            with self.assertRaises(corpus.CorpusError) as refused:
                corpus.screen_fixtures(root)

        self.assertIn("is outside", str(refused.exception))

    def test_a_link_inside_the_corpus_is_refused_as_well(self):
        """A screenshot is a file. Resolving one is not reading one."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "real.png").write_bytes(solid_png())
            (root / "cart.png").symlink_to(root / "real.png")
            _written(root, "cart", self._fixture())

            with self.assertRaises(corpus.CorpusError) as refused:
                corpus.screen_fixtures(root)

        self.assertIn("is a link", str(refused.exception))

    def test_a_file_that_is_not_a_png_is_refused(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cart.png").write_text("a private note", encoding="utf-8")
            _written(root, "cart", self._fixture())

            with self.assertRaises(corpus.CorpusError) as refused:
                corpus.screen_fixtures(root)

        self.assertIn("not a PNG", str(refused.exception))

    def test_a_screenshot_named_with_another_extension_is_refused(self):
        with self.assertRaises(corpus.CorpusError) as refused:
            self._loaded(screenshot="notes.txt")

        self.assertIn("not a .png", str(refused.exception))

    def test_a_missing_screenshot_is_refused(self):
        with self.assertRaises(corpus.CorpusError) as refused:
            self._loaded(screenshot="absent.png")

        self.assertIn("absent.png", str(refused.exception))

    def test_an_answer_outside_the_three_is_refused(self):
        with self.assertRaises(corpus.CorpusError) as refused:
            self._loaded(
                rules=[
                    {"kind": "expected", "expect": "probably", "text": "The cart is open"},
                    {"kind": "expected", "expect": "failed", "inverted": True,
                     "text": "The cart is empty"},
                ]
            )

        self.assertIn("rules.0.expect", str(refused.exception))

    def test_a_kind_outside_the_two_is_refused(self):
        with self.assertRaises(corpus.CorpusError) as refused:
            self._loaded(
                rules=[
                    {"kind": "performance", "expect": "passed", "text": "It is quick"},
                    {"kind": "expected", "expect": "failed", "inverted": True,
                     "text": "The cart is empty"},
                ]
            )

        self.assertIn("rules.0.kind", str(refused.exception))


if __name__ == "__main__":
    unittest.main()
