"""What one sweep of the corpus answers, and how a model's answer is classed."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fakes import solid_png

from tools.eval import corpus, tiers


def _verdict_fixture(expect: corpus.Expectation, **arguments) -> corpus.VerdictFixture:
    """A fixture built directly, so a tier test does not restate the file format."""
    values = {
        "run_result": None,
        "stop_reason": None,
        "all_verifications": [],
        "all_issues": [],
        "qa_outputs": [],
        "assessments": [1],
        "required_outcomes": [],
        "security_rules": [],
        "all_findings": [],
        "required_capabilities": [],
    }
    values.update(arguments)
    return corpus.VerdictFixture(
        id="one", why="because", expect=expect, arguments=values
    )


def _goal_reached():
    from yewo.agent.memory.history import AgentHistory
    from yewo.agent.run.outcomes import DeviceRunResult

    return DeviceRunResult(
        status="goal_reached", steps_executed=1, failure_reason=None,
        history=AgentHistory(),
    )


class DeterministicTierTests(unittest.TestCase):
    """Tier 1 needs no model, so an item is correct or it is wrong."""

    def test_a_fixture_the_judge_agrees_with_is_correct(self):
        fixture = _verdict_fixture(
            corpus.Expectation(status="UNVERIFIED", cause="scenario", checks_cause=True),
            run_result=_goal_reached(),
        )

        answer = tiers.judge_verdicts([fixture])[0]

        self.assertEqual(answer.state, tiers.CORRECT)
        self.assertTrue(answer.is_correct)

    def test_a_fixture_the_judge_disagrees_with_names_the_difference(self):
        fixture = _verdict_fixture(
            corpus.Expectation(status="PASSED"), run_result=_goal_reached()
        )

        answer = tiers.judge_verdicts([fixture])[0]

        self.assertEqual(answer.state, tiers.WRONG)
        self.assertIn("expected PASSED", answer.detail)

    def test_a_fixture_that_makes_the_judge_raise_is_wrong_and_not_a_crash(self):
        """One unreadable fixture must not stop the sweep that follows it."""
        broken = _verdict_fixture(corpus.Expectation(status="PASSED"), run_result=None)
        sound = _verdict_fixture(
            corpus.Expectation(status="UNVERIFIED"), run_result=_goal_reached()
        )

        answers = tiers.judge_verdicts([broken, sound])

        self.assertEqual(answers[0].state, tiers.WRONG)
        self.assertIn("raised", answers[0].detail)
        self.assertEqual(answers[1].state, tiers.CORRECT)


class _FakeAuditor:
    """Answers the rules it was built with, from a script of statuses."""

    def __init__(self, answers: dict, looks: list):
        self._answers = answers
        self._looks = looks

    def audit(self, screenshot: bytes, screen_id, gui_elements=()):
        from yewo.testing.judge.qa_auditor import ScreenAssessment
        from yewo.testing.models import RuleVerification

        self._looks.append(screen_id)
        turn = len(self._looks) - 1
        verifications = []
        for identifier, scripted in self._answers.items():
            status = scripted[turn % len(scripted)]
            if status is None:
                continue
            verifications.append(
                RuleVerification(
                    rule=identifier, rule_id=identifier, status=status,
                    reasoning="the fake read the screen",
                )
            )
        return ScreenAssessment(screen_name="Cart", rule_verifications=verifications)


class ScreenTierTests(unittest.TestCase):
    """Tier 2 asks a model, so one rule is read over several replicates."""

    def setUp(self):
        self.directory = TemporaryDirectory()
        root = Path(self.directory.name)
        (root / "cart.png").write_bytes(solid_png())
        self.fixture = corpus.ScreenFixture(
            id="cart",
            why="one item",
            digest="f1f2f3f4f5f6f7f8",
            screenshot=root / "cart.png",
            rules=[
                corpus.RuleAsked("The subtotal is $9.99", "expected", "passed"),
                corpus.RuleAsked("The subtotal is $19.98", "expected", "failed", True),
                corpus.RuleAsked("No card number is legible", "security", "passed"),
            ],
        )
        self.looks: list = []

    def tearDown(self):
        self.directory.cleanup()

    def _swept(self, answers: dict, replicates: int = 3):
        def build(_fixture):
            return _FakeAuditor(answers, self.looks)

        return {
            answer.item: answer
            for answer in tiers.audit_screens([self.fixture], build, replicates)
        }

    def test_a_rule_answered_as_the_corpus_says_every_time_is_correct(self):
        swept = self._swept({
            "expected:0": ["passed"], "expected:1": ["failed"], "security:0": ["passed"],
        })

        self.assertEqual(swept["cart/expected:0"].state, tiers.CORRECT)
        self.assertEqual(swept["cart/expected:1"].state, tiers.CORRECT)
        self.assertEqual(swept["cart/security:0"].state, tiers.CORRECT)

    def test_a_rule_answered_wrongly_every_time_names_what_came_back(self):
        """The inverted rule is the one a reader that agrees answers wrongly."""
        swept = self._swept({
            "expected:0": ["passed"], "expected:1": ["passed"], "security:0": ["passed"],
        })

        self.assertEqual(swept["cart/expected:1"].state, tiers.WRONG)
        self.assertIn("passed x3", swept["cart/expected:1"].detail)
        self.assertIn("wanted failed", swept["cart/expected:1"].detail)

    def test_a_rule_that_disagrees_with_itself_is_flaky(self):
        swept = self._swept({
            "expected:0": ["passed"], "expected:1": ["failed", "passed", "failed"],
            "security:0": ["passed"],
        })

        self.assertEqual(swept["cart/expected:1"].state, tiers.FLAKY)
        self.assertIn("failed x2", swept["cart/expected:1"].detail)

    def test_a_rule_no_verdict_named_is_unanswered(self):
        """Silence is not a pass and it is not a failure. It is no evidence."""
        swept = self._swept({
            "expected:0": ["passed"], "expected:1": [None], "security:0": ["passed"],
        })

        self.assertEqual(swept["cart/expected:1"].state, tiers.UNANSWERED)

    def test_a_rule_answered_only_sometimes_is_unanswered_and_not_flaky(self):
        """A look that said nothing is missing evidence, not disagreement.

        Read as flaky it would be covered by a `known_flaky` line, and a rule
        the auditor stopped answering at all would be excused by it.
        """
        swept = self._swept({
            "expected:0": ["passed", None, None], "expected:1": ["failed"],
            "security:0": ["passed"],
        })

        self.assertEqual(swept["cart/expected:0"].state, tiers.UNANSWERED)
        self.assertIn("nothing x2", swept["cart/expected:0"].detail)

    def test_a_sweep_of_no_replicates_is_refused(self):
        """Zero asks nothing, and would report every rule as gone silent."""
        with self.assertRaises(ValueError):
            tiers.audit_screens([self.fixture], lambda _f: None, replicates=0)

    def test_a_sweep_records_what_answered_each_call(self):
        """A state is an aggregate. It cannot say which call produced it."""
        from yewo.llm.usage import CallRecord

        record = CallRecord(
            model="asked/model", served_model="served/model", generation_id="g1",
            attempt=1, finish_reason="stop", latency_ms=12.0, cost=0.001,
            prompt_sha256="p1", schema_sha256="s1",
        )
        audits: list = []

        tiers.audit_screens(
            [self.fixture],
            lambda _f: _FakeAuditor({"expected:0": ["passed"]}, self.looks),
            replicates=2,
            audits=audits,
            call_for=lambda: record,
        )

        self.assertEqual([audit.replicate for audit in audits], [0, 1])
        self.assertTrue(all(a.digest == self.fixture.digest for a in audits))
        self.assertEqual(audits[0].requested_model, "asked/model")
        self.assertEqual(audits[0].served_model, "served/model")
        self.assertEqual(audits[0].prompt_sha256, "p1")
        self.assertEqual(audits[0].answered["expected:0"], "passed")
        self.assertIsNone(audits[0].answered["security:0"])

    def test_a_call_the_client_did_not_report_still_records_the_answer(self):
        """The client reports a call through a callback, and may report none."""
        audits: list = []

        tiers.audit_screens(
            [self.fixture],
            lambda _f: _FakeAuditor({"expected:0": ["passed"]}, self.looks),
            replicates=1,
            audits=audits,
            call_for=lambda: None,
        )

        self.assertIsNone(audits[0].requested_model)
        self.assertEqual(audits[0].digest, self.fixture.digest)
        self.assertEqual(audits[0].answered["expected:0"], "passed")

    def test_what_is_written_down_holds_no_copy_of_the_answer(self):
        """`answered` is derived from the assessment, so it is not written.

        A rule that got no verdict is a rule the assessment does not mention.
        A second copy of that would be one more thing to keep in step.
        """
        audits: list = []
        tiers.audit_screens(
            [self.fixture],
            lambda _f: _FakeAuditor({"expected:0": ["passed"]}, self.looks),
            replicates=1,
            audits=audits,
            call_for=lambda: None,
        )

        written = audits[0].as_dict()

        self.assertNotIn("answered", written)
        self.assertIn("assessment", written)
        self.assertIn("requested_model", written)
        self.assertIn("served_model", written)

    def test_every_answer_carries_the_fixture_it_answered(self):
        """A baseline entry is bound to the fixture, so the hash travels."""
        fixture = corpus.ScreenFixture(
            id="cart", why="one item", screenshot=self.fixture.screenshot,
            rules=self.fixture.rules, digest="abcd1234abcd1234",
        )

        answers = tiers.audit_screens(
            [fixture], lambda _f: _FakeAuditor({"expected:0": ["passed"]}, self.looks), 1
        )

        self.assertTrue(all(answer.digest == "abcd1234abcd1234" for answer in answers))

    def test_every_rule_of_the_fixture_is_reported_even_when_none_came_back(self):
        swept = self._swept({"expected:0": ["passed"]})

        self.assertEqual(len(swept), 3)
        self.assertEqual(swept["cart/security:0"].state, tiers.UNANSWERED)

    def test_each_look_is_a_fresh_auditor_and_never_a_cached_answer(self):
        """A replicate served from a cache is one answer counted twice."""
        built = []

        def build(fixture):
            built.append(fixture)
            return _FakeAuditor({"expected:0": ["passed"]}, self.looks)

        tiers.audit_screens([self.fixture], build, replicates=4)

        self.assertEqual(len(built), 4)
        self.assertEqual(self.looks, [None, None, None, None])

    def test_a_rule_is_numbered_by_its_place_among_its_own_kind(self):
        """The id is what the auditor is shown, so the corpus must number alike."""
        swept = self._swept({"expected:0": ["passed"]})

        self.assertEqual(
            sorted(swept), ["cart/expected:0", "cart/expected:1", "cart/security:0"]
        )


if __name__ == "__main__":
    unittest.main()
