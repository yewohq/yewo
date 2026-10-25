"""Which half of the loop owns an answer, and when an answer still counts."""

import unittest

from fakes import device_result


class AuditorOwnershipTests(unittest.TestCase):
    """Which half of the loop owns which answer."""

    def test_the_planner_is_not_asked_for_verdicts(self):
        """A planner grading the work it just chose is not a second opinion."""
        from yewo.testing.qa_planner import QAStepResponse

        fields = QAStepResponse.model_fields

        self.assertNotIn("rule_verifications", fields)
        self.assertNotIn("detected_issues", fields)

    def test_the_auditor_is_asked_for_them(self):
        from yewo.testing.judge.qa_auditor import ScreenAssessment

        fields = ScreenAssessment.model_fields

        self.assertIn("rule_verifications", fields)
        self.assertIn("detected_issues", fields)

    def test_completion_stays_with_the_planner(self):
        """Only the planner is told the goal, so only it can say the goal is met."""
        from yewo.testing.judge.qa_auditor import ScreenAssessment
        from yewo.testing.qa_planner import QAStepResponse

        self.assertIn("goal_satisfied", QAStepResponse.model_fields)
        self.assertNotIn("goal_satisfied", ScreenAssessment.model_fields)
        self.assertNotIn("goal_evidence", ScreenAssessment.model_fields)

    def test_feasibility_stays_with_the_planner(self):
        """Whether the agent can proceed is not a statement about the app."""
        from yewo.testing.judge.qa_auditor import ScreenAssessment
        from yewo.testing.qa_planner import QAStepResponse

        self.assertIn("is_scenario_feasible", QAStepResponse.model_fields)
        self.assertNotIn("is_scenario_feasible", ScreenAssessment.model_fields)


class CrashRankingTests(unittest.TestCase):
    """A crash is something the app did. The rest describe the run."""

    def _verdict_for(self, **overrides):
        from yewo.testing.models import QAIssue
        from yewo.testing.judge.verdicts import _terminal_outcome

        values = {
            "run_result": device_result(status="hook_stop"),
            "stop_reason": "Scenario is infeasible: the screen cannot be reached",
            "verifications": [],
            "issues": [
                QAIssue(
                    issue_type="crash_screen",
                    severity="critical",
                    description="blank screen",
                )
            ],
            "outputs": [],
        }
        values.update(overrides)
        status, reason, _cause = _terminal_outcome(**values) or (None, None, None)
        return status, reason

    def test_a_crash_outranks_an_infeasible_verdict(self):
        """Reported as skipped, a crash looks like a scenario nobody ran."""
        status, reason = self._verdict_for()

        self.assertEqual(status, "FAILED")
        self.assertIn("crash", reason.lower())

    def test_infeasibility_still_reports_when_nothing_crashed(self):
        status, _reason = self._verdict_for(issues=[])

        self.assertEqual(status, "SKIPPED")


class DurableOutcomeEvidenceTests(unittest.TestCase):
    """An outcome seen in passing stays proven after the screen is left.

    A long address is accepted on the checkout screen. The order status screen
    that follows has no address field, so the rule reports untested there.
    Reading that as the verdict lets walking away erase the proof.
    """

    def _verification(self, rule_id, status, applicable=True):
        from yewo.testing.models import RuleVerification

        return RuleVerification(
            rule="The address field accepts the long input",
            rule_id=rule_id,
            status=status,
            applicable=applicable,
            reasoning="because",
        )

    def _missing(self, verifications):
        from yewo.testing.judge.verdicts import _missing_required_ids

        return _missing_required_ids(
            verifications, ["The address field accepts the long input"], []
        )

    def test_a_pass_survives_a_screen_the_outcome_cannot_be_seen_on(self):
        missing = self._missing(
            [
                self._verification("expected:0", "passed"),
                self._verification("expected:0", "untested", applicable=False),
            ]
        )

        self.assertEqual(missing, [])

    def test_an_outcome_never_seen_is_still_missing(self):
        missing = self._missing(
            [self._verification("expected:0", "untested", applicable=False)]
        )

        self.assertEqual(missing, ["expected:0"])

    def test_an_outcome_that_later_broke_where_it_could_be_seen_is_missing(self):
        """A failure on a screen the rule applies to still overrides the pass."""
        missing = self._missing(
            [
                self._verification("expected:0", "passed"),
                self._verification("expected:0", "failed"),
            ]
        )

        self.assertEqual(missing, ["expected:0"])

    def test_an_early_not_yet_does_not_block_a_later_pass(self):
        missing = self._missing(
            [
                self._verification("expected:0", "untested"),
                self._verification("expected:0", "passed"),
            ]
        )

        self.assertEqual(missing, [])


class ExposureEvidenceTests(unittest.TestCase):
    """An exposure claim is weighed against a reading of the same screen.

    The auditor answers twice: in prose, where a bare "CVV:" caption can be
    called a plain-text CVV, and as a reading of each sensitive field, where it
    must choose between a legible value, a mask, a caption and nothing. The
    reading is the evidence.
    """

    def _issue(self, kind="sensitive_data_exposure", screen="Checkout Details"):
        from yewo.testing.models import QAIssue

        issue = QAIssue(
            issue_type=kind,
            severity="critical",
            description="The CVV is visible in plain text",
        )
        issue.current_screen = screen
        return issue

    def _finding(self, rendering, screen="Checkout Details"):
        from yewo.testing.models import SensitiveFinding

        finding = SensitiveFinding(field="CVV", rendering=rendering)
        finding.screen_name = screen
        return finding

    def _verdict_for(self, issues, findings):
        from yewo.testing.judge.verdicts import _terminal_outcome

        return _terminal_outcome(
            run_result=device_result(),
            stop_reason=None,
            verifications=[],
            issues=issues,
            outputs=[],
            findings=findings,
        )

    def test_a_caption_is_not_an_exposure(self):
        outcome = self._verdict_for([self._issue()], [self._finding("label_only")])

        self.assertIsNone(outcome)

    def test_a_mask_is_not_an_exposure(self):
        outcome = self._verdict_for([self._issue()], [self._finding("masked")])

        self.assertIsNone(outcome)

    def test_a_legible_value_is_an_exposure(self):
        outcome = self._verdict_for([self._issue()], [self._finding("value")])

        self.assertEqual(outcome[0], "FAILED")

    def test_a_claim_with_no_reading_behind_it_still_stands(self):
        """Nothing was examined, so there is nothing to weigh against."""
        outcome = self._verdict_for([self._issue()], [])

        self.assertEqual(outcome[0], "FAILED")

    def test_a_reading_of_another_screen_does_not_excuse_it(self):
        outcome = self._verdict_for(
            [self._issue(screen="Checkout Details")],
            [self._finding("masked", screen="Login Screen")],
        )

        self.assertEqual(outcome[0], "FAILED")

    def test_a_crash_is_never_weighed_this_way(self):
        outcome = self._verdict_for(
            [self._issue(kind="crash_screen")], [self._finding("masked")]
        )

        self.assertEqual(outcome[0], "FAILED")


class RenderingTests(unittest.TestCase):
    """Only one of the four readings is a leak."""

    def test_which_readings_are_exposures(self):
        from yewo.testing.models import Rendering, SensitiveFinding

        leaks = {
            rendering: SensitiveFinding(field="CVV", rendering=rendering).is_exposure()
            for rendering in Rendering
        }

        self.assertEqual(
            leaks,
            {
                Rendering.VALUE: True,
                Rendering.MASKED: False,
                Rendering.LABEL_ONLY: False,
                Rendering.ABSENT: False,
            },
        )
