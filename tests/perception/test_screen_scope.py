"""Which screen a mark came from, and whether it may decide the verdict.

`applicable` is the auditor's own claim and nothing can check it. So a mark
reaches only the screen it was made about. A mark about the screen an outcome
names still decides it.
"""

import unittest
from types import SimpleNamespace

from fakes import device_result


class CrossedOnWhichScreenTests(unittest.TestCase):
    """`applicable` is the auditor's own claim, and nothing can check it.

    So one wrong claim must reach no further than the screen it was made
    about. A claim about the screen an outcome names still decides it.
    """

    def _verification(self, rule_id, status, screen, applicable=True):
        from yewo.testing.models import RuleVerification

        return RuleVerification(
            rule=rule_id,
            rule_id=rule_id,
            status=status,
            applicable=applicable,
            reasoning="because",
            screen_name=screen,
        )

    def _verdict(self, verifications, outcomes=None, security=None):
        from yewo.testing.qa_agent import QAAgent

        status, reason, _cause = QAAgent.__new__(QAAgent)._determine_status(
            run_result=device_result(),
            stop_reason=None,
            all_verifications=verifications,
            all_issues=[],
            qa_outputs=[
                SimpleNamespace(is_scenario_feasible=True, infeasibility_reason=None)
            ],
            assessments=[SimpleNamespace(rule_verifications=[], detected_issues=[])],
            required_outcomes=outcomes if outcomes is not None else ["Location screen is visible"],
            security_rules=security or [],
        )
        return status, reason

    def test_a_cross_from_another_screen_does_not_erase_the_tick(self):
        """Taken from a run: one wrong answer in 17 failed a working journey.

        expected:1 is about the Location screen. The auditor answered it on the
        Home screen as failed, which its own prompt forbids.
        """
        marks = [self._verification("expected:0", "untested", "Splash", applicable=False)]
        marks.append(self._verification("expected:0", "passed", "Allow Location Access"))
        marks.append(self._verification("expected:0", "failed", "Home screen"))
        marks += [
            self._verification("expected:0", "untested", screen, applicable=False)
            for screen in ("My Cart", "Select Payment", "Checkout Details")
        ]

        status, reason = self._verdict(marks)

        self.assertEqual(status, "PASSED", reason)

    def test_a_cross_from_the_same_screen_still_decides_it(self):
        """There the outcome stopped holding where it could be seen."""
        marks = [
            self._verification("expected:0", "passed", "Checkout Details"),
            self._verification("expected:0", "failed", "Checkout Details"),
        ]

        status, reason = self._verdict(marks)

        self.assertEqual(status, "FAILED")
        self.assertIn("expected:0", reason)

    def test_a_cross_with_no_tick_anywhere_decides_it(self):
        marks = [self._verification("expected:0", "failed", "Order Status")]

        status, reason = self._verdict(marks)

        self.assertEqual(status, "FAILED")

    def test_an_outcome_only_ever_crossed_elsewhere_is_not_evidenced(self):
        """Passing nowhere is not the same as passing somewhere."""
        marks = [self._verification("expected:0", "untested", "Home", applicable=False)]

        status, reason = self._verdict(marks)

        self.assertEqual(status, "UNVERIFIED")
        self.assertIn("expected:0", reason)

    def test_a_security_rule_crossed_anywhere_still_fails_the_scenario(self):
        """These describe what must never happen, so one cross is terminal."""
        marks = [
            self._verification("expected:0", "passed", "Checkout Details"),
            self._verification("security:0", "passed", "Sign-in"),
            self._verification("security:0", "failed", "Order Status"),
        ]

        status, reason = self._verdict(marks, security=["card number stays masked"])

        self.assertEqual(status, "FAILED")
        self.assertIn("Security rule violated", reason)

    def test_an_answer_with_no_screen_named_is_still_weighed(self):
        """The auditor names the screen, and a judge of the order may not."""
        marks = [self._verification("expected:0", "passed", None)]

        status, reason = self._verdict(marks)

        self.assertEqual(status, "PASSED", reason)
