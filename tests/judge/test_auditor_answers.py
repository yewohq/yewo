"""Tying one verdict back to the rule it answers.

A verdict names its rule by the id the prompt gave it, and the gate matches
on that id alone. An answer that names the rule some other way is evidence
nothing reads, which reports as evidence the run never gathered.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock


def _assessment(*verifications):
    from yewo.testing.judge.qa_auditor import ScreenAssessment

    return ScreenAssessment(screen_name="Cart", rule_verifications=list(verifications))


def _verdict(rule, rule_id=None, status="passed"):
    from yewo.testing.models import RuleVerification

    return RuleVerification(rule=rule, rule_id=rule_id, status=status, reasoning="what it saw")


def _llm(*answers):
    """A client that answers once per call, in order."""
    return SimpleNamespace(generate_response=Mock(side_effect=list(answers)))


def _auditor(llm, outcomes=("Order confirmed",), security=("Card masked",)):
    from yewo.testing.judge.qa_auditor import QAAuditor

    return QAAuditor(llm, list(outcomes), list(security))


class NamingTheRuleAnsweredTests(unittest.TestCase):
    """A verdict that quotes its rule instead of its id still counts.

    Observed on a working model: every rule answered correctly, each quoted
    word for word, and `rule_id` null on all of them. The words are the words
    the prompt listed, so the id is a lookup and not a guess.
    """

    def test_a_verdict_that_quotes_its_rule_is_matched_to_it(self):
        llm = _llm(_assessment(_verdict("Order confirmed")))

        assessment = _auditor(llm, security=[]).audit(b"png", screen_id=None)

        self.assertEqual(assessment.rule_verifications[0].rule_id, "expected:0")

    def test_a_security_rule_is_matched_to_its_own_kind(self):
        llm = _llm(_assessment(_verdict("Card masked")))

        assessment = _auditor(llm, outcomes=[]).audit(b"png", screen_id=None)

        self.assertEqual(assessment.rule_verifications[0].rule_id, "security:0")

    def test_the_words_are_matched_whatever_space_surrounds_them(self):
        llm = _llm(_assessment(_verdict("  Order confirmed  ")))

        assessment = _auditor(llm, security=[]).audit(b"png", screen_id=None)

        self.assertEqual(assessment.rule_verifications[0].rule_id, "expected:0")

    def test_two_rules_reading_alike_are_not_resolved(self):
        """Settling the wrong rule is worse than settling neither."""
        llm = _llm(
            _assessment(_verdict("Card masked"), _verdict("Other")),
            _assessment(_verdict("Card masked"), _verdict("Other")),
        )

        assessment = _auditor(
            llm, outcomes=["Card masked", "Other"], security=["Card masked"]
        ).audit(b"png", screen_id=None)

        self.assertIsNone(assessment.rule_verifications[0].rule_id)
        self.assertEqual(assessment.rule_verifications[1].rule_id, "expected:1")

    def test_a_verdict_already_naming_its_rule_keeps_that_name(self):
        llm = _llm(
            _assessment(
                _verdict("Order confirmed", rule_id="expected:0"),
                _verdict("Card masked", rule_id="security:0"),
            )
        )

        assessment = _auditor(llm).audit(b"png", screen_id=None)

        self.assertEqual(assessment.rule_verifications[0].rule_id, "expected:0")

    def test_a_wrong_id_is_recovered_from_the_exact_rule_words(self):
        """A model may copy a rule's text prefix instead of its bracketed id."""
        llm = _llm(
            _assessment(
                _verdict("Order confirmed", rule_id="current_screen:0"),
                _verdict("Card masked", rule_id="secret_rule:0"),
            )
        )

        assessment = _auditor(llm).audit(b"png", screen_id=None)

        self.assertEqual(
            [verdict.rule_id for verdict in assessment.rule_verifications],
            ["expected:0", "security:0"],
        )
        self.assertEqual(llm.generate_response.call_count, 1)

    def test_an_outcome_no_screenshot_could_settle_keeps_the_others_in_place(self):
        """The auditor is shown the on-screen outcomes, blanks and all.

        An outcome about reaching a screen is withheld and its place is kept,
        so the second outcome is still `expected:1`. A map built from the
        listed rules has to number them the same way.
        """
        from yewo.testing.models import AssertionScope, ExpectedOutcome

        reached = ExpectedOutcome(
            statement="The cart screen is reached", scope=AssertionScope.TRANSITION
        )
        visible = ExpectedOutcome(statement="The subtotal is $9.99")
        llm = _llm(_assessment(_verdict("The subtotal is $9.99")))

        assessment = _auditor(llm, outcomes=[reached, visible], security=[]).audit(
            b"png", screen_id=None
        )

        self.assertEqual(assessment.rule_verifications[0].rule_id, "expected:1")


class AskingAgainTests(unittest.TestCase):
    """An audit that does not answer every listed rule is incomplete.

    A missing rule would report unevidenced, and the scenario could fail on
    evidence the app had given.
    """

    def test_an_audit_that_names_no_listed_rule_is_asked_again(self):
        llm = _llm(
            _assessment(_verdict("Something else", rule_id="expected:9")),
            _assessment(_verdict("Order confirmed", rule_id="expected:0")),
        )

        assessment = _auditor(llm).audit(b"png", screen_id=None)

        self.assertEqual(llm.generate_response.call_count, 2)
        self.assertEqual(assessment.rule_verifications[0].rule_id, "expected:0")

    def test_an_empty_answer_is_asked_again(self):
        llm = _llm(_assessment(), _assessment(_verdict("Order confirmed")))

        assessment = _auditor(llm).audit(b"png", screen_id=None)

        self.assertEqual(llm.generate_response.call_count, 2)
        self.assertEqual(assessment.rule_verifications[0].rule_id, "expected:0")

    def test_the_audit_is_asked_again_only_once(self):
        """A model that answers nothing twice has answered nothing.

        The evidence gate then reports the outcomes as unevidenced, which is
        the honest reading. Asking until it answers would spend a run's
        budget on one screen.
        """
        llm = _llm(_assessment(), _assessment())

        assessment = _auditor(llm).audit(b"png", screen_id=None)

        self.assertEqual(llm.generate_response.call_count, 2)
        self.assertEqual(assessment.rule_verifications, [])

    def test_a_partial_audit_is_asked_again(self):
        llm = _llm(
            _assessment(_verdict("Order confirmed")),
            _assessment(
                _verdict("Order confirmed"),
                _verdict("Card masked"),
            ),
        )

        assessment = _auditor(llm).audit(b"png", screen_id=None)

        self.assertEqual(llm.generate_response.call_count, 2)
        self.assertEqual(
            [verdict.rule_id for verdict in assessment.rule_verifications],
            ["expected:0", "security:0"],
        )

    def test_an_audit_that_answers_every_rule_is_not_asked_again(self):
        llm = _llm(
            _assessment(
                _verdict("Order confirmed"),
                _verdict("Card masked"),
            )
        )

        _auditor(llm).audit(b"png", screen_id=None)

        self.assertEqual(llm.generate_response.call_count, 1)

    def test_an_audit_with_no_rules_to_name_is_not_asked_again(self):
        """With nothing listed, naming nothing is the correct answer."""
        llm = _llm(_assessment())

        _auditor(llm, outcomes=[], security=[]).audit(b"png", screen_id=None)

        self.assertEqual(llm.generate_response.call_count, 1)

    def test_the_answer_kept_for_the_screen_is_the_one_that_answered(self):
        """A cached non-answer would make one bad call cover every later look."""
        llm = _llm(_assessment(), _assessment(_verdict("Order confirmed")))
        auditor = _auditor(llm)

        auditor.audit(b"png", screen_id="abc")
        again = auditor.audit(b"png", screen_id="abc")

        self.assertEqual(llm.generate_response.call_count, 2)
        self.assertEqual(again.rule_verifications[0].rule_id, "expected:0")


if __name__ == "__main__":
    unittest.main()


class WhereAnIssueWasSeenTests(unittest.TestCase):
    """A report can only link a finding to evidence the run recorded."""

    def _assessment(self, *issues):
        from types import SimpleNamespace

        return SimpleNamespace(detected_issues=list(issues))

    def _issue(self, **extra):
        from yewo.testing.models import QAIssue

        fields = {
            "issue_type": "layout_overflow",
            "severity": "low",
            "description": "Something overflows.",
        }
        fields.update(extra)
        return QAIssue(**fields)

    def test_an_issue_is_stamped_with_the_step_it_was_found_after(self):
        from yewo.testing.judge.screen_audits import record_issue_step

        issue = self._issue()
        record_issue_step([self._assessment(issue)], 4)

        self.assertEqual(issue.step, 4)

    def test_an_issue_already_stamped_keeps_its_step(self):
        """The closing look repeats what an earlier look returned."""
        from yewo.testing.judge.screen_audits import record_issue_step

        issue = self._issue(step=2)
        record_issue_step([self._assessment(issue)], 9)

        self.assertEqual(issue.step, 2)
