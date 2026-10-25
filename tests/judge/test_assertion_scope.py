"""What evidence settles an expected outcome, and when nothing can."""

import unittest

from fakes import device_result, walk


def _outcome(statement, scope=None):
    from yewo.testing.models import AssertionScope, ExpectedOutcome

    if scope is None:
        return ExpectedOutcome(statement=statement)
    return ExpectedOutcome(statement=statement, scope=AssertionScope(scope))


class ScopeDefaultTests(unittest.TestCase):
    def test_a_bare_string_is_a_claim_about_one_screen(self):
        """Journeys and older suites carry plain strings."""
        from yewo.testing.models import AssertionScope, GeneratedTestCase

        case = GeneratedTestCase(
            test_id="t",
            scenario_name="s",
            cell="functional.happy_path.0",
            category="functional",
            scenario_type="happy_path",
            priority="P0",
            steps=walk("Open the app."),
            expected_outcomes=["The order status screen shows CONFIRMING"],
        )

        self.assertEqual(case.expected_outcomes[0].statement.split()[0], "The")
        self.assertIs(case.expected_outcomes[0].scope, AssertionScope.CURRENT_SCREEN)


class AuditorScopeTests(unittest.TestCase):
    """The auditor sees one screenshot, so it is asked only what that settles."""

    def test_only_screen_outcomes_reach_the_auditor(self):
        from yewo.testing.judge import rules

        listing = rules.every_rule(
            rules.on_screen_only(
                [
                    _outcome("CONFIRMING is visible"),
                    _outcome("The login never hung", "run_timing"),
                ]
            ),
            [],
        )

        self.assertIn("CONFIRMING is visible", listing)
        self.assertNotIn("never hung", listing)

    def test_an_unasked_outcome_does_not_renumber_the_others(self):
        """A verdict's id is its position, so dropping one moves every later rule."""
        from yewo.testing.judge import rules

        listing = rules.every_rule(
            rules.on_screen_only(
                [
                    _outcome("The login never hung", "run_timing"),
                    _outcome("CONFIRMING is visible"),
                ]
            ),
            [],
        )

        self.assertIn("[expected:1] CONFIRMING is visible", listing)


class UnprovableOutcomeTests(unittest.TestCase):
    """An outcome nothing measures is named, not counted as a shortfall."""

    def test_a_timing_outcome_is_unprovable(self):
        from yewo.testing.judge import rules

        self.assertEqual(
            rules.unprovable([_outcome("No hang during login", "run_timing")]),
            ["expected:0"],
        )

    def test_a_screen_outcome_is_provable(self):
        from yewo.testing.judge import rules

        self.assertEqual(rules.unprovable([_outcome("CONFIRMING is visible")]), [])

    def test_the_scenario_says_what_could_not_be_settled(self):
        from yewo.testing.qa_agent import QAAgent

        status, reason, _cause = QAAgent.__new__(QAAgent)._determine_status(
            run_result=device_result(),
            stop_reason=None,
            all_verifications=[],
            all_issues=[],
            qa_outputs=[
                __import__("types").SimpleNamespace(
                    is_scenario_feasible=True, infeasibility_reason=None
                )
            ],
            assessments=[__import__("types").SimpleNamespace()],
            required_outcomes=[_outcome("No hang during login", "run_timing")],
            security_rules=[],
        )

        self.assertEqual(status, "UNVERIFIED")
        self.assertIn("expected:0", reason)
        self.assertIn("measures no part of that", reason)
