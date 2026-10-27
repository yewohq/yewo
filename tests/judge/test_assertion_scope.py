"""What evidence settles an expected outcome, and when nothing can."""

import unittest

from fakes import device_result, walk


def _outcome(statement, scope=None):
    from yewo.testing.models import AssertionScope, ExpectedOutcome
    from yewo.metrics import Metric

    if scope is None:
        return ExpectedOutcome(statement=statement)
    fields = {"statement": statement, "scope": AssertionScope(scope)}
    if fields["scope"] is AssertionScope.MEASURED:
        fields.update(metric=Metric.APP_LAUNCH_TTID_MS, threshold=800)
    return ExpectedOutcome(**fields)


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

    def test_provider_statement_and_scope_aliases_are_normalized(self):
        from yewo.testing.models import AssertionScope, ExpectedOutcome

        outcome = ExpectedOutcome.model_validate(
            {"description": "Home is visible.", "kind": "current_screen"}
        )

        self.assertEqual(outcome.statement, "Home is visible.")
        self.assertIs(outcome.scope, AssertionScope.CURRENT_SCREEN)


class AuditorScopeTests(unittest.TestCase):
    """The auditor sees one screenshot, so it is asked only what that settles."""

    def test_only_screen_outcomes_reach_the_auditor(self):
        from yewo.testing.judge import rules

        listing = rules.every_rule(
            rules.on_screen_only(
                [
                    _outcome("CONFIRMING is visible"),
                    _outcome("The login opens within the limit", "measured"),
                ]
            ),
            [],
        )

        self.assertIn("CONFIRMING is visible", listing)
        self.assertNotIn("within the limit", listing)

    def test_an_unasked_outcome_does_not_renumber_the_others(self):
        """A verdict's id is its position, so dropping one moves every later rule."""
        from yewo.testing.judge import rules

        listing = rules.every_rule(
            rules.on_screen_only(
                [
                    _outcome("The login opens within the limit", "measured"),
                    _outcome("CONFIRMING is visible"),
                ]
            ),
            [],
        )

        self.assertIn("[expected:1] CONFIRMING is visible", listing)


class MeasuredOutcomeTests(unittest.TestCase):
    """A measured outcome needs numeric evidence rather than screen evidence."""

    def test_a_screen_outcome_is_provable(self):
        self.assertEqual(_outcome("CONFIRMING is visible").scope, "current_screen")

    def test_a_missing_sample_is_missing_evidence(self):
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
            required_outcomes=[_outcome("Login opens within the limit", "measured")],
            security_rules=[],
        )

        self.assertEqual(status, "UNVERIFIED")
        self.assertIn("expected:0", reason)
        self.assertIn("Required verification evidence is missing", reason)
