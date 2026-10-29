"""What the report calls things, and what it gathers into one row.

Separate from the page's structure. These hold the words a reader acts on:
the severity on a row, the number on a rule, and whether two reports of one
category are one defect or two.
"""

import tempfile
import unittest
from pathlib import Path

from test_html_report import journey, page, scenario, step

from yewo.testing.models import QAIssue


class HowIssuesAreGatheredTests(unittest.TestCase):
    """A category is not a defect identity."""

    def _issue(self, **extra) -> QAIssue:
        fields = {
            "issue_type": "layout_overflow",
            "severity": "low",
            "description": "Something overflows.",
            "current_screen": "Basket",
        }
        fields.update(extra)
        return QAIssue(**fields)

    def _found(self, issues):
        from yewo.testing.report.page.view import _findings

        return _findings(journey(scenarios=[scenario(detected_issues=issues)]))

    def test_two_unrelated_issues_of_one_kind_stay_apart(self):
        """Gathering them invents a finding neither auditor reported."""
        found = self._found([
            self._issue(severity="critical", current_screen="Checkout",
                        description="The checkout total is clipped."),
            self._issue(severity="low", current_screen="Profile",
                        description="The biography overflows its card."),
        ])

        self.assertEqual(len(found), 2)
        worst = found[0]
        self.assertEqual(worst.severity, "Critical")
        self.assertIn("checkout total", worst.description)

    def test_the_same_report_twice_is_one_row(self):
        """The auditor is asked on arrival and on departure."""
        found = self._found([self._issue(), self._issue()])

        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].occurrence_count, 2)

    def _linked(self, issues, steps):
        from yewo.testing.report.page.view import _findings

        report = journey(scenarios=[scenario(steps=steps, detected_issues=issues)])
        return _findings(report)[0].anchor

    def test_a_finding_points_at_the_step_it_was_recorded_on(self):
        """Not the first step that shares the screen name.

        A run reaches one screen many times. The record keeps which step
        the audit ran after, and only that answers where to look.
        """
        anchor = self._linked(
            [self._issue(current_screen="Basket", step=2)],
            [step(1, "Open the basket"), step(2, "Tap Pay")],
        )

        self.assertTrue(anchor.endswith("-step-2"), anchor)

    def test_a_finding_with_no_recorded_step_offers_no_link(self):
        """A link to the wrong screenshot is worse than no link."""
        anchor = self._linked(
            [self._issue(current_screen="Basket")],
            [step(1, "Open the basket"), step(2, "Tap Pay")],
        )

        self.assertEqual(anchor, "")

    def test_the_step_it_points_at_is_in_the_page(self):
        report = journey(
            scenarios=[
                scenario(
                    steps=[step(1, "Open the basket"), step(2, "Tap Pay")],
                    detected_issues=[self._issue(current_screen="Basket", step=2)],
                )
            ]
        )

        written = page(report)

        self.assertIn('href="#ts_1-step-2">View evidence</a>', written)
        self.assertIn('id="ts_1-step-2"', written)

    def test_the_kind_is_written_as_words_rather_than_as_an_enum(self):
        written = page(journey(scenarios=[scenario(detected_issues=[self._issue()])]))

        self.assertNotIn("IssueKind.", written)
        self.assertIn("Layout Overflow", written)


class WhatTheMachineReportsSayTests(unittest.TestCase):
    """Markdown and JUnit are read by a person and by CI, not by Python."""

    def _written(self, directory: str):
        from yewo.testing.report.reports import save_junit_report, save_markdown_report

        report = journey(
            scenarios=[
                scenario(
                    status="FAILED",
                    detected_issues=[
                        QAIssue(issue_type="layout_overflow", severity="high",
                                description="The total is clipped.",
                                current_screen="Checkout")
                    ],
                )
            ]
        )
        out = Path(directory)
        return (
            save_markdown_report(report, out).read_text(encoding="utf-8"),
            save_junit_report(report, out).read_text(encoding="utf-8"),
        )

    def test_neither_report_prints_a_python_enum(self):
        with tempfile.TemporaryDirectory() as directory:
            markdown, junit = self._written(directory)

        for written in (markdown, junit):
            self.assertNotIn("IssueKind.", written)
            self.assertIn("layout_overflow", written)


class WhatTheReaderIsAskedToLookAtTests(unittest.TestCase):
    def test_rules_are_numbered_from_one(self):
        """The stored id indexes a list. A reader is not reading an array."""
        from yewo.testing.report.page.view import _format_rule_display_name

        self.assertEqual(_format_rule_display_name("expected:0"), "Expected outcome 1")
        self.assertEqual(_format_rule_display_name("security:2"), "Security rule 3")

    def test_an_id_that_is_not_an_index_is_left_alone(self):
        from yewo.testing.report.page.view import _format_rule_display_name

        self.assertEqual(_format_rule_display_name("dynamic:abc"), "dynamic:abc")

    def test_an_action_that_failed_inside_a_passing_scenario_says_recovered(self):
        """Only an explicit successful transaction can make this claim."""
        from yewo.agent.run.recovery import (
            RecoveryCause,
            RecoveryMode,
            RecoveryTransaction,
        )

        recovery = RecoveryTransaction(
            failed_step=1,
            opened_step=2,
            original_intent="Pay",
            failure_reason="The action failed.",
            cause=RecoveryCause.GROUNDING_DRIFT,
            mode=RecoveryMode.MODEL_PROPOSAL,
        )
        recovery.recover("The original postcondition passed.")
        report = journey(
            scenarios=[
                scenario(status="PASSED",
                         steps=[step(1, "Tap Pay", success=False), step(2, "Tap Pay")],
                         recovery_transactions=[recovery])
            ]
        )

        written = page(report)

        self.assertIn("Action failed · recovered", written)

    def test_a_passing_scenario_does_not_imply_a_failed_action_recovered(self):
        report = journey(
            scenarios=[
                scenario(status="PASSED",
                         steps=[step(1, "Tap Pay", success=False), step(2, "Tap Pay")])
            ]
        )

        written = page(report)

        self.assertNotIn("Action failed · recovered", written)

    def test_an_action_that_failed_a_scenario_is_not_called_recovered(self):
        report = journey(
            scenarios=[
                scenario(status="FAILED", steps=[step(1, "Tap Pay", success=False)])
            ]
        )

        written = page(report)

        self.assertIn("Action failed", written)
        self.assertNotIn("recovered", written)

    def test_one_scenario_gets_no_navigation_rail(self):
        """It would repeat the heading below it and narrow every step row."""
        written = page(journey(scenarios=[scenario()]))

        self.assertNotIn('class="rail"', written)

    def test_several_scenarios_keep_the_rail_with_status_in_words(self):
        report = journey(
            scenarios=[scenario(test_id="TS_1"), scenario(test_id="TS_2", status="FAILED")]
        )

        written = page(report)

        self.assertIn('class="rail"', written)
        self.assertIn("Failed · 0 steps", written)
