"""The id scheme the auditor and the gate agree on, and the drift it prevents."""

import unittest

from yewo.testing import taxonomy


class RuleIdTests(unittest.TestCase):
    def test_an_id_names_its_kind_and_position(self):
        from yewo.testing.judge import rules

        self.assertEqual(rules.rule_id(rules.SECURITY, 1), "security:1")
        self.assertEqual(rules.rule_id(rules.EXPECTED, 0), "expected:0")

    def test_ids_cover_every_rule(self):
        from yewo.testing.judge import rules

        self.assertEqual(rules.ids(["a", "b"], rules.EXPECTED), {"expected:0", "expected:1"})

    def test_a_kind_is_recognised_from_its_id(self):
        from yewo.testing.judge import rules

        self.assertTrue(rules.is_kind("security:3", rules.SECURITY))
        self.assertFalse(rules.is_kind("expected:3", rules.SECURITY))
        self.assertFalse(rules.is_kind(None, rules.SECURITY))


class RuleListingTests(unittest.TestCase):
    def test_both_kinds_are_listed_with_ids(self):
        from yewo.testing.judge import rules

        listing = rules.every_rule(["Order confirmed"], ["Card masked", "CVV masked"])

        self.assertIn("[expected:0] Order confirmed", listing)
        self.assertIn("[security:0] Card masked", listing)
        self.assertIn("[security:1] CVV masked", listing)

    def test_a_scenario_with_no_rules_says_so(self):
        from yewo.testing.judge import rules

        self.assertEqual(rules.every_rule([], []), rules.NONE_SPECIFIED)
        self.assertEqual(rules.listing([], rules.SECURITY), rules.NONE_SPECIFIED)


class AuditorNamingTests(unittest.TestCase):
    """The auditor echoes an id back and the gate matches on it.

    So the auditor has to name every rule exactly as `rules` spells it. Drift
    between the two would fail nowhere: every verdict would stop matching, and
    every scenario would report missing evidence while its assertions passed.
    """

    OUTCOMES = ["Order is confirmed"]
    SECURITY = ["The card number is masked"]

    def _auditor_prompt(self):
        """What the auditor sends, not what it ought to send."""
        from types import SimpleNamespace
        from unittest.mock import Mock

        from yewo.testing.judge.qa_auditor import ScreenAssessment, QAAuditor

        llm = SimpleNamespace(
            generate_response=Mock(return_value=ScreenAssessment(screen_name="Login"))
        )
        QAAuditor(llm, self.OUTCOMES, self.SECURITY).audit(b"png", "screen")

        (messages, _model), _ = llm.generate_response.call_args
        return " ".join(
            message.to_dict()["content"]
            for message in messages
            if isinstance(message.to_dict()["content"], str)
        )

    def test_the_auditor_names_every_rule_as_the_scheme_spells_it(self):
        from yewo.testing.judge import rules

        auditor = self._auditor_prompt()

        for line in rules.format_rules_for_prompt(
            self.OUTCOMES, rules.EXPECTED
        ) + rules.format_rules_for_prompt(self.SECURITY, rules.SECURITY):
            self.assertIn(line, auditor, "the auditor names this rule differently")


class SecurityAuditPolicyTests(unittest.TestCase):
    """Which scenarios have to read every screen they walk.

    A security rule is required only once some screen reported it as
    applicable. So a screen nobody audits takes its rules out of the
    required set, and a violation there passes. Security rules are declared
    per journey, so the taxonomy cell alone does not say who is exposed.
    """

    def test_a_plain_functional_cell_with_a_journey_rule_must_audit(self):
        """The common case, and the one a cell-only rule would miss."""
        self.assertTrue(
            taxonomy.must_audit_every_screen(
                taxonomy.Category.FUNCTIONAL,
                taxonomy.ScenarioType.HAPPY_PATH,
                ["A password must never be shown in plain text."],
            )
        )

    def test_a_cell_whose_claims_apply_everywhere_must_audit_without_rules(self):
        self.assertTrue(
            taxonomy.must_audit_every_screen(
                taxonomy.Category.SECURITY, taxonomy.ScenarioType.HAPPY_PATH, []
            )
        )

    def test_a_scenario_with_no_security_rule_need_not(self):
        """Its expected outcomes stay required whatever it skips, so a
        skipped screen can only cost evidence and never buy a pass."""
        self.assertFalse(
            taxonomy.must_audit_every_screen(
                taxonomy.Category.FUNCTIONAL, taxonomy.ScenarioType.HAPPY_PATH, []
            )
        )
