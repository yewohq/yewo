"""The words a prose step may open with, and where they come from.

The list is mapped from `ActionType` and `CapabilityKind` rather than typed
out. A hand-written one allowed five verbs for capabilities that do not
exist, so a model was invited to write steps the agent would fail on.

The scan itself is here too. It holds a sentence to opening with an action
and to carrying no claim, which is as much as an opening word can decide. A
typed step needs none of it.
"""

import unittest

from yewo.agent.actuation.actions import ActionType, CapabilityKind
from yewo.testing.step_language import (
    ACTION_VERBS,
    ASSERTION_VERBS,
    RUN_VERBS,
    VERBS_BY_ACTION,
    VERBS_BY_CAPABILITY,
    leaked_assertion,
    unknown_action,
)


class DerivationTests(unittest.TestCase):
    def test_every_action_and_capability_is_mapped(self):
        """A new capability reaches the generator without a second edit."""
        self.assertEqual(set(VERBS_BY_ACTION), set(ActionType))
        self.assertEqual(set(VERBS_BY_CAPABILITY), set(CapabilityKind))

    def test_every_capability_can_be_asked_for(self):
        """A capability with no verb is one no step can reach."""
        for kind, verbs in VERBS_BY_CAPABILITY.items():
            self.assertTrue(verbs, kind)

    def test_no_verb_names_something_yewo_cannot_do(self):
        """The hand-written list allowed all five, and none of them is shipped."""
        for absent in ("shake", "unlock", "foreground"):
            self.assertNotIn(absent, ACTION_VERBS, absent)

    def test_every_verb_traces_to_an_action_a_capability_or_the_run(self):
        mapped = {
            verb
            for group in (*VERBS_BY_ACTION.values(), *VERBS_BY_CAPABILITY.values())
            for verb in group
        }

        self.assertEqual(ACTION_VERBS - mapped, set(RUN_VERBS))


class VocabularyTests(unittest.TestCase):
    def test_a_checkbox_is_toggled_rather_than_checked(self):
        """One word means one thing. "Check" names an action and a claim."""
        self.assertNotIn("check", ACTION_VERBS)
        self.assertIn("check", ASSERTION_VERBS)
        self.assertTrue({"toggle", "tick", "select"} <= ACTION_VERBS)

    def test_no_word_is_both_an_action_and_a_claim(self):
        self.assertEqual(ACTION_VERBS & ASSERTION_VERBS, set())


class ScanTests(unittest.TestCase):
    def test_an_unknown_opening_word_is_named(self):
        self.assertEqual(unknown_action("Validate the total."), "validate")

    def test_a_known_action_opening_returns_nothing(self):
        self.assertIsNone(unknown_action("Swipe up on the list."))

    def test_a_button_label_holding_a_listed_word_is_accepted(self):
        self.assertIsNone(unknown_action("Tap the Confirm button."))
        self.assertIsNone(leaked_assertion("Tap the Confirm button."))

    def test_a_listed_word_inside_a_value_is_accepted(self):
        instruction = "Enter 'tell them and see you soon' in the Note field."

        self.assertIsNone(unknown_action(instruction))
        self.assertIsNone(leaked_assertion(instruction))

    def test_a_claim_after_a_conjunction_is_named(self):
        self.assertEqual(
            leaked_assertion("Tap Login, then ensure the home screen opens."), "ensure"
        )

    def test_a_claim_after_a_semicolon_is_named(self):
        self.assertEqual(
            leaked_assertion("Tap Login; verify the error appears."), "verify"
        )

    def test_an_empty_instruction_is_not_an_action(self):
        self.assertEqual(unknown_action("   "), "")

    def test_the_scan_decides_an_opening_word_and_nothing_more(self):
        """Named so nobody reads it as a parser. It passes this, and has to."""
        self.assertIsNone(unknown_action("Tap Login to verify the error appears."))
        self.assertIsNone(leaked_assertion("Tap Login to verify the error appears."))


if __name__ == "__main__":
    unittest.main()
