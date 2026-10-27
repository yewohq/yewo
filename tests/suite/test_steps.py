"""A step says what it is, so nothing has to be read out of a sentence.

Three invariants hold here. A value entered and the sentence the planner
reads come from one object, so they cannot name different values. A device
state carries the typed capability, so what a scenario needs is read off its
own steps rather than declared beside them. And a prose action carries one
action and no claim, which is all a sentence can be held to.
"""

import unittest

from pydantic import ValidationError

from yewo.agent.actuation.actions import (
    CapabilityKind,
    NetworkCapability,
    PermissionCapability,
)
from yewo.testing import steps
from yewo.testing.steps import ActionStep, CapabilityStep, EnterStep


class EnterStepTests(unittest.TestCase):
    def test_the_sentence_and_the_value_come_from_one_object(self):
        """A separate list of values could name one and the step another."""
        step = EnterStep(field="Password", value="rejected_password")

        self.assertEqual(
            step.rendered(), "Enter '{{rejected_password}}' in the Password field."
        )

    def test_the_rendered_name_is_what_substitution_resolves(self):
        from yewo.testing.journey_env import substitute

        step = EnterStep(field="Password", value="rejected_password")

        self.assertEqual(
            substitute(step.rendered(), {"rejected_password": "abc"}),
            "Enter 'abc' in the Password field.",
        )

    def test_a_dotted_account_value_resolves_too(self):
        from yewo.testing.journey_env import substitute

        step = EnterStep(field="Email", value="actors.editor.email")

        self.assertEqual(
            substitute(step.rendered(), {"actors.editor.email": "e@example.com"}),
            "Enter 'e@example.com' in the Email field.",
        )


class CapabilityStepTests(unittest.TestCase):
    def test_what_a_scenario_needs_is_read_off_its_steps(self):
        walk = [
            ActionStep(instruction="Launch the app."),
            CapabilityStep(capability=NetworkCapability(mode="offline")),
            ActionStep(instruction="Tap the Retry button."),
        ]

        self.assertEqual(steps.capability_kinds_in(walk), [CapabilityKind.NETWORK])

    def test_a_walk_with_no_capability_needs_none(self):
        self.assertEqual(
            steps.capability_kinds_in([ActionStep(instruction="Launch the app.")]), []
        )

    def test_the_same_capability_twice_is_named_once(self):
        walk = [
            CapabilityStep(capability=NetworkCapability(mode="offline")),
            CapabilityStep(capability=NetworkCapability(mode="online")),
        ]

        self.assertEqual(steps.capability_kinds_in(walk), [CapabilityKind.NETWORK])

    def test_one_field_reads_as_a_plain_sentence(self):
        step = CapabilityStep(capability=NetworkCapability(mode="offline"))

        self.assertEqual(step.rendered(), "Set the device network to offline.")

    def test_more_fields_are_each_named(self):
        step = CapabilityStep(
            capability=PermissionCapability(permission="camera", grant=False)
        )

        self.assertIn("camera", step.rendered())
        self.assertIn("permission", step.rendered())

    def test_a_capability_the_union_does_not_hold_is_refused(self):
        """The typed capability is the real one, so it cannot be invented."""
        with self.assertRaises(ValidationError):
            CapabilityStep.model_validate(
                {"capability": {"kind": "thermal_state", "state": "critical"}}
            )


class ActionStepTests(unittest.TestCase):
    def test_provider_action_and_target_shape_is_normalized(self):
        step = ActionStep.model_validate(
            {"type": "action", "action": "Tap", "target": "the Login button"}
        )

        self.assertEqual(step.instruction, "Tap the Login button")

    def test_nested_provider_instruction_is_normalized(self):
        step = ActionStep.model_validate(
            {"kind": "action", "instruction": {"action": "Launch the app."}}
        )

        self.assertEqual(step.instruction, "Launch the app.")

    def test_a_plain_action_is_accepted(self):
        self.assertEqual(
            ActionStep(instruction="Tap the Login button.").rendered(),
            "Tap the Login button.",
        )

    def test_a_statement_about_the_screen_is_refused(self):
        with self.assertRaisesRegex(ValidationError, "not an action"):
            ActionStep(instruction="The error message is visible.")

    def test_state_prose_is_refused(self):
        """App data is cleared first, so state the run needs is reached by actions."""
        with self.assertRaisesRegex(ValidationError, "not an action"):
            ActionStep(instruction="The cart already holds one item.")

    def test_a_claim_after_the_action_is_refused(self):
        with self.assertRaisesRegex(ValidationError, "one action"):
            ActionStep(instruction="Tap Login and verify the error appears.")

    def test_a_verb_for_something_yewo_cannot_do_is_refused(self):
        """No capability shakes the device, so no step may ask for it."""
        with self.assertRaisesRegex(ValidationError, "not an action"):
            ActionStep(instruction="Shake the device.")

    def test_the_refusal_names_the_actions_that_are_allowed(self):
        with self.assertRaises(ValidationError) as refused:
            ActionStep(instruction="Observe the error message.")

        for verb in ("tap", "type", "swipe"):
            self.assertIn(verb, str(refused.exception))


class RenderingTests(unittest.TestCase):
    def test_every_step_renders_in_order(self):
        walk = [
            ActionStep(instruction="Launch the app."),
            EnterStep(field="Email", value="email"),
            CapabilityStep(capability=NetworkCapability(mode="offline")),
        ]

        rendered = steps.rendered(walk)

        self.assertEqual(len(rendered), 3)
        self.assertTrue(rendered[0].startswith("Launch"))
        self.assertIn("{{email}}", rendered[1])
        self.assertIn("offline", rendered[2])


class DeclaredNameTests(unittest.TestCase):
    def test_a_name_the_file_never_declared_is_found(self):
        walk = [EnterStep(field="Email", value="made_up")]

        self.assertEqual(steps.undeclared(walk, frozenset({"email"})), "made_up")

    def test_a_declared_name_is_not_reported(self):
        walk = [EnterStep(field="Email", value="email")]

        self.assertIsNone(steps.undeclared(walk, frozenset({"email"})))

    def test_a_walk_entering_nothing_reports_nothing(self):
        walk = [ActionStep(instruction="Tap the Login button.")]

        self.assertIsNone(steps.undeclared(walk, frozenset()))


class OutcomeTests(unittest.TestCase):
    def test_the_outcome_names_the_field_it_is_about(self):
        """One screenshot has to be able to settle it."""
        self.assertIn("Card Number", steps.refusal_outcome("Card Number"))


if __name__ == "__main__":
    unittest.main()
