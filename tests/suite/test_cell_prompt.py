"""What one cell's generator call is told, and what it is not told.

The call carries the cell, the names the file declared, and the rules only
where the cell is about them. It carries no resolved value, no example from
one app, and no invitation to invent a value.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from yewo.agent.actuation.actions import CapabilityKind
from yewo.testing.generator import CAPABILITY_BY_CATEGORY, ScenarioGenerator, _requirements
from yewo.testing.models import Actor, UserJourney
from yewo.testing.step_language import ACTION_VERBS
from yewo.testing.taxonomy import (
    Category,
    CoverageCell,
    Priority,
    ScenarioType,
    describe,
)

PASSWORD = "secret-password123"
RULE = "The password must never appear in plain text."


def _journey(**extra) -> UserJourney:
    """One field that works, one the file says this app refuses."""
    extra.setdefault(
        "values", {"email": "qa.tester@example.com", "rejected_password": "abc"}
    )
    extra.setdefault("refused", frozenset({"rejected_password"}))
    return UserJourney(
        id="j",
        name="Sign in",
        description=(
            f"A user enters qa.tester@example.com in the Email field and "
            f"{PASSWORD} in the Password field, then taps Login."
        ),
        security_rules=[RULE],
        secrets={"password": PASSWORD},
        **extra,
    )


def _prompt(cell: CoverageCell, journey=None) -> str:
    return ScenarioGenerator(None)._prompt(journey or _journey(), cell)


NEGATIVE = CoverageCell(Category.INPUT, ScenarioType.NEGATIVE, Priority.P1)
HAPPY = CoverageCell(Category.FUNCTIONAL, ScenarioType.HAPPY_PATH, Priority.P0)
SECURITY = CoverageCell(Category.SECURITY, ScenarioType.NEGATIVE, Priority.P0)
NETWORK = CoverageCell(Category.NETWORK, ScenarioType.RECOVERY, Priority.P1)


class CellIdentityTests(unittest.TestCase):
    def test_both_axes_reach_the_model(self):
        prompt = _prompt(SECURITY)

        self.assertIn("Category: security", prompt)
        self.assertIn("Scenario type: negative", prompt)

    def test_each_axis_carries_its_own_description(self):
        """Not the shared one. A member's docstring reads back as the class's."""
        prompt = _prompt(NETWORK)

        self.assertIn(describe(Category.NETWORK), prompt)
        self.assertIn(describe(ScenarioType.RECOVERY), prompt)
        self.assertNotIn(Category.__doc__.strip(), prompt)

    def test_every_axis_value_is_described(self):
        for category in Category:
            self.assertTrue(describe(category), category)
        for kind in ScenarioType:
            self.assertTrue(describe(kind), kind)

    def test_reusable_policy_is_separate_from_cell_context(self):
        llm = SimpleNamespace(generate_response=Mock(return_value=object()))
        generator = ScenarioGenerator(llm)

        generator._draft(_journey(), HAPPY)

        messages = llm.generate_response.call_args.kwargs["messages"]
        self.assertEqual(messages[0].role, "system")
        self.assertNotIn("qa.tester@example.com", messages[0].content)
        self.assertEqual(messages[1].role, "user")
        self.assertIn("$PASSWORD", messages[1].content)


class ValueTests(unittest.TestCase):
    def test_every_declared_name_is_offered(self):
        prompt = _prompt(NEGATIVE)

        self.assertIn("- email —", prompt)
        self.assertIn("- rejected_password —", prompt)

    def test_the_prompt_says_which_values_this_app_refuses(self):
        prompt = _prompt(NEGATIVE)

        offered = {
            line.split("—")[0].strip(" -"): line.split("—")[1].strip()
            for line in prompt.splitlines()
            if line.startswith("- ") and "—" in line
        }

        self.assertEqual(offered["rejected_password"], "this app refuses this value")
        self.assertEqual(offered["email"], "a value that is meant to work")

    def test_no_resolved_value_reaches_the_call(self):
        """A name is all a step needs, and a value here would leak a secret."""
        journey = _journey(
            values={"email": "qa@example.com", "rejected_password": "floor5"}
        )

        prompt = _prompt(NEGATIVE, journey)

        self.assertNotIn("floor5", prompt)
        self.assertNotIn("qa@example.com", prompt.split("### The values")[1])

    def test_an_actor_value_is_offered_by_reference_and_never_resolved(self):
        """The model receives an actor value's name and never its contents."""
        journey = _journey(
            values={
                "query": "release notes",
                "actors.reviewer.workspace": "confidential-workspace",
            },
            actors={
                "reviewer": Actor(values={"workspace": "confidential-workspace"})
            },
        )

        prompt = _prompt(SECURITY, journey)

        self.assertIn("actors.reviewer.workspace — declared for actor 'reviewer'", prompt)
        self.assertNotIn("confidential-workspace", prompt)
        self.assertNotIn("account", prompt.lower())

    def test_the_prompt_never_asks_for_a_value_of_the_models_own(self):
        """It used to forbid that and then ask for malformed values anyway."""
        prompt = _prompt(NEGATIVE)

        self.assertIn("Never write a value of your own", prompt)
        for invited in ("malformed", "wrong shape", "checksum"):
            self.assertNotIn(invited, prompt, invited)


class RequirementTests(unittest.TestCase):
    """What an answer is read against, beyond its own shape."""

    def test_every_cell_is_held_to_the_declared_names(self):
        self.assertEqual(
            _requirements(HAPPY, _journey())["declared"],
            frozenset({"email", "rejected_password"}),
        )

    def test_a_negative_cell_is_held_to_a_refused_value(self):
        self.assertEqual(
            _requirements(NEGATIVE, _journey())["refused"],
            frozenset({"rejected_password"}),
        )

    def test_a_negative_cell_with_nothing_refused_is_not_held_to_one(self):
        """Holding it would cost the cell for a file that declared nothing."""
        journey = _journey(values={"email": "q@e.com"}, refused=frozenset())

        self.assertNotIn("refused", _requirements(NEGATIVE, journey))

    def test_a_network_cell_is_held_to_a_network_capability(self):
        self.assertEqual(
            _requirements(NETWORK, _journey())["capability"], CapabilityKind.NETWORK
        )

    def test_a_cell_about_nothing_in_particular_is_held_to_no_capability(self):
        self.assertNotIn("capability", _requirements(HAPPY, _journey()))

    def test_every_mapped_category_names_a_real_capability(self):
        for category, kind in CAPABILITY_BY_CATEGORY.items():
            self.assertIn(category, Category)
            self.assertIn(kind, CapabilityKind)


class SecurityRuleTests(unittest.TestCase):
    def test_security_cells_are_not_limited_to_secrets_or_refused_values(self):
        self.assertEqual(
            describe(Category.SECURITY),
            "whether the app satisfies its declared security rules",
        )

    def test_a_security_cell_is_given_the_rules(self):
        """Reaching the screen where a rule can break is navigation."""
        self.assertIn(RULE, _prompt(SECURITY))

    def test_a_cell_about_anything_else_is_not(self):
        """The auditor reads every rule on every screen regardless."""
        self.assertNotIn(RULE, _prompt(NEGATIVE))
        self.assertNotIn(RULE, _prompt(HAPPY))


class SecretTests(unittest.TestCase):
    def test_a_declared_secret_does_not_reach_this_call(self):
        prompt = _prompt(NEGATIVE)

        self.assertNotIn(PASSWORD, prompt)
        self.assertIn("$PASSWORD", prompt)


class OutcomeScopeTests(unittest.TestCase):
    def test_measured_outcomes_require_an_explicit_threshold(self):
        prompt = _prompt(HAPPY)

        self.assertIn('"measured"', prompt)
        self.assertIn("`app_launch_ttid_ms`", prompt)
        self.assertIn("`min_touch_target`", prompt)
        self.assertIn("Never invent", prompt)


class DerivedVocabularyTests(unittest.TestCase):
    """The prompt offers the verbs the enums define, and no hand-kept list."""

    def test_the_allowed_verbs_are_the_derived_ones(self):
        prompt = _prompt(HAPPY)

        for verb in ACTION_VERBS:
            self.assertIn(verb, prompt, verb)


if __name__ == "__main__":
    unittest.main()
