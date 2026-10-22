"""The `yewo.lock` file: what it holds, refuses, and considers stale.

A cell key names a question. The lock names the test that answered it. These
cases hold the two apart, because every comparison between two runs depends on
it.
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from fakes import walk

from yewo.errors import UsageError
from yewo.testing import lock_file
from yewo.testing.generator import ScenarioGenerator
from yewo.testing.lock_file import Lock, LockEntry, contract_sha256
from yewo.testing.models import ScenarioDraft, UserJourney
from yewo.testing.steps import ActionStep, EnterStep
from yewo.testing.taxonomy import Category, CoverageCell, Priority, ScenarioType

HAPPY = CoverageCell(Category.FUNCTIONAL, ScenarioType.HAPPY_PATH, Priority.P0)
NEGATIVE = CoverageCell(Category.SECURITY, ScenarioType.NEGATIVE, Priority.P0)


def journey(*cells, **extra):
    return UserJourney(
        id="j",
        name="J",
        description="A user signs in.",
        coverage=list(cells) or [HAPPY],
        **extra,
    )


def draft(name="Sign in", steps=None, outcomes=("the app opens",)):
    return ScenarioDraft(
        scenario_name=name,
        steps=steps or walk("Launch the app."),
        expected_outcomes=list(outcomes),
    )


def generator(*responses):
    client = SimpleNamespace(generate_response=Mock(side_effect=list(responses)))
    return ScenarioGenerator(client), client


class ContractTests(unittest.TestCase):
    """What makes a locked scenario stale, and what must never make it so."""

    def test_the_same_question_hashes_the_same_way(self):
        self.assertEqual(
            contract_sha256(journey(), HAPPY), contract_sha256(journey(), HAPPY)
        )

    def test_two_cells_of_one_journey_are_different_contracts(self):
        self.assertNotEqual(
            contract_sha256(journey(HAPPY, NEGATIVE), HAPPY),
            contract_sha256(journey(HAPPY, NEGATIVE), NEGATIVE),
        )

    def test_rewritten_prose_is_a_new_question(self):
        """The scenario was written from the description, so the test is stale."""
        other = journey()
        other.description = "A user signs in with a one-time code."

        self.assertNotEqual(contract_sha256(journey(), HAPPY), contract_sha256(other, HAPPY))

    def test_a_new_security_rule_is_a_new_question(self):
        self.assertNotEqual(
            contract_sha256(journey(), HAPPY),
            contract_sha256(journey(security_rules=["no card number on screen"]), HAPPY),
        )

    def test_a_renamed_value_is_a_new_question(self):
        self.assertNotEqual(
            contract_sha256(journey(values={"email": "a@b.c"}), HAPPY),
            contract_sha256(journey(values={"username": "a@b.c"}), HAPPY),
        )

    def test_a_changed_value_is_the_same_question(self):
        """The test asks for the name. A rotated password is not a new test.

        This is also what keeps a credential out of the hash. A lock is
        committed, and a hash of a six-character password is recovered
        offline in seconds.
        """
        self.assertEqual(
            contract_sha256(journey(values={"password": "abc123"}), HAPPY),
            contract_sha256(journey(values={"password": "zzz999"}), HAPPY),
        )


class EntryTests(unittest.TestCase):
    """What one entry carries, and what it rebuilds."""

    def _entry(self, **extra):
        from yewo.testing.models import GeneratedTestCase

        scenario = GeneratedTestCase.of(NEGATIVE, draft(**extra))
        return scenario, LockEntry.of(scenario, journey(), "flow", "digest")

    def test_the_entry_keeps_the_identity_the_scenario_was_drawn_with(self):
        scenario, entry = self._entry()

        self.assertEqual(entry.scenario_uuid, scenario.scenario_uuid)
        self.assertEqual(entry.cell, "security.negative.0")
        self.assertEqual(entry.journey, "j")
        self.assertEqual(entry.workflow, "flow")

    def test_a_locked_scenario_comes_back_with_the_same_identity(self):
        """Two runs of one test have to agree on which test it was."""
        scenario, entry = self._entry()
        restored = entry.scenario()

        self.assertEqual(restored.scenario_uuid, scenario.scenario_uuid)
        self.assertEqual(restored.cell, scenario.cell)
        self.assertEqual(restored.test_id, scenario.test_id)
        self.assertEqual(restored.priority, scenario.priority)

    def test_nothing_derived_is_stored(self):
        """A stored copy of a derived field can disagree with the derivation."""
        _, entry = self._entry()
        stored = set(entry.model_dump())

        self.assertNotIn("steps_instructions", stored)
        self.assertNotIn("required_capabilities", stored)

    def test_the_derived_fields_come_back_anyway(self):
        _, entry = self._entry(
            steps=[
                ActionStep(instruction="Launch the app."),
                EnterStep(field="Email", value="email"),
            ]
        )

        self.assertEqual(
            entry.scenario().steps_instructions,
            ["Launch the app.", "Enter '{{email}}' in the Email field."],
        )

    def test_the_entry_holds_the_name_and_never_the_value(self):
        """A lock is committed. A value in it is a credential in the repository."""
        _, entry = self._entry(
            steps=[EnterStep(field="Password", value="rejected_password")]
        )

        written = json.dumps(entry.model_dump(mode="json"))
        self.assertIn("rejected_password", written)
        self.assertNotIn("abc", written)


class FileTests(unittest.TestCase):
    def test_a_lock_round_trips_through_its_file(self):
        from yewo.testing.models import GeneratedTestCase

        scenario = GeneratedTestCase.of(HAPPY, draft())
        original = Lock(entries=[LockEntry.of(scenario, journey(), "", "d")])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "yewo.lock"
            lock_file.write(path, original)
            loaded_lock = lock_file.read(path)

        self.assertEqual(loaded_lock.entries[0].scenario_uuid, scenario.scenario_uuid)
        self.assertEqual(loaded_lock.sha256(), original.sha256())

    def test_an_absent_lock_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(lock_file.read(Path(directory) / "yewo.lock"))

    def test_an_unreadable_lock_is_never_read_as_an_absent_one(self):
        """Reading it as absent would silently regenerate the tests it held."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "yewo.lock"
            path.write_text("{not json")

            with self.assertRaises(UsageError):
                lock_file.read(path)

    def test_a_lock_from_another_schema_is_refused_by_name(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "yewo.lock"
            path.write_text(json.dumps({"schema_version": 99, "entries": []}))

            with self.assertRaises(UsageError) as raised:
                lock_file.read(path)

        self.assertIn("yewo lock", str(raised.exception))

    def test_nothing_is_left_behind_when_a_write_succeeds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "yewo.lock"
            lock_file.write(path, Lock())

            self.assertEqual([p.name for p in Path(directory).iterdir()], ["yewo.lock"])
