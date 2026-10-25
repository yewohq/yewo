"""The coverage contract: which cells a run covers, and in what order.

A cell is one pair of axes at one priority, and its key has to name the same
test across two runs. These tests hold the axes apart, hold the declared
order, keep two declarations of one pair from sharing a key, and hold the
line that a file cannot ask for no coverage at all.
"""

import tempfile
import unittest
from pathlib import Path

from yewo.config import ConfigError
from yewo.testing.suite import load_workflows
from yewo.testing.taxonomy import (
    Category,
    CoverageCell,
    Priority,
    ScenarioType,
    audits_every_screen,
    sort_cells_by_priority,
)


def _file(body: str) -> Path:
    path = Path(tempfile.mkdtemp()) / "yewo.yaml"
    path.write_text(body)
    return path


JOURNEY = """
journeys:
  - id: j1
    journey: A user types {{email}} in the Email field, then taps Login.
    env:
      email: qa.tester@example.com
"""


class CellKeyTests(unittest.TestCase):
    def test_a_cell_names_itself_the_same_way_every_run(self):
        cell = CoverageCell(Category.SECURITY, ScenarioType.NEGATIVE, Priority.P0)

        self.assertEqual(cell.key, "security.negative.0")
        self.assertEqual(cell.test_id, "TS_security_negative_0")

    def test_two_types_in_one_category_are_different_cells(self):
        """A category alone cannot say what shape a test takes."""
        negative = CoverageCell(Category.SECURITY, ScenarioType.NEGATIVE, Priority.P0)
        role = CoverageCell(Category.SECURITY, ScenarioType.ROLE, Priority.P0)

        self.assertNotEqual(negative.key, role.key)


class OrderTests(unittest.TestCase):
    def test_priority_decides_what_a_dying_run_got_through(self):
        late = CoverageCell(Category.INPUT, ScenarioType.BOUNDARY, Priority.P2)
        early = CoverageCell(Category.FUNCTIONAL, ScenarioType.CONCURRENCY, Priority.P0)

        self.assertEqual([c.key for c in sort_cells_by_priority([late, early])], [early.key, late.key])

    def test_the_declared_order_holds_inside_one_priority(self):
        first = CoverageCell(Category.FUNCTIONAL, ScenarioType.HAPPY_PATH, Priority.P1)
        second = CoverageCell(Category.INPUT, ScenarioType.EMPTY, Priority.P1)

        self.assertEqual(
            [c.key for c in sort_cells_by_priority([first, second])], [first.key, second.key]
        )


class EveryScreenTests(unittest.TestCase):
    def test_a_security_cell_reads_every_screen(self):
        """A masked field has to stay masked on the screen after it."""
        self.assertTrue(
            audits_every_screen(Category.SECURITY, ScenarioType.HAPPY_PATH)
        )

    def test_a_concurrency_cell_reads_every_screen(self):
        """An out-of-order tap acts where no assertion expected it."""
        self.assertTrue(
            audits_every_screen(Category.FUNCTIONAL, ScenarioType.CONCURRENCY)
        )

    def test_a_plain_functional_walk_may_replay(self):
        self.assertFalse(
            audits_every_screen(Category.FUNCTIONAL, ScenarioType.HAPPY_PATH)
        )


class DeclarationTests(unittest.TestCase):
    def test_declared_coverage_becomes_one_cell_per_type(self):
        path = _file(
            "coverage:\n"
            "  - {category: security, types: [negative, role], priority: P0}\n"
            "  - {category: input, types: [boundary]}\n" + JOURNEY
        )

        journey = load_workflows(path)[0].journeys[0]

        self.assertEqual(
            [cell.key for cell in journey.coverage],
            ["security.negative.0", "security.role.0", "input.boundary.0"],
        )
        self.assertEqual(
            [cell.priority for cell in journey.coverage],
            [Priority.P0, Priority.P0, Priority.P1],
        )

    def test_a_file_declaring_nothing_still_gets_a_suite(self):
        journey = load_workflows(_file(JOURNEY))[0].journeys[0]

        self.assertTrue(journey.coverage)
        self.assertIn("functional.happy_path.0", [c.key for c in journey.coverage])

    def test_an_empty_coverage_list_is_refused(self):
        """Zero cells writes zero scenarios, and zero failures reads as a pass."""
        path = _file("coverage: []\n" + JOURNEY)

        with self.assertRaisesRegex(ConfigError, "is empty"):
            load_workflows(path)

    def test_an_unknown_category_names_what_is_allowed(self):
        path = _file("coverage:\n  - {category: usability, types: [negative]}\n" + JOURNEY)

        with self.assertRaisesRegex(ConfigError, "security"):
            load_workflows(path)

    def test_one_pair_declared_twice_gets_two_keys(self):
        """Two cells sharing a key would share a test id, so no report could
        tell them apart and no later run could match either one."""
        path = _file(
            "coverage:\n"
            "  - {category: security, types: [negative, negative]}\n"
            "  - {category: security, types: [negative]}\n" + JOURNEY
        )

        journey = load_workflows(path)[0].journeys[0]

        keys = [cell.key for cell in journey.coverage]
        self.assertEqual(
            keys, ["security.negative.0", "security.negative.1", "security.negative.2"]
        )
        self.assertEqual(len(set(cell.test_id for cell in journey.coverage)), 3)

    def test_a_workflow_replaces_the_file_coverage_rather_than_adding_to_it(self):
        path = _file(
            "coverage:\n  - {category: security, types: [negative]}\n"
            "workflows:\n"
            "  checkout:\n"
            "    coverage:\n      - {category: network, types: [recovery]}\n"
            "    journeys:\n"
            "      - {id: j1, journey: A user opens the app.}\n"
        )

        journey = load_workflows(path)[0].journeys[0]

        self.assertEqual([c.key for c in journey.coverage], ["network.recovery.0"])


class JourneyValueTests(unittest.TestCase):
    def test_a_journey_keeps_the_names_it_declared(self):
        """The generator asks for a value by name rather than copying prose."""
        journey = load_workflows(_file(JOURNEY))[0].journeys[0]

        self.assertEqual(journey.values, {"email": "qa.tester@example.com"})

    def test_a_declared_refusal_is_recorded_beside_the_value(self):
        """One namespace, and the entry says what the app does with it."""
        path = _file(
            "journeys:\n"
            "  - id: j1\n"
            "    journey: A user signs in.\n"
            "    env:\n"
            "      email: qa.tester@example.com\n"
            "      rejected_password: {value: abc, refused: true}\n"
        )

        journey = load_workflows(path)[0].journeys[0]

        self.assertEqual(journey.values["rejected_password"], "abc")
        self.assertEqual(journey.refused, frozenset({"rejected_password"}))

    def test_a_value_saying_nothing_is_meant_to_work(self):
        journey = load_workflows(_file(JOURNEY))[0].journeys[0]

        self.assertEqual(journey.refused, frozenset())

    def test_an_unknown_key_on_a_declaration_is_named(self):
        path = _file(
            "journeys:\n"
            "  - id: j1\n"
            "    journey: A user signs in.\n"
            "    env:\n"
            "      password: {value: abc, rejected: true}\n"
        )

        with self.assertRaisesRegex(ConfigError, "rejected"):
            load_workflows(path)

    def test_a_refused_value_can_still_be_secret(self):
        path = _file(
            "journeys:\n"
            "  - id: j1\n"
            "    journey: A user signs in.\n"
            "    env:\n"
            "      rejected_password:\n"
            "        value: other-secret\n"
            "        secret: true\n"
            "        refused: true\n"
        )

        journey = load_workflows(path)[0].journeys[0]

        self.assertEqual(journey.secrets["rejected_password"], "other-secret")
        self.assertIn("rejected_password", journey.refused)

    def test_another_actor_is_declared_beside_env(self):
        """An actor groups its context values outside the shared environment."""
        path = _file(
            "actors:\n"
            "  reviewer:\n"
            "    workspace: draft-library\n"
            "    access_code: {value: reviewer-secret, secret: true}\n" + JOURNEY
        )

        journey = load_workflows(path)[0].journeys[0]

        self.assertEqual(
            journey.actors["reviewer"].values,
            {"workspace": "draft-library", "access_code": "reviewer-secret"},
        )

    def test_an_actor_holds_whatever_establishes_its_context(self):
        path = _file(
            "actors:\n"
            "  reviewer:\n"
            "    workspace: draft-library\n"
            "    access_code: {value: '4821', secret: true}\n" + JOURNEY
        )

        journey = load_workflows(path)[0].journeys[0]

        self.assertEqual(
            sorted(journey.actors["reviewer"].values),
            ["access_code", "workspace"],
        )

    def test_an_actor_value_joins_the_one_namespace_under_a_dotted_name(self):
        """A step names it like any other value, so nothing tracks where it came from."""
        path = _file(
            "actors:\n  reviewer:\n    workspace: draft-library\n" + JOURNEY
        )

        journey = load_workflows(path)[0].journeys[0]

        self.assertEqual(
            journey.values["actors.reviewer.workspace"],
            "draft-library",
        )

    def test_an_actor_secret_is_masked_under_the_path_the_file_wrote(self):
        """A name Yewo invented reads back in a report nobody can trace."""
        path = _file(
            "actors:\n"
            "  editor:\n"
            "    username: editor@example.com\n"
            "    password: {value: editor-secret, secret: true}\n" + JOURNEY
        )

        journey = load_workflows(path)[0].journeys[0]

        self.assertEqual(journey.secrets["actors.editor.password"], "editor-secret")

    def test_an_actor_with_no_values_is_refused(self):
        path = _file("actors:\n  reviewer: {}\n" + JOURNEY)

        with self.assertRaisesRegex(ConfigError, "actors.reviewer"):
            load_workflows(path)

    def test_a_workflow_adds_its_own_actors_to_the_file(self):
        path = _file(
            "actors:\n  reader:\n    workspace: published-library\n"
            "workflows:\n"
            "  documents:\n"
            "    actors:\n      editor:\n        workspace: draft-library\n"
            "    journeys:\n"
            "      - {id: j1, journey: A user opens the app.}\n"
        )

        journey = load_workflows(path)[0].journeys[0]

        self.assertEqual(sorted(journey.actors), ["editor", "reader"])


if __name__ == "__main__":
    unittest.main()
