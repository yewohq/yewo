"""How a run resolves locked scenarios and scenarios it must generate.

`test_lock_file.py` holds the file and contract hash. These hold the policy:
what is reused, what is regenerated, what is refused, and what a run does
when the replacement for a stale cell never arrives.
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
from yewo.testing.lock_file import contract_sha256
from yewo.testing.lock_resolution import staleness, suite_for
from yewo.testing.lifecycle import ScenarioPreconditions
from yewo.testing.models import ScenarioDraft, UserJourney
from yewo.agent.actuation.actions import CapabilityKind, PermissionCapability
from yewo.testing.steps import EnterStep
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


class ResolutionTests(unittest.TestCase):
    def _suite(self, path, responses=(), **kwargs):
        made, client = generator(*responses)
        resolution = suite_for(
            journey(*kwargs.pop("cells", [])) if "cells" in kwargs else journey(),
            lambda j, cells: made.generate_suite(j, cells),
            path=path,
            **kwargs,
        )
        return resolution, client

    def test_a_first_run_generates_and_writes_the_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "yewo.lock"
            resolution, client = self._suite(path, [draft()])

            self.assertEqual(resolution.generated_count, 1)
            self.assertEqual(resolution.reused_count, 0)
            self.assertTrue(resolution.lock_written)
            self.assertTrue(path.is_file())
            self.assertEqual(client.generate_response.call_count, 1)

    def test_a_second_run_reads_it_and_asks_for_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "yewo.lock"
            first, _ = self._suite(path, [draft()])
            second, client = self._suite(path)

            self.assertEqual(second.reused_count, 1)
            self.assertEqual(second.generated_count, 0)
            self.assertFalse(second.lock_written)
            self.assertEqual(client.generate_response.call_count, 0)
            self.assertEqual(
                second.scenarios[0].scenario_uuid,
                first.scenarios[0].scenario_uuid,
            )

    def test_a_locked_scenario_receives_current_journey_preconditions(self):
        declared = journey(
            preconditions=ScenarioPreconditions(
                capabilities=[PermissionCapability(permission="camera", grant=False)]
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "yewo.lock"
            made, _ = generator(draft())
            suite_for(declared, made.generate_suite, path=path)
            unused, client = generator()

            resolution = suite_for(declared, unused.generate_suite, path=path)

        self.assertEqual(client.generate_response.call_count, 0)
        self.assertEqual(
            resolution.scenarios[0].initial_capabilities[0].kind,
            CapabilityKind.PERMISSION,
        )

    def test_a_changed_journey_regenerates_only_that_cell(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "yewo.lock"
            made, _ = generator(draft("A"), draft("B"))
            two = journey(HAPPY, NEGATIVE)
            suite_for(two, lambda j, cells: made.generate_suite(j, cells), path=path)

            changed = journey(HAPPY, NEGATIVE)
            changed.coverage = [
                HAPPY,
                CoverageCell(Category.SECURITY, ScenarioType.NEGATIVE, Priority.P1),
            ]
            again, client = generator(draft("C"))
            resolution = suite_for(
                changed, lambda j, cells: again.generate_suite(j, cells), path=path
            )

            self.assertEqual(resolution.reused_count, 1)
            self.assertEqual(resolution.generated_count, 1)
            self.assertEqual(client.generate_response.call_count, 1)

    def test_values_are_substituted_whether_or_not_the_lock_answered(self):
        """The lock holds names. What runs holds values. One filling, two paths."""
        steps = [EnterStep(field="Email", value="email")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "yewo.lock"
            made, _ = generator(draft(steps=steps))
            live = UserJourney(
                id="j",
                name="J",
                description="A user signs in.",
                coverage=[HAPPY],
                values={"email": "qa@example.com"},
            )
            first = suite_for(live, lambda j, c: made.generate_suite(j, c), path=path)
            again, _ = generator()
            second = suite_for(live, lambda j, c: again.generate_suite(j, c), path=path)

        for resolution in (first, second):
            self.assertEqual(
                resolution.scenarios[0].steps_instructions,
                ["Enter 'qa@example.com' in the Email field."],
            )

    def test_entries_for_another_journey_survive_a_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "yewo.lock"
            other = UserJourney(
                id="other", name="O", description="Something else.", coverage=[HAPPY]
            )
            made, _ = generator(draft())
            suite_for(other, lambda j, c: made.generate_suite(j, c), path=path)

            mine, _ = generator(draft())
            suite_for(journey(), lambda j, c: mine.generate_suite(j, c), path=path)

            held = lock_file.read(path)

        self.assertEqual(sorted(e.journey for e in held.entries), ["j", "other"])

    def test_a_cell_the_file_stopped_declaring_is_dropped(self):
        """A lock is whole or it is a bug, and an entry nothing asks for is neither."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "yewo.lock"
            made, _ = generator(draft("A"), draft("B"))
            suite_for(
                journey(HAPPY, NEGATIVE),
                lambda j, c: made.generate_suite(j, c),
                path=path,
            )

            fewer, _ = generator(draft("C"))
            journey_now = journey(NEGATIVE)
            journey_now.description = "A user signs in."
            suite_for(journey_now, lambda j, c: fewer.generate_suite(j, c), path=path)

            held = lock_file.read(path)

        self.assertEqual([entry.cell for entry in held.entries], ["security.negative.0"])


class LockedRunTests(unittest.TestCase):
    """`--locked` never generates, so CI cannot run a silently different test."""

    def _refused(self, **kwargs):
        made, client = generator(draft())
        with self.assertRaises(UsageError) as raised:
            suite_for(
                journey(),
                lambda j, cells: made.generate_suite(j, cells),
                locked=True,
                **kwargs,
            )
        self.assertEqual(client.generate_response.call_count, 0)
        return str(raised.exception)

    def test_an_absent_lock_fails_and_names_the_command(self):
        with tempfile.TemporaryDirectory() as directory:
            message = self._refused(path=Path(directory) / "yewo.lock")

        self.assertIn("yewo lock", message)

    def test_a_stale_lock_names_the_cell_and_why(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "yewo.lock"
            made, _ = generator(draft())
            suite_for(journey(), lambda j, c: made.generate_suite(j, c), path=path)

            changed = journey()
            changed.description = "A user signs in with a one-time code."
            made2, client = generator(draft())
            with self.assertRaises(UsageError) as raised:
                suite_for(
                    changed,
                    lambda j, c: made2.generate_suite(j, c),
                    path=path,
                    locked=True,
                )

        message = str(raised.exception)
        self.assertIn("functional.happy_path.0", message)
        self.assertIn("changed", message)
        self.assertEqual(client.generate_response.call_count, 0)

    def test_a_fresh_lock_runs_without_asking_for_anything(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "yewo.lock"
            made, _ = generator(draft())
            suite_for(journey(), lambda j, c: made.generate_suite(j, c), path=path)

            second, client = generator()
            resolution = suite_for(
                journey(),
                lambda j, c: second.generate_suite(j, c),
                path=path,
                locked=True,
            )

        self.assertEqual(resolution.reused_count, 1)
        self.assertFalse(resolution.lock_written)
        self.assertEqual(client.generate_response.call_count, 0)

    def test_staleness_names_the_two_problems_apart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "yewo.lock"
            made, _ = generator(draft())
            suite_for(journey(), lambda j, c: made.generate_suite(j, c), path=path)
            held = lock_file.read(path)

        self.assertEqual(staleness(held, journey(), ""), {})
        self.assertEqual(
            staleness(held, journey(HAPPY, NEGATIVE), ""),
            {"security.negative.0": "not in the lock"},
        )
        self.assertEqual(
            staleness(None, journey(), ""), {"functional.happy_path.0": "not in the lock"}
        )


class StableFileTests(unittest.TestCase):
    """The file's own order, so a run never commits a diff that changed nothing."""

    def _lock_one(self, path, journey_id, responses=None):
        one = UserJourney(
            id=journey_id,
            name=journey_id,
            description="A user signs in.",
            coverage=[HAPPY],
        )
        made, client = generator(*(responses if responses is not None else [draft()]))
        return suite_for(one, made.generate_suite, path=path), client

    def _two_journeys(self, path, order):
        for journey_id in order:
            self._lock_one(path, journey_id)
        return path.read_text()

    def test_the_order_does_not_depend_on_which_journey_ran_first(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self._two_journeys(Path(directory) / "a.lock", ["a", "b"])
            second = self._two_journeys(Path(directory) / "b.lock", ["b", "a"])

        self.assertEqual(
            [entry["journey"] for entry in json.loads(first)["entries"]],
            [entry["journey"] for entry in json.loads(second)["entries"]],
        )

    def test_a_second_run_of_two_journeys_rewrites_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "yewo.lock"
            self._two_journeys(path, ["a", "b"])
            before = path.read_text()

            for journey_id in ("a", "b"):
                resolution, client = self._lock_one(path, journey_id, responses=())
                self.assertFalse(resolution.lock_written, journey_id)
                self.assertEqual(client.generate_response.call_count, 0)

            self.assertEqual(before, path.read_text())


class RefusalCostTests(unittest.TestCase):
    """`--locked` with no lock has to cost nothing.

    CI runs this. A stale lock there must fail in a second, rather than after
    an emulator boots and an app installs.
    """

    def _locked_run(self, directory):
        from unittest.mock import patch

        from yewo.testing.models import UserJourney
        from yewo.testing.runner import YewoTestRunner

        path = Path(directory) / "yewo.yaml"
        path.write_text("device: Pixel_8\n")
        runner = YewoTestRunner(
            device_name="Pixel_8",
            app_path="app.apk",
            assets_dir=Path(directory) / "assets",
            journey_file=path,
            locked=True,
        )
        with (
            patch("yewo.testing.runner.ensure_device_ready") as ready,
            patch("yewo.testing.runner.DeviceSession") as session,
            self.assertRaises(UsageError) as raised,
        ):
            runner.run_journey(UserJourney(id="j", name="J", description="A user signs in."))
        return str(raised.exception), ready, session, Path(directory)

    def test_no_device_is_asked_for(self):
        with tempfile.TemporaryDirectory() as directory:
            message, ready, session, _ = self._locked_run(directory)

        self.assertIn("yewo lock", message)
        self.assertEqual(ready.call_count, 0)
        self.assertEqual(session.call_count, 0)

    def test_nothing_is_written(self):
        """A run that never started leaves no artifact tree behind."""
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, root = self._locked_run(directory)

            self.assertFalse((root / "assets").exists())


class StaleAndUnregenerableTests(unittest.TestCase):
    """A stale cell whose replacement never arrived must not run the old test.

    This is the whole guarantee. A lock that silently falls back to the
    scenario the file no longer describes is worse than no lock: the run
    reports a reused entry, and nothing says the test was not the test.
    """

    def _changed(self, path, failing):
        """Locks a journey, changes it, then fails to regenerate the cell."""
        made, _ = generator(draft("Original"))
        suite_for(journey(), made.generate_suite, path=path)

        altered = journey()
        altered.description = "A user signs in with a one-time code."
        # Three, because a schema mismatch is asked again before it is given up on.
        broken, client = generator(*([failing] * 3))
        return suite_for(altered, broken.generate_suite, path=path), client

    def test_the_old_scenario_does_not_run(self):
        from pydantic import ValidationError

        mismatch = ValidationError.from_exception_data("ScenarioDraft", [])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "yewo.lock"
            resolution, _ = self._changed(path, mismatch)

        self.assertEqual(resolution.scenarios, [])
        self.assertEqual(resolution.reused_count, 0)
        self.assertEqual(resolution.generated_count, 0)

    def test_the_cell_is_reported_as_uncovered_with_its_cause(self):
        from pydantic import ValidationError

        from yewo.errors import Cause

        mismatch = ValidationError.from_exception_data("ScenarioDraft", [])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "yewo.lock"
            resolution, _ = self._changed(path, mismatch)

        self.assertEqual(len(resolution.uncovered), 1)
        self.assertEqual(resolution.uncovered[0].cell.key, "functional.happy_path.0")
        self.assertIs(resolution.uncovered[0].cause, Cause.PROVIDER)

    def test_the_lock_is_not_rewritten_by_a_run_that_produced_nothing(self):
        """A provider outage must not delete a reviewed test.

        The entry stays, and staleness is recomputed every run, so it is
        refused again rather than quietly reused.
        """
        from pydantic import ValidationError

        mismatch = ValidationError.from_exception_data("ScenarioDraft", [])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "yewo.lock"
            before = path.read_text() if path.exists() else None
            resolution, _ = self._changed(path, mismatch)
            after = path.read_text()
            held = lock_file.read(path)

        self.assertFalse(resolution.lock_written)
        self.assertEqual(len(held.entries), 1)
        self.assertEqual(held.entries[0].scenario_name, "Original")
        self.assertNotEqual(before, after)

    def test_the_entry_it_kept_is_still_refused_on_the_next_run(self):
        from pydantic import ValidationError

        mismatch = ValidationError.from_exception_data("ScenarioDraft", [])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "yewo.lock"
            self._changed(path, mismatch)
            again, _ = self._changed(path, mismatch)

        self.assertEqual(again.scenarios, [])
        self.assertEqual(again.reused_count, 0)

    def test_a_provider_that_would_not_answer_is_the_same_refusal(self):
        from yewo.errors import ProviderUnavailable

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "yewo.lock"
            resolution, _ = self._changed(path, ProviderUnavailable("spent"))

        self.assertEqual(resolution.scenarios, [])
        self.assertEqual(len(resolution.uncovered), 1)

    def test_a_fresh_cell_beside_a_stale_one_still_runs(self):
        """One cell nobody could regenerate is not a reason to test nothing."""
        from pydantic import ValidationError

        mismatch = ValidationError.from_exception_data("ScenarioDraft", [])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "yewo.lock"
            made, _ = generator(draft("A"), draft("B"))
            two = journey(HAPPY, NEGATIVE)
            suite_for(two, made.generate_suite, path=path)

            # Only the security cell's question changes.
            altered = journey(
                HAPPY, CoverageCell(Category.SECURITY, ScenarioType.NEGATIVE, Priority.P1)
            )
            broken, _ = generator(mismatch)
            resolution = suite_for(altered, broken.generate_suite, path=path)

        self.assertEqual(resolution.reused_count, 1)
        self.assertEqual(resolution.generated_count, 0)
        self.assertEqual([s.cell for s in resolution.scenarios], ["functional.happy_path.0"])


class PartialRegenerationTests(unittest.TestCase):
    """One cell regenerates, another cannot, and the lock holds both answers.

    The all-or-nothing case is protected by the guard in `_persist`, which
    declines to write when a run produced nothing. A run that produced
    *something* writes, and a stale entry left out of that write is deleted.
    Deleting a reviewed test because a provider was down for one call is the
    failure this covers.
    """

    def _half_failed(self, path):
        from pydantic import ValidationError

        made, _ = generator(draft("A"), draft("B"))
        suite_for(journey(HAPPY, NEGATIVE), made.generate_suite, path=path)

        # Both questions change. One is answered, one is not.
        altered = journey(
            CoverageCell(Category.FUNCTIONAL, ScenarioType.HAPPY_PATH, Priority.P1),
            CoverageCell(Category.SECURITY, ScenarioType.NEGATIVE, Priority.P1),
        )
        mismatch = ValidationError.from_exception_data("ScenarioDraft", [])
        broken, _ = generator(draft("C"), *([mismatch] * 3))
        return suite_for(altered, broken.generate_suite, path=path), altered

    def test_the_cell_that_could_not_be_written_keeps_its_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "yewo.lock"
            self._half_failed(path)
            held = lock_file.read(path)

        self.assertEqual(
            sorted(entry.cell for entry in held.entries),
            ["functional.happy_path.0", "security.negative.0"],
        )

    def test_it_keeps_the_contract_it_was_written_under(self):
        """Which is what makes the next run read it as stale again."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "yewo.lock"
            _, altered = self._half_failed(path)
            held = lock_file.read(path)
            stale = next(e for e in held.entries if e.cell == "security.negative.0")
            now = staleness(held, altered, "")

        self.assertIn("security.negative.0", now)
        self.assertNotEqual(
            stale.contract_sha256,
            contract_sha256(altered, altered.coverage[1]),
        )

    def test_the_cell_that_was_written_is_fresh(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "yewo.lock"
            _, altered = self._half_failed(path)
            held = lock_file.read(path)

            self.assertNotIn("functional.happy_path.0", staleness(held, altered, ""))

    def test_only_the_cell_that_was_written_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "yewo.lock"
            resolution, _ = self._half_failed(path)

        self.assertEqual([s.cell for s in resolution.scenarios], ["functional.happy_path.0"])
        self.assertEqual(resolution.generated_count, 1)
        self.assertEqual(resolution.reused_count, 0)

    def test_the_stale_cell_is_still_refused_on_the_next_run(self):
        from pydantic import ValidationError

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "yewo.lock"
            _, altered = self._half_failed(path)

            mismatch = ValidationError.from_exception_data("ScenarioDraft", [])
            again, _ = generator(*([mismatch] * 3))
            second = suite_for(altered, again.generate_suite, path=path)

        self.assertEqual([s.cell for s in second.scenarios], ["functional.happy_path.0"])
        self.assertEqual(second.reused_count, 1)
