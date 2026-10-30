"""A cold run proves its empty start, or it refuses to be called cold."""

import json
import tempfile
import unittest
from pathlib import Path

from yewo.errors import UsageError
from yewo.testing.cold_start import ColdStartEvidence, require_clean_room


def _clean_room(directory, **overrides):
    root = Path(directory)
    arguments = {
        "journey_file": root / "yewo.yaml",
        "knowledge_root": root / "cold" / "knowledge",
        "output_root": root / "cold",
        "replay": False,
        "durable_replay": False,
        "reuse_app": False,
        "locked": False,
    }
    arguments.update(overrides)
    positional = (
        arguments.pop("journey_file"),
        arguments.pop("knowledge_root"),
        arguments.pop("output_root"),
    )
    return require_clean_room(*positional, **arguments)


class CleanRoomTests(unittest.TestCase):
    def test_an_empty_start_is_recorded_with_both_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = _clean_room(directory)

            self.assertIsInstance(evidence, ColdStartEvidence)
            self.assertEqual(evidence.prior_artifact_count, 0)
            self.assertTrue(evidence.durable_replay_disabled)
            self.assertEqual(evidence.knowledge_root, str(Path(directory) / "cold" / "knowledge"))
            self.assertEqual(evidence.output_root, str(Path(directory) / "cold"))

    def test_a_reviewed_lock_beside_the_journey_refuses_the_run(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "yewo.lock").write_text(json.dumps({"entries": []}))

            with self.assertRaises(UsageError) as refusal:
                _clean_room(directory)

            self.assertIn("yewo.lock", str(refusal.exception))

    def test_stored_knowledge_refuses_the_run_and_names_what_it_found(self):
        with tempfile.TemporaryDirectory() as directory:
            maps = Path(directory) / "cold" / "knowledge" / "app_maps"
            maps.mkdir(parents=True)
            (maps / "existing-map.json").write_text("{}")

            with self.assertRaises(UsageError) as refusal:
                _clean_room(directory)

            self.assertIn("existing-map.json", str(refusal.exception))

    def test_a_used_output_root_refuses_the_run(self):
        with tempfile.TemporaryDirectory() as directory:
            used = Path(directory) / "cold"
            used.mkdir()
            (used / "run.json").write_text("{}")

            with self.assertRaises(UsageError) as refusal:
                _clean_room(directory)

            self.assertIn("not empty", str(refusal.exception))

    def test_saved_path_reuse_cannot_be_called_a_cold_run(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(UsageError) as refusal:
                _clean_room(directory, durable_replay=True)

            self.assertIn("reuse navigation", str(refusal.exception))

    def test_in_run_replay_cannot_be_called_a_cold_run(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(UsageError):
            _clean_room(directory, replay=True)

    def test_kept_app_state_cannot_be_called_a_cold_run(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(UsageError) as refusal:
                _clean_room(directory, reuse_app=True)

            self.assertIn("app state", str(refusal.exception))

    def test_a_locked_run_cannot_also_be_a_cold_run(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(UsageError) as refusal:
                _clean_room(directory, locked=True)

            self.assertIn("--locked", str(refusal.exception))

    def test_a_journey_off_disk_has_no_lock_to_inspect(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = _clean_room(directory, journey_file=None)

            self.assertEqual(evidence.prior_artifact_count, 0)


class ColdWiringTests(unittest.TestCase):
    """`--cold` must redirect both roots and reach the runner, not just parse."""

    @staticmethod
    def _runner(directory, argv):
        from unittest.mock import patch

        from yewo.config import load_config
        from yewo.testing.cli import YewoCLI
        from yewo.testing.suite import Workflow
        from yewo.testing.wiring import runner_for

        config = load_config()
        journey_file = Path(directory) / "yewo.yaml"
        journey_file.write_text("journeys: []\n")
        workflow = Workflow("checkout", settings={"output_dir": str(Path(directory) / "out")})
        with patch(
            "yewo.testing.wiring.resolve_app_path",
            return_value=Path(directory) / "app.apk",
        ):
            return runner_for(
                workflow,
                YewoCLI(config).parser.parse_args(argv),
                journey_file,
                config,
            )

    def test_a_cold_run_redirects_both_roots_and_disables_saved_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self._runner(directory, ["j.yaml", "--cold", "--device", "Pixel_8"])

            cold_root = Path(directory) / "out" / "cold"
            self.assertEqual(runner.assets_dir, cold_root)
            self.assertEqual(runner.knowledge_root, cold_root / "knowledge")
            self.assertFalse(runner.durable_replay)
            self.assertIsNotNone(runner.cold_start)
            self.assertEqual(runner.cold_start.prior_artifact_count, 0)

    def test_an_ordinary_run_keeps_its_roots_and_saved_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self._runner(directory, ["j.yaml", "--device", "Pixel_8"])

            self.assertEqual(runner.assets_dir, Path(directory) / "out")
            self.assertIsNone(runner.knowledge_root)
            self.assertIsNone(runner.cold_start)
            self.assertTrue(runner.durable_replay)

    def test_a_cold_run_beside_a_lock_is_refused_before_a_runner_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "yewo.lock").write_text(json.dumps({"entries": []}))

            with self.assertRaises(UsageError):
                self._runner(directory, ["j.yaml", "--cold", "--device", "Pixel_8"])


class ColdRecordTests(unittest.TestCase):
    """The record is what tells a later reader the run was cold."""

    def test_the_run_record_carries_the_empty_starting_set(self):
        from fakes import suite_run, walk
        from yewo.testing.models import GeneratedTestCase, ScenarioDraft, UserJourney
        from yewo.testing.taxonomy import Category, CoverageCell, Priority, ScenarioType

        journey = UserJourney(
            id="trainer",
            name="Trainer",
            description="A coach edits a schedule.",
            coverage=[CoverageCell(Category.FUNCTIONAL, ScenarioType.HAPPY_PATH, Priority.P0)],
        )
        scenario = GeneratedTestCase.of(
            journey.coverage[0],
            ScenarioDraft(
                scenario_name="Edit the schedule",
                steps=walk("Open the schedule."),
                expected_outcomes=[
                    {
                        "statement": "The schedule is visible.",
                        "evidence": "A coach edits a schedule.",
                    }
                ],
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            evidence = _clean_room(directory)
            _, written = suite_run(
                [scenario],
                directory,
                journey=journey,
                cold_start=evidence,
                durable_replay=False,
            )

            recorded = written["inputs"]["cold_start"]
            self.assertEqual(recorded["prior_artifact_count"], 0)
            self.assertTrue(recorded["durable_replay_disabled"])
            self.assertNotIn("app_state_cleared", recorded)
            self.assertEqual(recorded["knowledge_root"], evidence.knowledge_root)

    def test_an_ordinary_run_records_no_cold_start(self):
        from fakes import suite_run

        with tempfile.TemporaryDirectory() as directory:
            _, written = suite_run([], directory)

            self.assertIsNone(written["inputs"]["cold_start"])
