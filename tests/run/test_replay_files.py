"""Keeping a judged path between runs, and refusing every doubtful one.

A trace is an optimisation. So a file that cannot be read, or was written
by another version, or belongs to another target, costs a run nothing: it
plans the scenario, which is what it did before any of this existed.

What it must never do is fail a run, and what it must never hold is a value
the journey declared.
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from yewo.agent.memory import replay_files, replay_store


def identity(**overrides):
    fields = {
        "scenario_uuid": "scenario-1",
        "scenario_digest": "d" * 16,
        "platform": "android",
        "target_profile": "Pixel_8",
        "detector": "detector@abcdef12",
        "grounding": "marks",
    }
    fields.update(overrides)
    return replay_store.TraceIdentity(**fields)


def trace(**overrides):
    return replay_store.ReplayTrace(identity=identity(**overrides))


class RoundTripTests(unittest.TestCase):
    """What was written is what comes back."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.identity = identity()
        self.path = replay_files.trace_path(self.root, self.identity)

    def test_a_written_trace_reads_back(self):
        replay_files.write_trace(self.path, trace())

        self.assertEqual(replay_files.read_trace(self.path, self.identity), trace())

    def test_a_missing_file_is_simply_no_trace(self):
        self.assertIsNone(replay_files.read_trace(self.path, self.identity))

    def test_two_targets_of_one_scenario_keep_separate_paths(self):
        other = identity(target_profile="Pixel_6a")

        self.assertNotEqual(
            replay_files.trace_path(self.root, other), self.path
        )

    def test_a_profile_naming_a_directory_cannot_escape_the_store(self):
        """A target name reaches this from a file somebody else wrote."""
        path = replay_files.trace_path(self.root, identity(target_profile="../../etc"))

        self.assertNotIn("..", path.parts)

    def test_writing_again_replaces_rather_than_appends(self):
        replay_files.write_trace(self.path, trace())
        replay_files.write_trace(self.path, trace(scenario_digest="e" * 16))

        payload = json.loads(self.path.read_text())

        self.assertEqual(payload["identity"]["scenario_digest"], "e" * 16)

    def test_no_part_file_is_left_behind(self):
        replay_files.write_trace(self.path, trace())

        self.assertEqual(
            [p.name for p in self.path.parent.iterdir()], [self.path.name]
        )


class RefusalTests(unittest.TestCase):
    """Every doubtful file ends the same way: plan the scenario."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.identity = identity()
        self.path = replay_files.trace_path(self.root, self.identity)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def test_a_file_that_is_not_json_is_ignored(self):
        self.path.write_text("not a trace at all")

        self.assertIsNone(replay_files.read_trace(self.path, self.identity))

    def test_a_file_holding_a_list_is_ignored(self):
        self.path.write_text("[]")

        self.assertIsNone(replay_files.read_trace(self.path, self.identity))

    def test_a_trace_from_another_schema_is_ignored(self):
        payload = trace().model_dump(by_alias=True)
        payload["schema"] = replay_store.SCHEMA + 1
        self.path.write_text(json.dumps(payload))

        self.assertIsNone(replay_files.read_trace(self.path, self.identity))

    def test_a_trace_a_looser_matcher_accepted_is_ignored(self):
        """A stricter matcher must distrust what an older one let through,
        even where the shape of the file did not change."""
        payload = trace().model_dump(by_alias=True)
        payload["matcher"] = replay_store.MATCHER - 1
        self.path.write_text(json.dumps(payload))

        self.assertIsNone(replay_files.read_trace(self.path, self.identity))

    def test_a_trace_for_another_platform_is_ignored(self):
        replay_files.write_trace(self.path, trace())

        self.assertIsNone(
            replay_files.read_trace(self.path, identity(platform="ios"))
        )

    def test_a_trace_for_another_detector_is_ignored(self):
        replay_files.write_trace(self.path, trace())

        self.assertIsNone(
            replay_files.read_trace(self.path, identity(detector="detector@99999999"))
        )

    def test_a_trace_for_another_scenario_digest_is_ignored(self):
        """The cell may be unchanged while the generated steps are not."""
        replay_files.write_trace(self.path, trace())

        self.assertIsNone(
            replay_files.read_trace(self.path, identity(scenario_digest="f" * 16))
        )

    def test_a_malformed_record_is_ignored(self):
        self.path.write_text(json.dumps({"schema": replay_store.SCHEMA,
                                         "matcher": replay_store.MATCHER}))

        self.assertIsNone(replay_files.read_trace(self.path, self.identity))

    def test_a_write_that_cannot_land_says_so_rather_than_raising(self):
        blocked = self.root / "a-file"
        blocked.write_text("in the way")

        self.assertFalse(replay_files.write_trace(blocked / "deeper.json", trace()))


class BuildTests(unittest.TestCase):
    """A rebuild is recorded, and is not a reason to discard the path."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.identity = identity()
        self.path = replay_files.trace_path(self.root, self.identity)

    def test_a_trace_learned_on_another_build_is_still_read(self):
        """Gating on the artifact would miss on every run after a commit,
        which is the run this phase exists for. The build is a fact about
        the trace, and each step is verified against the screen instead."""
        replay_files.write_trace(self.path, trace().model_copy(update={"learned_from": "old"}))

        read = replay_files.read_trace(self.path, self.identity)

        self.assertIsNotNone(read)
        self.assertEqual(read.learned_from, "old")




class PathTests(unittest.TestCase):
    """Both path segments arrive from a file a person edits."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_no_identity_can_name_a_directory_above_the_store(self):
        for escaping in ("../../outside", "..", "a/b/c", ""):
            with self.subTest(value=escaping):
                path = replay_files.trace_path(
                    self.root, identity(scenario_uuid=escaping)
                )

                self.assertNotIn("..", path.parts)
                self.assertTrue(path.is_relative_to(self.root))

    def test_a_scenario_whose_uuid_is_unusable_still_gets_one_file(self):
        path = replay_files.trace_path(self.root, identity(scenario_uuid="///"))

        self.assertTrue(path.is_relative_to(self.root))
        self.assertTrue(path.name.endswith(".json"))



class OverwriteTests(unittest.TestCase):
    """What a run is allowed to write over a path that already works."""

    def setUp(self):
        from yewo.agent.memory import replay_scenarios

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.store = replay_scenarios.TraceStore(
            root=self.root, values={}, platform="android",
            target_profile="Pixel_8", detector="d", grounding="marks",
            key=b"test replay digest key",
        )

    def _scenario(self):
        return SimpleNamespace(
            scenario_uuid="s", setup="CLEAR_DATA", steps=[], expected_outcomes=[]
        )

    def _replay(self, steps=1):
        from yewo.agent.memory.replay_learning import DurableReplay

        replay = DurableReplay(
            trace=None, values={}, names=replay_store.ValueNames()
        )
        for _ in range(steps):
            replay.record_step(object(), object())
        return replay

    def test_a_failed_scenario_writes_nothing(self):
        """A path to a failure is a path to a result already known."""
        self.store.save_trace(self._scenario(), self._replay(), "FAILED")

        self.assertEqual(list(self.root.rglob("*.json")), [])

    def test_a_run_that_took_no_step_writes_nothing(self):
        self.store.save_trace(self._scenario(), self._replay(steps=0), "PASSED")

        self.assertEqual(list(self.root.rglob("*.json")), [])


if __name__ == "__main__":
    unittest.main()
