"""Saved-path ownership and target identity."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from yewo.agent.memory import replay_scenarios


def scenario():
    return SimpleNamespace(
        scenario_uuid="checkout",
        setup="CLEAR_DATA",
        steps=[],
        expected_outcomes=[],
    )


class StoreTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.store = replay_scenarios.TraceStore(
            root=self.root,
            values={"password": "secret"},
            platform="android",
            target_profile="Pixel_8",
            detector="d@abc",
            grounding="marks",
            key=b"test replay digest key",
        )

    def test_a_scenario_with_no_uuid_owns_no_path(self):
        nameless = SimpleNamespace(
            scenario_uuid="", setup="CLEAR_DATA", steps=[], expected_outcomes=[]
        )

        self.assertIsNone(self.store.resolve_identity(nameless))

    def test_an_identity_carries_this_run_and_this_target(self):
        identity = self.store.resolve_identity(scenario())

        self.assertEqual(identity.scenario_uuid, "checkout")
        self.assertEqual(identity.target_profile, "Pixel_8")
        self.assertTrue(identity.scenario_digest)

    def test_a_changed_scenario_reads_as_a_different_path(self):
        changed = SimpleNamespace(
            scenario_uuid="checkout",
            setup="CLEAR_DATA",
            steps=["tap something else"],
            expected_outcomes=[],
        )

        self.assertNotEqual(
            self.store.resolve_identity(scenario()).scenario_digest,
            self.store.resolve_identity(changed).scenario_digest,
        )

    def test_a_replay_with_nothing_on_disk_still_answers(self):
        replay = self.store.build_replay(self.store.resolve_identity(scenario()))

        self.assertFalse(replay.has_path)
        self.assertEqual(replay.values, {"password": "secret"})

    def test_a_target_nothing_identified_gets_no_store_at_all(self):
        nameless = SimpleNamespace(
            platform="android",
            detector="d",
            device=SimpleNamespace(model=None, serial=None),
            app=SimpleNamespace(artifact_sha256=None),
        )

        self.assertIsNone(
            replay_scenarios.build_trace_store(
                root=self.root, values={}, environment=nameless, grounding="marks"
            )
        )

    def test_no_environment_at_all_gets_no_store(self):
        self.assertIsNone(
            replay_scenarios.build_trace_store(
                root=self.root, values={}, environment=None, grounding="marks"
            )
        )

    def test_a_named_target_gets_one(self):
        known = SimpleNamespace(
            platform="android",
            detector="d@abc",
            device=SimpleNamespace(model="Pixel_8", serial="emulator-5554"),
            app=SimpleNamespace(artifact_sha256="abc123"),
        )

        store = replay_scenarios.build_trace_store(
            root=self.root,
            values={},
            environment=known,
            grounding="marks",
            key=b"test replay digest key",
        )

        self.assertIsNotNone(store)
        self.assertEqual(store.resolve_identity(scenario()).target_profile, "Pixel_8")


if __name__ == "__main__":
    unittest.main()
