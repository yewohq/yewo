"""Saved paths are writable by an opted-in local run, never by CI."""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from yewo.agent.memory import replay_files, replay_store
from yewo.errors import UsageError
from yewo.testing import replay_paths


class ReplayPathPolicyTests(unittest.TestCase):
    def test_ci_environment_is_detected(self):
        with patch.dict("os.environ", {"GITHUB_ACTIONS": "true"}, clear=True):
            self.assertTrue(replay_paths.is_ci())

    def test_false_ci_value_does_not_make_a_local_run_read_only(self):
        with patch.dict("os.environ", {"CI": "false"}, clear=True):
            self.assertFalse(replay_paths.is_ci())

    def test_ci_refuses_updates_even_with_the_flag(self):
        with patch.dict("os.environ", {"CI": "1"}, clear=True):
            self.assertFalse(replay_paths.approve_update(Path("path.json"), True))

    def test_noninteractive_update_requires_the_flag(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("sys.stdin.isatty", return_value=False),
        ):
            self.assertFalse(replay_paths.approve_update(Path("path.json"), False))
            self.assertTrue(replay_paths.approve_update(Path("path.json"), True))

    def test_ci_refuses_committed_paths_without_the_shared_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / replay_files.STORE_DIRECTORY / "v1" / "path.json"
            path.parent.mkdir(parents=True)
            path.write_text("{}", encoding="utf-8")

            with (
                patch.dict("os.environ", {"CI": "1"}, clear=True),
                self.assertRaises(UsageError) as caught,
            ):
                replay_paths.validate_path_key(root)

        self.assertIn(replay_store.PATH_KEY_ENV, str(caught.exception))

    def test_ci_accepts_paths_with_the_shared_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / replay_files.STORE_DIRECTORY / "v1" / "path.json"
            path.parent.mkdir(parents=True)
            path.write_text("{}", encoding="utf-8")

            with patch.dict(
                "os.environ",
                {"CI": "1", replay_store.PATH_KEY_ENV: "shared"},
                clear=True,
            ):
                replay_paths.validate_path_key(root)


if __name__ == "__main__":
    unittest.main()
