"""Defaults applied by the AndroidWorld process wrapper."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "android_world"))

from launcher import (
    child_arguments,
    configured_environment,
    has_flag,
    redactor,
    sdk_root,
)


class LauncherTests(unittest.TestCase):
    def test_configured_environment_loads_integration_then_yewo_dotenv(self):
        with tempfile.TemporaryDirectory() as directory:
            yewo = Path(directory) / "yewo"
            root = yewo / "eval" / "integrations" / "android_world"
            root.mkdir(parents=True)
            (root / ".env").write_text("BENCHMARK_ONLY=benchmark\nSHARED=benchmark\n")
            (yewo / ".env").write_text("YEWO_ONLY=yewo\nSHARED=yewo\n")

            environment = configured_environment(root, {"SHARED": "exported"})

            self.assertEqual(environment["BENCHMARK_ONLY"], "benchmark")
            self.assertEqual(environment["YEWO_ONLY"], "yewo")
            self.assertEqual(environment["SHARED"], "exported")

    def test_an_explicit_sdk_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(sdk_root({"ANDROID_SDK_ROOT": directory}), Path(directory).resolve())

    def test_both_flag_spellings_are_recognized(self):
        self.assertTrue(has_flag(["--checkpoint_dir=somewhere"], "--checkpoint_dir"))
        self.assertTrue(has_flag(["--checkpoint_dir", "somewhere"], "--checkpoint_dir"))

    def test_a_new_run_uses_the_artifact_checkpoint_root(self):
        command = child_arguments(Path("run.py"), ["--tasks=One"], Path("checkpoints"))

        self.assertIn("--output_path=checkpoints", command)

    def test_an_explicit_checkpoint_is_not_replaced(self):
        command = child_arguments(
            Path("run.py"), ["--checkpoint_dir=resume-here"], Path("checkpoints")
        )

        self.assertNotIn("--output_path=checkpoints", command)

    def test_logs_remove_environment_dumps_and_secret_values(self):
        redact = redactor({"ANTHROPIC_API_KEY": "private-value"})

        self.assertEqual(
            redact("self._os_env_vars: {'ANTHROPIC_API_KEY': 'private-value'}\n"),
            "self._os_env_vars: <redacted>\n",
        )
        self.assertEqual(redact("failure private-value\n"), "failure <redacted>\n")


if __name__ == "__main__":
    unittest.main()
