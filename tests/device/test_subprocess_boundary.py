"""The subprocess boundary, exercised with real commands.

`true`, `false` and `echo` are deterministic, and need no network or device.
Each costs a millisecond. A fake here would only prove that the fake agrees
with itself. These tests exist for a defect that proves the point: run_command
never raised the error its callers catch, and every mock agreed that it did.
"""

import subprocess
import unittest
from unittest.mock import patch

from yewo.agent.device.adb_control import AdbControl
from yewo.errors import DeviceUnavailable
from yewo.utils.subprocess_runner import DID_NOT_RUN, SubprocessRunError, run_command

MISSING_BINARY = "yewo-no-such-binary-ckZQ"


class RunCommandTests(unittest.TestCase):
    def test_output_comes_back_decoded(self):
        self.assertEqual(run_command(["echo", "hi"]).stdout.strip(), "hi")

    def test_a_non_zero_exit_is_returned_when_nobody_asked_to_check(self):
        self.assertEqual(run_command(["false"]).returncode, 1)

    def test_checking_turns_a_non_zero_exit_into_the_error_callers_catch(self):
        with self.assertRaises(SubprocessRunError) as raised:
            run_command(["false"], check=True)

        self.assertEqual(raised.exception.returncode, 1)

    def test_a_command_that_cannot_start_raises_the_same_error(self):
        """A missing binary and a failed command are one event to every caller."""
        with self.assertRaises(SubprocessRunError) as raised:
            run_command([MISSING_BINARY])

        self.assertEqual(raised.exception.returncode, DID_NOT_RUN)
        self.assertIn(MISSING_BINARY, raised.exception.stderr)

    def test_uncaptured_output_leaves_the_result_empty_rather_than_none(self):
        result = run_command(["echo", "hi"], capture_output=False)

        self.assertIsNone(result.stdout)
        self.assertEqual(result.returncode, 0)

    def test_a_timeout_is_not_disguised_as_a_command_failure(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            run_command(["sleep", "5"], timeout=0.05)


class DeviceTranslationTests(unittest.TestCase):
    """The device layer's promise, checked through a real subprocess."""

    def test_a_failing_command_reaches_the_caller_as_device_unavailable(self):
        with self.assertRaises(DeviceUnavailable) as raised:
            AdbControl.run_adb_command(["false"], doing="turn the screen on")

        self.assertIn("turn the screen on", str(raised.exception))
        self.assertIsInstance(raised.exception.__cause__, SubprocessRunError)

    def test_a_missing_adb_reaches_the_caller_as_device_unavailable(self):
        with self.assertRaises(DeviceUnavailable):
            AdbControl.run_adb_command([MISSING_BINARY], doing="list devices")

    def test_an_unknown_serial_answers_with_no_packages(self):
        """installed_packages answers with an empty set rather than failing."""
        with patch.object(AdbControl, "ask", return_value=None):
            self.assertEqual(AdbControl.installed_packages(MISSING_BINARY), set())


if __name__ == "__main__":
    unittest.main()
