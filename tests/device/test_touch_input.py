"""Android handwriting is isolated from Yewo's touch-only sessions."""

import unittest
from unittest.mock import patch

from yewo.agent.device import touch_input
from yewo.agent.device.adb_control import AdbControl
from yewo.agent.device.touch_input import TouchInputSettings

SERIAL = "emulator-5554"


class TouchInputTests(unittest.TestCase):
    def test_enabled_disabled_and_unset_handwriting_are_distinct(self):
        for raw, expected in (
            ("1", TouchInputSettings(True)),
            ("0", TouchInputSettings(False)),
            ("null", TouchInputSettings(None)),
        ):
            with self.subTest(raw=raw), patch.object(AdbControl, "ask", return_value=raw):
                self.assertEqual(touch_input.read_touch_input(SERIAL), expected)

    def test_an_unreadable_setting_is_not_changed(self):
        with patch.object(AdbControl, "ask", return_value=None):
            self.assertIsNone(touch_input.read_touch_input(SERIAL))

    def test_handwriting_is_disabled_for_a_touch_session(self):
        with patch.object(AdbControl, "shell") as shell:
            touch_input.prepare_touch_input(SERIAL, TouchInputSettings(None))

        shell.assert_called_once_with(
            SERIAL,
            "settings",
            "put",
            "secure",
            "stylus_handwriting_enabled",
            "0",
            doing="disable stylus handwriting for touch input",
        )

    def test_disabled_handwriting_is_left_unchanged(self):
        with patch.object(AdbControl, "shell") as shell:
            touch_input.prepare_touch_input(SERIAL, TouchInputSettings(False))

        shell.assert_not_called()

    def test_an_unset_preference_is_restored_by_deleting_it(self):
        with patch.object(AdbControl, "shell") as shell:
            touch_input.restore_touch_input(SERIAL, TouchInputSettings(None))

        self.assertEqual(
            shell.call_args.args[2:5],
            ("delete", "secure", "stylus_handwriting_enabled"),
        )

    def test_an_explicit_preference_is_restored(self):
        with patch.object(AdbControl, "shell") as shell:
            touch_input.restore_touch_input(SERIAL, TouchInputSettings(True))

        self.assertEqual(
            shell.call_args.args[2:6],
            ("put", "secure", "stylus_handwriting_enabled", "1"),
        )


if __name__ == "__main__":
    unittest.main()
