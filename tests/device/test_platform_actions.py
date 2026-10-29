"""Actions exposed only on platforms that can execute them."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from yewo.agent.device.control import DeviceControl
from yewo.agent.device.platform import Platform


class PlatformActionTests(unittest.TestCase):
    def test_ios_rejects_android_integer_keycodes(self):
        control = DeviceControl.__new__(DeviceControl)
        control.platform = Platform.IOS
        control.driver = SimpleNamespace(press_keycode=Mock())

        with self.assertRaisesRegex(ValueError, "only on Android"):
            control.key_event(4)

        control.driver.press_keycode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
