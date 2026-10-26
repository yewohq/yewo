"""Touch gestures sent to Appium."""

import unittest
from types import SimpleNamespace
from unittest import mock


class SwipeTests(unittest.TestCase):
    def test_swipe_moves_while_the_finger_is_down(self):
        """Pausing before movement is a long press on the Android launcher."""
        from yewo.agent.device.control import DeviceControl

        control = DeviceControl.__new__(DeviceControl)
        control.driver = SimpleNamespace()
        control._coordinate_scale_x = 1.0
        control._coordinate_scale_y = 1.0
        pointer_actions = mock.Mock()
        builder = mock.Mock(pointer_action=pointer_actions)
        touch = object()

        with (
            mock.patch("yewo.agent.device.control.PointerInput", return_value=touch),
            mock.patch(
                "yewo.agent.device.control.ActionBuilder", return_value=builder
            ) as action_builder,
            mock.patch("yewo.agent.device.control.time.sleep"),
        ):
            control.swipe(540, 2300, 540, 300, duration_ms=275)

        action_builder.assert_called_once_with(control.driver, mouse=touch, duration=275)
        self.assertEqual(
            pointer_actions.method_calls,
            [
                mock.call.move_to_location(540, 2300),
                mock.call.pointer_down(),
                mock.call.move_to_location(540, 300),
                mock.call.release(),
            ],
        )
        builder.perform.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
