"""Screenshot pixels map to the coordinate space the device driver accepts."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from yewo.agent.device.control import DeviceControl
from yewo.agent.device.platform import Platform


class ScreenshotCoordinateSpaceTests(unittest.TestCase):
    def _control(self):
        control = DeviceControl.__new__(DeviceControl)
        control._driver_width = 440
        control._driver_height = 956
        control.screen_width = 440
        control.screen_height = 956
        control._coordinate_scale_x = 1.0
        control._coordinate_scale_y = 1.0
        return control

    @staticmethod
    def _png_header(width: int, height: int) -> bytes:
        return (
            b"\x89PNG\r\n\x1a\n"
            + b"\x00\x00\x00\x0dIHDR"
            + width.to_bytes(4, "big")
            + height.to_bytes(4, "big")
        )

    def test_a_retina_screenshot_calibrates_perception_pixels_to_driver_points(self):
        control = self._control()

        control._calibrate_coordinate_space(self._png_header(880, 1912))

        self.assertEqual((control.screen_width, control.screen_height), (880, 1912))
        self.assertEqual(control.driver_point(396, 1510), (198, 755))

    def test_a_same_size_screenshot_keeps_android_coordinates_unchanged(self):
        control = self._control()

        control._calibrate_coordinate_space(self._png_header(440, 956))

        self.assertEqual(control.driver_point(396, 755), (396, 755))


class IOSTextEntryTests(unittest.TestCase):
    def test_ios_verifies_focus_then_types_through_the_focused_element(self):
        element = SimpleNamespace(
            send_keys=Mock(),
            get_attribute=Mock(return_value=None),
            text="",
            rect={"x": 0, "y": 0, "width": 20, "height": 20},
        )
        control = DeviceControl.__new__(DeviceControl)
        control.platform = Platform.IOS
        control.driver = SimpleNamespace(
            switch_to=SimpleNamespace(active_element=element)
        )
        control._driver_width = 440
        control._driver_height = 956
        control.is_keyboard_open = Mock(side_effect=(False, True))

        def focus_target(_x, _y):
            element.rect = {"x": 500, "y": 1100, "width": 100, "height": 300}

        control.tap = Mock(side_effect=focus_target)

        control.type("test@email.com", 537, 1260)

        control.tap.assert_called_once_with(537, 1260)
        element.send_keys.assert_called_once_with("test@email.com")


if __name__ == "__main__":
    unittest.main()
