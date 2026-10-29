"""Asking the driver what is at a point the detector could not explain."""

import unittest


class FakeFocus:
    """A driver whose focus moves to one rectangle when that point is tapped."""

    def __init__(self, rect=None, lands_on=None):
        self.current = rect
        self.lands_on = lands_on

    def take(self, x, y):
        if self.lands_on and self.lands_on[0] <= x <= self.lands_on[2]:
            self.current = {
                "x": self.lands_on[0],
                "y": self.lands_on[1],
                "width": self.lands_on[2] - self.lands_on[0],
                "height": self.lands_on[3] - self.lands_on[1],
            }

    def rect(self):
        return self.current

    def wait_for(self, x, y, before=None, timeout=None):
        return None

    def contains(self, x, y):
        from yewo.agent.device.focus import bounds_of

        bounds = bounds_of(self.current)
        if bounds is None:
            return False
        return bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3]

    def reading(self):
        from yewo.agent.device.focus import FocusedFieldReading, bounds_of

        return FocusedFieldReading(value=None, bounds=bounds_of(self.current))


class FakeControl:
    """A control whose driver points are half its screenshot pixels."""

    SCALE = 0.5

    def __init__(self, focus, keyboard_states=(False, False)):
        self.focus = focus
        self.taps = []
        self.keyboard_states = iter(keyboard_states)

    def driver_point(self, x, y):
        return round(x * self.SCALE), round(y * self.SCALE)

    def perception_box(self, bounds):
        if bounds is None:
            return None
        return tuple(round(value / self.SCALE) for value in bounds)

    def tap(self, x, y):
        self.taps.append((x, y))
        self.focus.take(*self.driver_point(x, y))

    def is_keyboard_open(self):
        return next(self.keyboard_states)


class FocusProbeTests(unittest.TestCase):
    def test_a_point_that_takes_focus_reports_the_control_in_screen_pixels(self):
        from yewo.agent.device import probe

        control = FakeControl(FakeFocus(lands_on=(100, 200, 300, 400)))

        result = probe.focus_at(control, 400, 600)

        self.assertTrue(result.is_focused)
        self.assertEqual(result.bounds, (200, 400, 600, 800))

    def test_the_tap_uses_screenshot_pixels_and_the_query_uses_driver_points(self):
        from yewo.agent.device import probe

        control = FakeControl(FakeFocus(lands_on=(100, 200, 300, 400)))

        probe.focus_at(control, 400, 600)

        self.assertEqual(control.taps, [(400, 600)])

    def test_a_point_that_takes_no_focus_is_rejected(self):
        from yewo.agent.device import probe

        result = probe.focus_at(FakeControl(FakeFocus()), 400, 600)

        self.assertFalse(result.is_focused)
        self.assertIsNone(result.bounds)
        self.assertEqual(result.verdict, "no_focus")

    def test_focus_that_lands_elsewhere_is_rejected_and_says_so(self):
        from yewo.agent.device import probe

        control = FakeControl(FakeFocus(lands_on=(0, 0, 50, 50)))

        result = probe.focus_at(control, 20, 900)

        self.assertFalse(result.is_focused)
        self.assertTrue(result.focus_changed)
        self.assertEqual(result.verdict, "focus_elsewhere")

    def test_focus_that_never_moved_is_not_a_change(self):
        from yewo.agent.device import probe

        standing = {"x": 0, "y": 0, "width": 50, "height": 50}
        control = FakeControl(FakeFocus(rect=standing))

        result = probe.focus_at(control, 400, 600)

        self.assertFalse(result.focus_changed)

    def test_keyboard_reflow_accepts_the_changed_active_field(self):
        from yewo.agent.device import probe

        control = FakeControl(
            FakeFocus(lands_on=(100, 100, 300, 200)),
            keyboard_states=(False, True),
        )

        result = probe.focus_at(control, 400, 600)

        self.assertTrue(result.is_focused)
        self.assertEqual(result.bounds, (200, 200, 600, 400))

    def test_a_driver_rect_that_is_not_a_rect_reads_as_no_bounds(self):
        from yewo.agent.device.focus import bounds_of

        self.assertIsNone(bounds_of(None))
        self.assertIsNone(bounds_of({"x": 1, "y": 2, "width": None, "height": 4}))
        self.assertIsNone(bounds_of({"x": None, "y": 2, "width": 3, "height": 4}))
        self.assertEqual(bounds_of({"x": 1, "y": 2, "width": 3, "height": 4}), (1, 2, 4, 6))
