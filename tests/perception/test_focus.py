"""Whether the text lands in the field that was tapped, after the form moves.

Tapping a field opens the keyboard, and a form that scrolls to clear the
keyboard carries the field away from the point that named it. The point then
measures a move the tap itself caused, so it cannot say whether the tap found
the right field.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from yewo.agent.device.control import DeviceControl
from yewo.agent.device.focus import Focus
from yewo.agent.device.platform import Platform
from yewo.defaults import ANDROID_FOCUS_IME_GRACE_TIMEOUT, FOCUS_POLL_TIMEOUT
from yewo.errors import ActionRejected, TargetNotInteractive

SCREEN = (1080, 2400)
POINT = (545, 1248)

BEFORE_KEYBOARD = {"x": 63, "y": 1180, "width": 954, "height": 147}
"""The Email field where the planner read it, with the keyboard closed."""

AFTER_KEYBOARD = {"x": 63, "y": 780, "width": 954, "height": 147}
"""The same field, carried up by the form when the keyboard opened."""

ELSEWHERE = {"x": 63, "y": 820, "width": 954, "height": 147}


class Field:
    """The focused element, reporting whatever the driver says it is now."""

    def __init__(self, driver):
        self.driver = driver

    @property
    def rect(self):
        return self.driver.rect

    def send_keys(self, text):
        self.driver.typed.append(text)

    def get_attribute(self, name):
        return self.driver.attributes.get(name)


class FakeDriver:
    """A driver whose focus is one rect, or nothing."""

    def __init__(self, rect=None):
        self.rect = rect
        self.typed: list = []
        self.attributes: dict = {}

    @property
    def switch_to(self):
        return SimpleNamespace(active_element=Field(self) if self.rect else None)


def control(driver, keyboard=(False, True), focuses=None, platform=Platform.ANDROID):
    """A DeviceControl with no device: just a driver, a keyboard and a tap.

    `keyboard` is what `is_keyboard_open` answers, before the tap and after
    it. `focuses` is the rect the tap puts focus on.
    """
    device = DeviceControl.__new__(DeviceControl)
    device.driver = driver
    device.platform = platform
    device.screen_width, device.screen_height = SCREEN
    answers = iter(keyboard)
    device.is_keyboard_open = lambda: next(answers)
    device.taps: list = []

    def tap(x, y, with_feedback=True):
        device.taps.append((x, y))
        if focuses is not None:
            driver.rect = focuses

    device.tap = tap
    return device


class TypeAfterTheKeyboardOpensTests(unittest.TestCase):
    """A form that scrolls for the keyboard, which is most forms."""

    def test_a_field_carried_off_its_own_point_still_takes_the_text(self):
        """The tap focused the field. The field then moved, and only then."""
        driver = FakeDriver()
        device = control(driver, keyboard=(False, True), focuses=AFTER_KEYBOARD)

        device.type("someone@example.com", *POINT)

        self.assertEqual(driver.typed, ["someone@example.com"])
        self.assertEqual(device.taps, [POINT])

    def test_a_keyboard_that_was_already_open_still_refuses(self):
        """Nothing scrolled, so the point is still evidence of where the field is."""
        driver = FakeDriver()
        device = control(driver, keyboard=(True, True), focuses=AFTER_KEYBOARD)

        with self.assertRaises(ActionRejected) as refusal:
            device.type("someone@example.com", *POINT)

        self.assertIn("landed on a different element", str(refusal.exception))
        self.assertEqual(driver.typed, [])

    def test_a_keyboard_that_cannot_be_read_refuses(self):
        """Neither Appium nor adb answered. Not knowing is not permission."""
        driver = FakeDriver()
        device = control(driver, keyboard=(None, None), focuses=AFTER_KEYBOARD)

        with self.assertRaises(ActionRejected):
            device.type("someone@example.com", *POINT)

    def test_a_field_already_focused_at_that_point_is_not_tapped_again(self):
        """The tap would open the keyboard and scroll the form off the point."""
        driver = FakeDriver(rect=BEFORE_KEYBOARD)
        device = control(driver, keyboard=())

        device.type("someone@example.com", *POINT)

        self.assertEqual(device.taps, [])
        self.assertEqual(driver.typed, ["someone@example.com"])

    def test_a_target_that_holds_no_text_is_still_named_as_such(self):
        """Focus that did not move says the tap hit a label or a picture."""
        driver = FakeDriver(rect=ELSEWHERE)
        device = control(driver, keyboard=(False, False), focuses=ELSEWHERE)

        with (
            patch("yewo.agent.device.focus.time.sleep"),
            self.assertRaises(TargetNotInteractive) as refusal,
        ):
            device.type("someone@example.com", *POINT)

        self.assertIn("holds no text", str(refusal.exception))

    def test_nothing_reporting_focus_is_still_refused(self):
        driver = FakeDriver()
        device = control(driver, keyboard=(False, False))

        with (
            patch("yewo.agent.device.focus.time.sleep"),
            self.assertRaises(ActionRejected) as refusal,
        ):
            device.type("someone@example.com", *POINT)

        self.assertIn("nothing reported focus", str(refusal.exception))

    def test_delayed_focus_gets_a_bounded_input_method_grace_period(self):
        driver = FakeDriver()
        device = control(driver, keyboard=(False, True))
        waits = []

        def settle_focus(_focus, _x, _y, before=None, timeout=FOCUS_POLL_TIMEOUT):
            waits.append(timeout)
            if timeout == ANDROID_FOCUS_IME_GRACE_TIMEOUT:
                driver.rect = AFTER_KEYBOARD

        with patch.object(Focus, "wait_for", autospec=True, side_effect=settle_focus):
            device.type("someone@example.com", *POINT)

        self.assertEqual(
            waits, [FOCUS_POLL_TIMEOUT, ANDROID_FOCUS_IME_GRACE_TIMEOUT]
        )
        self.assertEqual(driver.typed, ["someone@example.com"])

    def test_input_method_without_verifiable_focus_receives_no_text(self):
        driver = FakeDriver()
        device = control(driver, keyboard=(False, True))

        with (
            patch("yewo.agent.device.focus.time.sleep"),
            self.assertRaises(ActionRejected) as refusal,
        ):
            device.type("someone@example.com", *POINT)

        self.assertIn("input-method UI", str(refusal.exception))
        self.assertEqual(driver.typed, [])

    def test_ios_text_entry_uses_the_same_verified_focus_contract(self):
        driver = FakeDriver()
        device = control(
            driver,
            keyboard=(False, True),
            focuses=AFTER_KEYBOARD,
            platform=Platform.IOS,
        )

        device.type("مرحبا 世界", *POINT)

        self.assertEqual(device.taps, [POINT])
        self.assertEqual(driver.typed, ["مرحبا 世界"])


class AssertAtTests(unittest.TestCase):
    """The rule itself, without a device around it."""

    def _focus(self, rect):
        focus = Focus(SimpleNamespace(), screen_area=SCREEN[0] * SCREEN[1])
        focus._rect = lambda: rect
        return focus

    def test_a_moved_layout_does_not_excuse_focus_that_never_moved(self):
        """An unchanged rect is itself proof that nothing scrolled."""
        focus = self._focus(ELSEWHERE)

        with self.assertRaises(TargetNotInteractive):
            focus.assert_at(*POINT, "TYPE", before=ELSEWHERE, layout_moved=True)

    def test_the_point_is_enough_on_its_own(self):
        """A field still under the point needs no excuse at all."""
        self._focus(BEFORE_KEYBOARD).assert_at(*POINT, "TYPE", before=None)

    def test_an_app_rendered_as_one_surface_passes_unchecked(self):
        """It focuses the whole screen, so it contains every point."""
        whole = {"x": 0, "y": 0, "width": SCREEN[0], "height": SCREEN[1]}

        self._focus(whole).assert_at(*POINT, "TYPE", before=whole)


class FocusedFieldReadingTests(unittest.TestCase):
    def test_android_password_attribute_is_normalized(self):
        driver = FakeDriver(rect=BEFORE_KEYBOARD)
        driver.attributes = {"value": "••••••", "password": "true"}

        reading = Focus(driver, screen_area=SCREEN[0] * SCREEN[1]).reading()

        self.assertTrue(reading.is_secure)

    def test_ios_secure_field_type_is_normalized(self):
        driver = FakeDriver(rect=BEFORE_KEYBOARD)
        driver.attributes = {
            "value": "••••••",
            "type": "XCUIElementTypeSecureTextField",
        }

        reading = Focus(driver, screen_area=SCREEN[0] * SCREEN[1]).reading()

        self.assertTrue(reading.is_secure)

    def test_unknown_secure_state_is_not_guessed(self):
        driver = FakeDriver(rect=BEFORE_KEYBOARD)
        driver.attributes = {"value": "text"}

        reading = Focus(driver, screen_area=SCREEN[0] * SCREEN[1]).reading()

        self.assertIsNone(reading.is_secure)


class WaitForTests(unittest.TestCase):
    """Waiting for a condition that can no longer come true spends the budget."""

    def _focus(self, rects):
        answers = iter(rects)
        last = [None]

        def read():
            last[0] = next(answers, last[0])
            return last[0]

        focus = Focus(SimpleNamespace(), screen_area=SCREEN[0] * SCREEN[1])
        focus._rect = read
        return focus

    def test_focus_moving_off_the_point_counts_as_settled(self):
        """A form that scrolled for the keyboard gives this answer and no other."""
        focus = self._focus([ELSEWHERE, AFTER_KEYBOARD])

        with patch("yewo.agent.device.focus.time.sleep") as sleep:
            focus.wait_for(*POINT, before=ELSEWHERE)

        self.assertEqual(sleep.call_count, 1)

    def test_focus_that_stays_put_is_waited_out(self):
        """Nothing moved, so the full budget is what says the tap found no field."""
        focus = self._focus([ELSEWHERE])

        with patch("yewo.agent.device.focus.time.sleep") as sleep:
            focus.wait_for(*POINT, before=ELSEWHERE)

        self.assertGreater(sleep.call_count, 1)

    def test_a_caller_can_request_a_bounded_grace_period(self):
        focus = self._focus([None])

        with patch("yewo.agent.device.focus.time.sleep") as sleep:
            focus.wait_for(*POINT, timeout=0.5)

        self.assertEqual(sleep.call_count, 5)
