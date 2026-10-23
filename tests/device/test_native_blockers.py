"""System UI that stands between the agent and the app under test."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch


class SystemPromptTests(unittest.TestCase):
    """A system prompt is the app's own flow, drawn by the operating system.

    The foreground package stops matching while it is up, so the drift check
    reads it as the app losing focus. Recovery presses back, and back on a
    permission prompt denies it — the journey then fails several steps later
    for a reason that looks nothing like a denied permission.
    """

    def _control(self, package):
        from yewo.agent.device.control import DeviceControl
        from yewo.agent.device.platform import Platform

        control = DeviceControl.__new__(DeviceControl)
        control.platform = Platform.ANDROID
        control.package_id = "com.example.testing_app"
        control.driver = SimpleNamespace(current_package=package)
        return control

    def test_android_recognises_the_permission_packages(self):
        for package in (
            "com.google.android.permissioncontroller",
            "com.android.permissioncontroller",
            "com.android.packageinstaller",
        ):
            self.assertTrue(
                self._control(package).showing_system_prompt(), package
            )

    def test_the_app_itself_is_not_a_prompt(self):
        self.assertFalse(
            self._control("com.example.testing_app").showing_system_prompt()
        )

    def test_another_app_is_not_a_prompt(self):
        """Drifting into a browser is real drift and must still be recovered."""
        self.assertFalse(self._control("com.android.chrome").showing_system_prompt())

    def test_a_driver_that_cannot_answer_reports_no_prompt(self):
        """An unreachable session must not be read as a prompt to leave alone."""
        from yewo.agent.device.control import DeviceControl
        from yewo.agent.device.platform import Platform

        class DeadDriver:
            @property
            def current_package(self):
                raise RuntimeError("no session")

        control = DeviceControl.__new__(DeviceControl)
        control.platform = Platform.ANDROID
        control.driver = DeadDriver()

        self.assertFalse(control.showing_system_prompt())

    def test_ios_claims_nothing(self):
        """Appium reports the app's own modals as alerts too, with nothing to
        tell them apart. Claiming a prompt would tell the planner an app dialog
        belongs to the system and must be left alone, which strands the run.

        Being silent costs little there: iOS recovery brings the app forward
        rather than pressing back, so no permission is answered by accident.
        """
        from yewo.agent.device.control import DeviceControl
        from yewo.agent.device.platform import Platform

        control = DeviceControl.__new__(DeviceControl)
        control.platform = Platform.IOS
        control.driver = SimpleNamespace(
            switch_to=SimpleNamespace(alert=SimpleNamespace(text="Delete this item?"))
        )

        self.assertFalse(control.showing_system_prompt())


class FocusTests(unittest.TestCase):
    """Whether the device is still in the app, asked of a real session shape.

    Both platforms answer from `app_id`, which is an Android package or an
    iOS bundle id. Asking iOS for a field of its own left the check raising
    an AttributeError, which the handler below reads as "still in the app" —
    so drift went undetected and recovery never ran.
    """

    def _control(self, platform, app_id, foreground):
        from yewo.agent.device.control import DeviceControl

        control = DeviceControl.__new__(DeviceControl)
        control.platform = platform
        control.app_id = app_id
        control.package_id = app_id
        control.driver = SimpleNamespace(
            current_package=foreground,
            query_app_state=lambda bundle: 4 if bundle == foreground else 2,
        )
        return control

    def _in_sut(self, control) -> bool:
        from yewo.agent.device.state import DeviceState

        return DeviceState(control).is_still_in_sut()

    def test_an_ios_app_in_the_foreground_is_in_the_app(self):
        from yewo.agent.device.platform import Platform

        control = self._control(Platform.IOS, "com.example.App", "com.example.App")

        self.assertTrue(self._in_sut(control))

    def test_an_ios_device_outside_the_app_is_reported_as_drift(self):
        """The whole point of the check, and it answered True for a year."""
        from yewo.agent.device.platform import Platform

        control = self._control(Platform.IOS, "com.example.App", "com.apple.Preferences")

        self.assertFalse(self._in_sut(control))

    def test_an_android_device_outside_the_app_is_reported_as_drift(self):
        from yewo.agent.device.platform import Platform

        control = self._control(Platform.ANDROID, "com.example.app", "com.android.settings")

        self.assertFalse(self._in_sut(control))

    def test_a_session_that_names_no_app_claims_nothing(self):
        """Nothing can be compared, and claiming drift would recover blindly."""
        from yewo.agent.device.platform import Platform

        control = self._control(Platform.IOS, None, "com.example.App")

        self.assertTrue(self._in_sut(control))

    def test_recovery_brings_the_app_forward_by_its_own_id(self):
        from yewo.agent.device.platform import Platform
        from yewo.agent.device.state import DeviceState

        control = self._control(Platform.IOS, "com.example.App", "com.apple.Preferences")
        control.driver.activate_app = Mock()

        with patch("yewo.agent.device.state.time.sleep"):
            DeviceState(control).recover_to_sut()

        control.driver.activate_app.assert_called_once_with("com.example.App")


class DriftRecoveryTests(unittest.TestCase):
    """What the agent does about being outside the app it is testing."""

    def _device(self, in_sut, system_prompt=False):
        return SimpleNamespace(
            is_still_in_sut=lambda: in_sut,
            showing_system_prompt=lambda: system_prompt,
            recover_to_sut=Mock(),
        )

    def test_back_is_never_pressed_under_a_permission_prompt(self):
        from yewo.agent.device.state import context_drift

        device = self._device(in_sut=False, system_prompt=True)

        note = context_drift(device, last_external_nav=False, settle_time=0)

        device.recover_to_sut.assert_not_called()
        self.assertIn("permission", note)

    def test_the_planner_is_told_not_to_navigate_away(self):
        from yewo.agent.device.state import context_drift

        note = context_drift(
            self._device(in_sut=False, system_prompt=True),
            last_external_nav=False,
            settle_time=0,
        )

        self.assertIn("denies the permission", note)

    def test_real_drift_is_still_recovered(self):
        from yewo.agent.device.state import context_drift

        device = self._device(in_sut=False, system_prompt=False)

        context_drift(device, last_external_nav=False, settle_time=0)

        device.recover_to_sut.assert_called_once()

    def test_staying_in_the_app_needs_no_recovery(self):
        from yewo.agent.device.state import context_drift

        device = self._device(in_sut=True)

        self.assertIsNone(context_drift(device, last_external_nav=False, settle_time=0))
        device.recover_to_sut.assert_not_called()


class KeyboardDismissTests(unittest.TestCase):
    """Whether the keyboard went away, read back rather than assumed.

    The caller spends a screen's one recovery attempt on this answer, so a key
    press that raised nothing must not be reported as a keyboard that closed.
    """

    def _control(self, platform, keyboard_after, raises=False):
        from yewo.agent.device.control import DeviceControl

        control = DeviceControl.__new__(DeviceControl)
        control.platform = platform
        pressed = []

        def key_event(key):
            pressed.append(key)
            if raises:
                raise RuntimeError("session gone")

        control.key_event = key_event
        control.is_keyboard_open = lambda: keyboard_after
        control.driver = SimpleNamespace(
            hide_keyboard=Mock(side_effect=RuntimeError("x") if raises else None)
        )
        return control, pressed

    def test_android_reports_failure_when_the_keyboard_is_still_up(self):
        from yewo.agent.device.platform import Platform

        control, pressed = self._control(Platform.ANDROID, keyboard_after=True)

        self.assertFalse(control.dismiss_keyboard())
        self.assertEqual(pressed, [4])

    def test_android_reports_success_when_the_keyboard_went_away(self):
        from yewo.agent.device.platform import Platform

        control, _pressed = self._control(Platform.ANDROID, keyboard_after=False)

        self.assertTrue(control.dismiss_keyboard())

    def test_an_unanswerable_check_is_taken_as_closed(self):
        """Refusing to believe the press would retry it forever."""
        from yewo.agent.device.platform import Platform

        control, _pressed = self._control(Platform.ANDROID, keyboard_after=None)

        self.assertTrue(control.dismiss_keyboard())

    def test_a_press_that_raised_is_a_failure(self):
        from yewo.agent.device.platform import Platform

        control, _pressed = self._control(
            Platform.ANDROID, keyboard_after=False, raises=True
        )

        self.assertFalse(control.dismiss_keyboard())
