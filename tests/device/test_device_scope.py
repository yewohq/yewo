"""Attaching to a benchmark-owned Android device."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from yewo.agent.device.adb_control import AdbControl, VisualFeedbackSettings
from yewo.agent.device.control import DeviceControl
from yewo.agent.device.platform import Platform
from yewo.agent.device.scope import DeviceScope
from yewo.agent.device.session import DeviceSession
from yewo.agent.device.state import DeviceState
from yewo.errors import DeviceUnavailable


class DeviceScopedControlTests(unittest.TestCase):
    def test_attach_requires_a_running_android_target(self):
        with (
            patch.object(AdbControl, "resolve_emulator_serial", return_value=None),
            self.assertRaisesRegex(DeviceUnavailable, "is not running"),
        ):
            DeviceControl(
                device_name="AndroidWorldAvd",
                app_path=None,
                scope=DeviceScope.DEVICE,
            )

    def test_attach_uses_the_existing_serial_without_an_app(self):
        captured = {}

        def open_session(_control, options):
            captured.update(options.to_capabilities())
            return object()

        with (
            patch.object(AdbControl, "resolve_emulator_serial", return_value="emulator-5554"),
            patch.object(DeviceControl, "_start_session", open_session),
            patch.object(DeviceControl, "_init_device_info"),
            patch.object(AdbControl, "get_apk_info") as apk_info,
        ):
            control = DeviceControl(
                device_name="AndroidWorldAvd",
                app_path=None,
                scope=DeviceScope.DEVICE,
            )

        self.assertEqual(captured["appium:udid"], "emulator-5554")
        self.assertTrue(captured["appium:noReset"])
        self.assertTrue(captured["appium:disableSuppressAccessibilityService"])
        self.assertNotIn("appium:app", captured)
        self.assertNotIn("appium:appPackage", captured)
        self.assertTrue(control.keep_device)
        apk_info.assert_not_called()

    def test_device_scope_accepts_any_foreground_package(self):
        control = SimpleNamespace(
            scope=DeviceScope.DEVICE,
            platform=Platform.ANDROID,
            driver=SimpleNamespace(current_package="com.android.settings"),
        )

        self.assertTrue(DeviceState(control).is_still_in_sut())

    def test_device_scope_never_recovers_to_an_app(self):
        control = SimpleNamespace(
            scope=DeviceScope.DEVICE,
            key_event=Mock(),
            driver=SimpleNamespace(activate_app=Mock()),
        )

        DeviceState(control).recover_to_sut()

        control.key_event.assert_not_called()
        control.driver.activate_app.assert_not_called()


class DeviceScopedSessionTests(unittest.TestCase):
    def test_attach_does_not_require_an_app_artifact(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("yewo.agent.device.session.run_android_preflight") as preflight,
            patch("yewo.agent.device.session.attach_run_log", return_value=Mock()),
        ):
            session = DeviceSession(
                device_name="AndroidWorldAvd",
                app_path=None,
                session_dir=Path(directory),
                scope=DeviceScope.DEVICE,
            )

        preflight.assert_not_called()
        self.assertIsNone(session.app_path)

    def test_app_scope_still_requires_an_artifact(self):
        with self.assertRaisesRegex(ValueError, "requires an app path"):
            DeviceSession(device_name="Pixel_8", app_path=None)

    def test_device_scope_disables_all_visual_feedback(self):
        control = SimpleNamespace(serial="emulator-5554", driver=object())
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("yewo.agent.device.session.attach_run_log", return_value=Mock()),
        ):
            session = DeviceSession(
                device_name="AndroidWorldAvd",
                app_path=None,
                session_dir=Path(directory),
                scope=DeviceScope.DEVICE,
            )
            with (
                patch.object(DeviceSession, "_start_appium"),
                patch.object(DeviceSession, "_note_app_ownership"),
                patch("yewo.agent.device.session.DeviceControl", return_value=control),
                patch.object(
                    AdbControl,
                    "read_visual_feedback",
                    return_value=VisualFeedbackSettings(True, True),
                ) as read_feedback,
                patch.object(AdbControl, "enable_feedback") as enable_feedback,
                patch.object(session._device_logs, "start"),
                patch("yewo.agent.device.session.time.sleep"),
            ):
                session.start()

        read_feedback.assert_called_once_with("emulator-5554")
        enable_feedback.assert_called_once_with(
            serial="emulator-5554",
            pointer_location=False,
            tap_highlights=False,
        )
