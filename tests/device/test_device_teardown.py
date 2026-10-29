"""What a run removes from the device, and what it must leave alone.

A rotation of apps fills a device, and a full device fails runs for reasons
that look nothing like storage. Teardown removes what the run put there, which
means it has to know what it put there.
"""

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fakes import PACKAGE, SERIAL, LoggingDriver, log_entries

from yewo.agent.artifacts.layout import RunLayout
from yewo.agent.device.device_logs import DeviceLogRecorder
from yewo.agent.device import touch_input
from yewo.agent.device.adb_control import AdbControl, VisualFeedbackSettings
from yewo.agent.device.touch_input import TouchInputSettings
from yewo.agent.device.files import DeviceFiles
from yewo.agent.device.session import DeviceSession
from yewo.agent.device.simctl_control import SimctlControl
from yewo.agent.device.platform import Platform
from yewo.devices.discovery import Device, RUNNING, SHUTDOWN


class StaleRecordingSweepTests(unittest.TestCase):
    """Removing recordings earlier sessions left behind."""

    def _sweep(self, sdcard, helper_dir):
        listings = {
            "/sdcard": sdcard,
            "/sdcard/Android/data/io.appium.settings/files": helper_dir,
        }
        with (
            patch.object(DeviceFiles, "list_directory", side_effect=lambda _s, d: listings[d]),
            patch.object(DeviceFiles, "remove_file") as remove,
        ):
            count = DeviceFiles.remove_stale_recordings(SERIAL)
        return count, [call.args[1] for call in remove.call_args_list]

    def test_an_appium_screenrecord_file_is_removed(self):
        """Appium names them after eight characters of a uuid."""
        count, removed = self._sweep(["dd1c421d.mp4"], [])

        self.assertEqual(count, 1)
        self.assertEqual(removed, ["/sdcard/dd1c421d.mp4"])

    def test_a_yewo_media_projection_file_is_removed(self):
        count, removed = self._sweep([], ["yewo_32424254-28e2-4508.mp4"])

        self.assertEqual(count, 1)
        self.assertEqual(
            removed,
            ["/sdcard/Android/data/io.appium.settings/files/yewo_32424254-28e2-4508.mp4"],
        )

    def test_test_fixtures_on_the_device_are_left_alone(self):
        """Journeys need media pushed to the device, and it has to survive."""
        count, removed = self._sweep(
            ["bbb_360.mp4", "bbb_external_subs.mp4", "Movies", "holiday.mp4"], []
        )

        self.assertEqual(count, 0)
        self.assertEqual(removed, [])

    def test_a_name_that_merely_contains_a_uuid_is_not_swept(self):
        """fullmatch, not search: my_dd1c421d.mp4 belongs to whoever made it."""
        count, _ = self._sweep(["my_dd1c421d.mp4", "dd1c421d.mp4.bak"], [])

        self.assertEqual(count, 0)


class OwnershipDetectionTests(unittest.TestCase):
    """Deciding whether this run is the one that installed the app.

    The answer has to be positive knowledge, because the consequence of
    guessing wrong is uninstalling somebody else's app.
    """

    def _session(self, serial, packages, package=PACKAGE):
        session = DeviceSession.__new__(DeviceSession)
        session.platform = Platform.ANDROID
        session.device_name = "Pixel_8"
        session.app_path = "app.apk"
        session._installed_by_run = None
        with (
            patch.object(AdbControl, "resolve_emulator_serial", return_value=serial),
            patch.object(AdbControl, "installed_packages", return_value=packages),
            patch.object(AdbControl, "get_apk_info", return_value=(package, None)),
        ):
            session._note_app_ownership()
        return session

    def test_an_app_absent_before_the_run_belongs_to_the_run(self):
        session = self._session(SERIAL, {"com.android.settings"})

        self.assertTrue(session._installed_by_run)

    def test_an_app_already_there_does_not(self):
        session = self._session(
            SERIAL, {"com.android.settings", PACKAGE}
        )

        self.assertFalse(session._installed_by_run)

    def test_a_device_that_has_not_booted_answers_nothing(self):
        """Appium boots the AVD after this runs, and it may hold the app."""
        session = self._session(None, {"anything"})

        self.assertFalse(session._installed_by_run)

    def test_an_adb_call_that_fails_answers_nothing(self):
        """installed_packages returns an empty set when it cannot ask."""
        session = self._session(SERIAL, set())

        self.assertFalse(session._installed_by_run)

    def test_an_answer_the_caller_supplied_is_not_recalculated(self):
        """Recovery knows what the failed session learned.

        By then the app is installed, so looking again would call it
        preexisting and leave it on the device.
        """
        session = DeviceSession.__new__(DeviceSession)
        session.platform = Platform.ANDROID
        session._installed_by_run = True
        session._visual_feedback_before_run = None

        with patch.object(AdbControl, "resolve_emulator_serial") as resolve:
            session._note_app_ownership()

        self.assertTrue(session._installed_by_run)
        resolve.assert_not_called()


class UninstallPolicyTests(unittest.TestCase):
    """Teardown removes the app this run installed, and nothing else."""

    def _control(self, package=PACKAGE, driver=None):
        """A real DeviceControl holds a driver and quits through it, so a stub
        of one does too. Quitting is what makes the log unreadable."""

        class Control:
            serial = SERIAL
            package_id = package

            def __init__(self):
                self.driver = driver if driver is not None else LoggingDriver()

            def quit(self):
                self.driver.quit()

        return Control()

    def _session(self, **overrides):
        session = DeviceSession.__new__(DeviceSession)
        session.platform = Platform.ANDROID
        session.device_name = "Pixel_8"
        session.keep_app = False
        session.reuse_app = False
        session.keep_device = False
        session._installed_by_run = True
        session._ios_simulator_udid = None
        session._ios_simulator_was_running = False
        session.device_control = self._control()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        session.layout = RunLayout.under(Path(directory.name), "s1")
        session.layout.create()
        session._device_logs = DeviceLogRecorder(session.layout, session.platform)
        for name, value in overrides.items():
            setattr(session, name, value)
        return session

    def _clean(self, session, final=True):
        with patch.object(AdbControl, "uninstall_app") as uninstall:
            session._clean_device(final=final)
        return uninstall

    def test_session_recovery_keeps_the_app(self):
        """Recovery opens a new session against the same app.

        The scenarios still to run need it, so a close that is not the end of
        the run must leave it installed.
        """
        uninstall = self._clean(self._session(), final=False)

        uninstall.assert_not_called()

    def test_an_app_this_run_installed_is_removed(self):
        uninstall = self._clean(self._session())

        uninstall.assert_called_once_with(SERIAL, PACKAGE)

    def test_keep_app_leaves_it_installed(self):
        uninstall = self._clean(self._session(keep_app=True))

        uninstall.assert_not_called()

    def test_an_app_that_was_already_installed_is_never_removed(self):
        """It belongs to whoever put it there, not to this run."""
        uninstall = self._clean(self._session(_installed_by_run=False))

        uninstall.assert_not_called()

    def test_an_app_of_unknown_ownership_is_left_alone(self):
        """A cold emulator cannot be asked, and silence is not permission.

        Detection runs before Appium boots the AVD, so on a cold device
        nothing is known. Uninstalling then would remove an app the run
        never installed.
        """
        uninstall = self._clean(self._session(_installed_by_run=None))

        uninstall.assert_not_called()

    def test_reusing_a_build_keeps_it(self):
        """--reuse-app means that build on the device is the one under test."""
        uninstall = self._clean(self._session(reuse_app=True))

        uninstall.assert_not_called()

    def test_nothing_is_uninstalled_without_a_package(self):
        """A session that died before resolving one has no package to name."""
        uninstall = self._clean(self._session(device_control=self._control(package=None)))

        uninstall.assert_not_called()

    def test_an_ios_session_is_left_to_simctl(self):
        """adb cannot reach a simulator."""
        uninstall = self._clean(self._session(platform=Platform.IOS))

        uninstall.assert_not_called()

    def test_starting_a_session_starts_the_device_log_timer(self):
        """The timer is what makes draining happen inside a scenario, and a
        session that never starts it drains only at its boundaries."""
        driver = LoggingDriver(offers=("logcat",), logs={"logcat": log_entries("boot")})
        control = self._control(driver=driver)
        session = self._session(
            app_path="app.apk",
            record_session=False,
            keep_device=False,
            reuse_app=False,
            _appium_url="http://127.0.0.1:4723",
        )

        with (
            patch.object(DeviceSession, "_start_appium"),
            patch.object(DeviceSession, "_note_app_ownership"),
            patch.object(
                AdbControl,
                "read_visual_feedback",
                return_value=VisualFeedbackSettings(False, False),
            ),
            patch.object(
                touch_input, "read_touch_input", return_value=TouchInputSettings(None)
            ),
            patch.object(touch_input, "prepare_touch_input"),
            patch.object(AdbControl, "enable_feedback"),
            patch("yewo.agent.device.session.DeviceControl", return_value=control),
            patch("yewo.agent.device.session.time.sleep"),
        ):
            session.start()

        try:
            self.assertTrue(
                any(
                    thread.name == "yewo-device-logs"
                    for thread in threading.enumerate()
                ),
                "the session started without a device log timer",
            )
        finally:
            session._device_logs.finish(driver)

    def test_the_device_log_is_taken_before_the_driver_is_quit(self):
        """A quit driver reads nothing, and the run worth a device log is the
        one that died on the device."""
        driver = LoggingDriver(offers=("logcat",), logs={"logcat": log_entries("oom")})
        session = self._session(_owns_appium=False, device_control=self._control(driver=driver))
        with (
            patch.object(DeviceSession, "save_recording"),
            patch.object(AdbControl, "uninstall_app"),
        ):
            session.close(final=True)

        self.assertTrue(session.layout.device_log("logcat").exists())
        self.assertTrue(driver.quit_called)

    def test_android_visual_feedback_is_restored_before_disconnect(self):
        settings = VisualFeedbackSettings(pointer_location=True, tap_highlights=False)
        session = self._session(
            _owns_appium=False,
            _visual_feedback_before_run=settings,
            device_control=self._control(),
        )
        with (
            patch.object(DeviceSession, "save_recording"),
            patch.object(AdbControl, "uninstall_app"),
            patch.object(AdbControl, "restore_visual_feedback") as restore,
        ):
            session.close(final=True)

        restore.assert_called_once_with(SERIAL, settings)
        self.assertIsNone(session._visual_feedback_before_run)

    def test_android_touch_input_is_restored_before_disconnect(self):
        settings = TouchInputSettings(stylus_handwriting=None)
        session = self._session(
            _owns_appium=False,
            _touch_input_before_run=settings,
            device_control=self._control(),
        )
        with (
            patch.object(DeviceSession, "save_recording"),
            patch.object(AdbControl, "uninstall_app"),
            patch.object(touch_input, "restore_touch_input") as restore,
        ):
            session.close(final=True, interrupted=True)

        restore.assert_called_once_with(SERIAL, settings)
        self.assertIsNone(session._touch_input_before_run)

    def test_android_input_settings_are_restored_when_startup_fails(self):
        settings = TouchInputSettings(stylus_handwriting=True)
        session = self._session(
            app_path="app.apk",
            record_session=False,
            _appium_url="http://127.0.0.1:4723",
        )
        control = self._control()
        with (
            patch.object(DeviceSession, "_start_appium"),
            patch.object(DeviceSession, "_note_app_ownership"),
            patch("yewo.agent.device.session.DeviceControl", return_value=control),
            patch.object(
                AdbControl,
                "read_visual_feedback",
                return_value=VisualFeedbackSettings(False, False),
            ),
            patch.object(
                touch_input,
                "read_touch_input",
                return_value=settings,
            ),
            patch.object(touch_input, "prepare_touch_input"),
            patch.object(AdbControl, "enable_feedback", side_effect=RuntimeError("offline")),
            patch.object(AdbControl, "restore_visual_feedback") as restore_feedback,
            patch.object(touch_input, "restore_touch_input") as restore_touch,
            self.assertRaisesRegex(RuntimeError, "offline"),
        ):
            session.start()

        restore_feedback.assert_called_once()
        restore_touch.assert_called_once_with(SERIAL, settings)

    def test_a_session_with_no_driver_still_closes(self):
        """A session that died before it opened one has nothing to read."""
        session = self._session(_owns_appium=False, device_control=self._control(driver=None))
        session.device_control.driver = None
        with (
            patch.object(DeviceSession, "save_recording"),
            patch.object(AdbControl, "uninstall_app") as uninstall,
        ):
            session.close(final=True)

        uninstall.assert_called_once()

    def test_a_failing_log_capture_does_not_cost_the_teardown(self):
        """The app, the driver and the Appium server all come down after the
        log is taken. Evidence must not be what strands them."""
        session = self._session(_owns_appium=False)
        with (
            patch.object(DeviceSession, "save_recording"),
            patch.object(
                DeviceSession,
                "_save_device_logs",
                side_effect=RuntimeError("the capture exploded"),
            ),
            patch.object(AdbControl, "uninstall_app") as uninstall,
        ):
            session.close(final=True)

        uninstall.assert_called_once()
        self.assertTrue(session.device_control.driver.quit_called)

    def test_an_interrupted_session_skips_dead_driver_evidence(self):
        session = self._session(_owns_appium=False)
        session.device_control = Mock()
        with (
            patch.object(session._device_logs, "stop") as stop_logs,
            patch.object(DeviceSession, "save_recording") as save_recording,
            patch.object(DeviceSession, "_save_device_logs") as save_logs,
            patch.object(AdbControl, "uninstall_app"),
        ):
            session.close(final=True, interrupted=True)

        stop_logs.assert_called_once_with()
        save_recording.assert_not_called()
        save_logs.assert_not_called()
        session.device_control.quit.assert_called_once_with(interrupted=True)

    def test_an_interrupted_ios_start_stops_the_simulator_it_booted(self):
        session = self._session(
            platform=Platform.IOS,
            device_control=None,
            _owns_appium=False,
            _ios_simulator_udid="new-simulator",
        )
        with (
            patch.object(session._device_logs, "stop"),
            patch.object(SimctlControl, "shutdown") as shutdown,
        ):
            session.close(interrupted=True)

        shutdown.assert_called_once_with("new-simulator")

    def test_an_ios_simulator_that_was_already_running_is_preserved(self):
        session = self._session(
            platform=Platform.IOS,
            device_control=None,
            _owns_appium=False,
            _ios_simulator_udid="existing-simulator",
            _ios_simulator_was_running=True,
        )
        with (
            patch.object(session._device_logs, "stop"),
            patch.object(SimctlControl, "shutdown") as shutdown,
        ):
            session.close(interrupted=True)

        shutdown.assert_not_called()

    def test_ios_target_ownership_is_noted_before_appium_can_boot_it(self):
        session = self._session(platform=Platform.IOS, device_name="iPhone 17 Pro Max")
        target = Device(
            "iPhone 17 Pro Max", "ios", "simulator", SHUTDOWN, "simulator-udid"
        )
        with patch("yewo.agent.device.session.ios_devices", return_value=[target]):
            session._note_ios_target()

        self.assertEqual(session._ios_simulator_udid, "simulator-udid")
        self.assertFalse(session._ios_simulator_was_running)

        target.state = RUNNING
        with patch("yewo.agent.device.session.ios_devices", return_value=[target]):
            session._note_ios_target()
        self.assertTrue(session._ios_simulator_was_running)

    def test_the_video_is_saved_before_the_app_goes(self):
        """Saving is what takes the video off the device.

        Uninstalling first would take the recorder's directory with it, and
        the video with that.
        """
        order = []
        session = self._session(_owns_appium=False)
        session.device_control.quit = lambda: None
        with (
            patch.object(
                DeviceSession,
                "save_recording",
                side_effect=lambda: order.append("recording"),
            ),
            patch.object(
                AdbControl, "uninstall_app", side_effect=lambda *_: order.append("app")
            ),
        ):
            session.close(final=True)

        self.assertEqual(order, ["recording", "app"])
