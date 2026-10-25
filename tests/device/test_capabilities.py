"""Setting a device state, and refusing to pretend it was set.

Two things are under test. A capability that cannot be applied raises rather
than logging a warning and returning, because a scenario about being offline
must not report a pass while the device stayed online. And a capability that
was applied is read back, because a command exiting 0 says it ran and not that
the state changed.

The driver is asked first and the command line second, so each case says which
one it expects, and the fallback is exercised by a driver that lacks the method.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from fakes import (
    BUNDLE,
    EVERY_CAPABILITY,
    PACKAGE,
    SERIAL,
    FakeDriver,
    controller,
    silent_reads,
)

from yewo.agent.actuation.actions import (
    AppStateCapability,
    BatteryCapability,
    CapabilityKind,
    DeeplinkCapability,
    InterruptCapability,
    LocationCapability,
    MediaInjectCapability,
    NetworkCapability,
    PermissionCapability,
    TimezoneCapability,
)
from yewo.agent.device import capability_controller
from yewo.agent.device.platform import Platform
from yewo.errors import ActionRejected

class DriverFirstTests(unittest.TestCase):
    """The session already holds a driver, so it is asked before adb."""

    def setUp(self):
        self._reads, self.reads = silent_reads()
        self.addCleanup(self._reads.stop)
        self.driver = FakeDriver()
        self.control = controller(driver=self.driver)

    def _run(self, capability):
        with mock.patch.object(capability_controller, "AdbControl") as adb, \
             mock.patch.object(capability_controller, "SimctlControl") as simctl:
            self.control.execute(capability)
        return adb, simctl

    def test_going_offline_turns_off_both_radios(self):
        """The driver takes the radios apart, where adb has only airplane mode."""
        adb, _ = self._run(NetworkCapability(mode="offline"))

        self.assertEqual(
            self.driver.asked("setConnectivity"),
            {"wifi": False, "data": False, "airplaneMode": True},
        )
        adb.set_airplane_mode.assert_not_called()

    def test_wifi_only_leaves_data_off_without_a_bitmask(self):
        self._run(NetworkCapability(mode="wifi_only"))

        self.assertEqual(
            self.driver.asked("setConnectivity"),
            {"wifi": True, "data": False, "airplaneMode": False},
        )
        self.driver.set_network_connection.assert_not_called()

    def test_a_permission_goes_through_the_driver(self):
        adb, _ = self._run(PermissionCapability(permission="camera", grant=True))

        self.assertEqual(
            self.driver.asked("changePermissions"),
            {
                "permissions": ["android.permission.CAMERA"],
                "appPackage": PACKAGE,
                "action": "grant",
            },
        )
        adb.set_app_permission.assert_not_called()

    def test_a_microphone_permission_uses_the_platform_permission(self):
        self._run(PermissionCapability(permission="microphone", grant=True))

        self.assertEqual(
            self.driver.asked("changePermissions"),
            {
                "permissions": ["android.permission.RECORD_AUDIO"],
                "appPackage": PACKAGE,
                "action": "grant",
            },
        )

    def test_injected_media_uses_the_matching_android_collection(self):
        cases = {
            "/tmp/photo.png": "/sdcard/Pictures/photo.png",
            "/tmp/clip.mp4": "/sdcard/Movies/clip.mp4",
            "/tmp/tone.wav": "/sdcard/Music/tone.wav",
            "/tmp/sample.bin": "/sdcard/Download/sample.bin",
        }
        for local_path, destination in cases.items():
            with self.subTest(local_path=local_path), mock.patch.object(
                capability_controller.os.path, "exists", return_value=True
            ):
                adb, _ = self._run(MediaInjectCapability(local_path=local_path))

            adb.push_file.assert_called_once_with(SERIAL, local_path, destination)
            adb.scan_media_file.assert_called_once_with(SERIAL, destination)

    def test_a_deep_link_is_one_call_on_either_platform(self):
        self._run(DeeplinkCapability(url="app://cart"))

        self.assertEqual(self.driver.asked("deepLink"), {"url": "app://cart"})

    def test_stopping_the_app_goes_through_the_driver(self):
        adb, _ = self._run(AppStateCapability(action="force_stop"))

        self.assertIsNotNone(self.driver.asked("terminateApp"))
        adb.force_stop_app.assert_not_called()

    def test_low_memory_needs_no_pid_through_the_driver(self):
        """The adb path needs a pid, so it refuses a stopped app. This does not."""
        adb, _ = self._run(InterruptCapability(type="low_memory"))

        self.assertEqual(
            self.driver.asked("sendTrimMemory"),
            {"pkg": PACKAGE, "level": "RUNNING_CRITICAL"},
        )
        adb.get_app_pid.assert_not_called()

    def test_every_capability_has_a_case_in_the_dispatch(self):
        """A model in the union with no case would fall to the refusal."""
        self.assertEqual(set(EVERY_CAPABILITY), set(CapabilityKind))

        for kind, capability in EVERY_CAPABILITY.items():
            if kind is CapabilityKind.TIMEZONE:
                continue  # Refused on both platforms; covered in RefusalTests.
            with self.subTest(kind=kind.value):
                self._run(capability)


class FallbackTests(unittest.TestCase):
    """A driver without the method is not a reason to skip the capability."""

    def setUp(self):
        self._reads, self.reads = silent_reads()
        self.addCleanup(self._reads.stop)

    def _run(self, capability, without, platform=Platform.ANDROID):
        driver = FakeDriver(without=without)
        control = controller(platform=platform, driver=driver)
        with mock.patch.object(capability_controller, "AdbControl") as adb, \
             mock.patch.object(capability_controller, "SimctlControl") as simctl:
            control.execute(capability)
        return adb, simctl, driver

    def test_connectivity_falls_back_to_airplane_mode(self):
        adb, _, driver = self._run(
            NetworkCapability(mode="offline"), without=["setConnectivity"]
        )

        self.assertIsNotNone(driver.asked("setConnectivity"))
        adb.set_airplane_mode.assert_called_once_with(SERIAL, True)

    def test_a_mode_adb_cannot_set_falls_back_to_the_appium_bitmask(self):
        _, _, driver = self._run(
            NetworkCapability(mode="wifi_only"), without=["setConnectivity"]
        )

        driver.set_network_connection.assert_called_once_with(2)

    def test_a_permission_falls_back_to_pm(self):
        adb, _, _ = self._run(
            PermissionCapability(permission="camera", grant=False),
            without=["changePermissions"],
        )

        adb.set_app_permission.assert_called_once_with(
            SERIAL, PACKAGE, "android.permission.CAMERA", False
        )

    def test_low_memory_falls_back_to_the_pid_path(self):
        adb, _, _ = self._run(
            InterruptCapability(type="low_memory"), without=["sendTrimMemory"]
        )

        adb.get_app_pid.assert_called_once_with(SERIAL, PACKAGE)

    def test_a_stopped_app_with_no_pid_is_refused_on_the_fallback(self):
        driver = FakeDriver(without=["sendTrimMemory"])
        control = controller(driver=driver)
        with mock.patch.object(capability_controller, "AdbControl") as adb:
            adb.get_app_pid.return_value = ""
            with self.assertRaises(ActionRejected) as raised:
                control.execute(InterruptCapability(type="low_memory"))

        self.assertIn("pid", str(raised.exception))

    def test_a_memory_warning_carries_the_bundle_id_the_driver_requires(self):
        """`mobile: sendMemoryWarning` names it required, and refuses without it."""
        _, _, driver = self._run(
            InterruptCapability(type="low_memory"), without=[], platform=Platform.IOS
        )

        self.assertEqual(driver.asked("sendMemoryWarning"), {"bundleId": BUNDLE})

    def test_a_memory_warning_without_a_known_app_goes_to_simctl(self):
        """simctl asks for neither a bundle id nor a running app.

        The driver path needs both, so a session that cannot name the app
        would be refused where simctl would have posted the warning.
        """
        driver = FakeDriver(package=None, platform=Platform.IOS)
        control = controller(platform=Platform.IOS, package="", driver=driver)
        with mock.patch.object(capability_controller, "SimctlControl") as simctl:
            control.execute(InterruptCapability(type="low_memory"))

        simctl.notify_low_memory.assert_called_once_with(SERIAL)
        self.assertIsNone(driver.asked("sendMemoryWarning"))

    def test_a_simulator_location_falls_back_to_simctl(self):
        _, simctl, _ = self._run(
            LocationCapability(latitude=1.0, longitude=2.0),
            without=["setSimulatedLocation"],
            platform=Platform.IOS,
        )

        simctl.set_location.assert_called_once_with(SERIAL, 1.0, 2.0)


class AppIdentityTests(unittest.TestCase):
    """The app under test, asked for by what it is and not by one platform.

    Reading `appPackage` alone left every iOS capability that acts on the app
    with nothing. A permission and a force-stop were refused as though no app
    were installed, and a low-memory warning fell through to simctl, which a
    physical phone does not have.
    """

    def setUp(self):
        self._reads, self.reads = silent_reads()
        self.addCleanup(self._reads.stop)

    def _run(self, capability, platform=Platform.IOS):
        control = controller(platform=platform)
        driver = control.driver
        with mock.patch.object(capability_controller, "AdbControl"), \
             mock.patch.object(capability_controller, "SimctlControl") as simctl:
            control.execute(capability)
        return driver, simctl

    def test_a_permission_reaches_the_app_by_its_bundle_id(self):
        driver, _ = self._run(PermissionCapability(permission="camera", grant=True))

        self.assertEqual(
            driver.asked("setPermission"),
            {"access": {"camera": "yes"}, "bundleId": BUNDLE},
        )

    def test_a_microphone_permission_uses_the_ios_privacy_service(self):
        driver, _ = self._run(
            PermissionCapability(permission="microphone", grant=True)
        )

        self.assertEqual(
            driver.asked("setPermission"),
            {"access": {"microphone": "yes"}, "bundleId": BUNDLE},
        )

    def test_stopping_the_app_reaches_it_by_its_bundle_id(self):
        driver, _ = self._run(AppStateCapability(action="force_stop"))

        self.assertEqual(driver.asked("terminateApp")["bundleId"], BUNDLE)

    def test_a_memory_warning_takes_the_driver_path_on_a_device(self):
        """The simctl fallback is not there on a phone, so this must not need it."""
        driver, simctl = self._run(InterruptCapability(type="low_memory"))

        self.assertEqual(driver.asked("sendMemoryWarning"), {"bundleId": BUNDLE})
        simctl.notify_low_memory.assert_not_called()

    def test_the_android_package_is_still_what_android_capabilities_get(self):
        """One contract, and each platform's own value inside it."""
        driver, _ = self._run(
            PermissionCapability(permission="camera", grant=True),
            platform=Platform.ANDROID,
        )

        self.assertEqual(driver.asked("changePermissions")["appPackage"], PACKAGE)


class SessionIdentityTests(unittest.TestCase):
    """What the session reads off the driver to name the app."""

    def _resolved(self, capabilities, package_id=None):
        from yewo.agent.device.control import DeviceControl

        control = object.__new__(DeviceControl)
        control.driver = SimpleNamespace(capabilities=capabilities)
        control.package_id = package_id
        return control._resolve_app_id()

    def test_an_android_session_is_named_by_its_package(self):
        self.assertEqual(self._resolved({"appPackage": PACKAGE}), PACKAGE)

    def test_an_ios_session_is_named_by_its_bundle_id(self):
        """The driver works it out from the app it installed."""
        self.assertEqual(self._resolved({"bundleId": BUNDLE}), BUNDLE)

    def test_a_prefixed_capability_is_read_too(self):
        self.assertEqual(self._resolved({"appium:bundleId": BUNDLE}), BUNDLE)

    def test_the_apk_reading_answers_where_the_driver_does_not(self):
        """Android resolves the id before the session, by another route."""
        self.assertEqual(self._resolved({}, package_id=PACKAGE), PACKAGE)

    def test_a_session_that_names_no_app_answers_nothing(self):
        self.assertIsNone(self._resolved({"udid": SERIAL}))


class RefusalTests(unittest.TestCase):
    """A capability this platform cannot apply raises, and names both."""

    def setUp(self):
        self._reads, self.reads = silent_reads()
        self.addCleanup(self._reads.stop)

    def _refuse(self, capability, platform=Platform.IOS, package=PACKAGE):
        control = controller(platform, package)
        with mock.patch.object(capability_controller, "AdbControl"), \
             mock.patch.object(capability_controller, "SimctlControl"), \
             self.assertRaises(ActionRejected) as raised:
            control.execute(capability)
        return str(raised.exception)

    def test_the_refusal_names_the_capability_and_the_platform(self):
        message = self._refuse(NetworkCapability(mode="offline"))

        self.assertIn("network", message)
        self.assertIn("ios", message)

    def test_a_timezone_no_path_can_set_is_refused_on_android(self):
        """The setting that looks like it works is not the one the device reads."""
        message = self._refuse(
            TimezoneCapability(timezone="UTC"), platform=Platform.ANDROID
        )

        self.assertIn("timezone", message)

    def test_a_timezone_is_refused_on_ios_too(self):
        self.assertIn("timezone", self._refuse(TimezoneCapability(timezone="UTC")))

    def test_a_call_the_simulator_cannot_place_is_refused(self):
        self.assertIn(
            "interrupt", self._refuse(InterruptCapability(type="incoming_call"))
        )

    def test_a_capability_needing_the_app_is_refused_without_one(self):
        message = self._refuse(
            PermissionCapability(permission="camera", grant=True),
            platform=Platform.ANDROID,
            package="",
        )

        self.assertIn("app id", message)

    def test_a_permission_the_platform_does_not_name_is_refused(self):
        """simctl has no notification service, and 'all' would grant the rest."""
        message = self._refuse(
            PermissionCapability(permission="notifications", grant=True)
        )

        self.assertIn("notifications", message)

    def test_a_limited_grant_is_refused_before_anything_writes_to_it(self):
        """A person picks which photos in a picker, and no command picks again.

        Overwritten, the decision could not be put back, so the refusal has
        to come before the write rather than from the receipt afterwards.
        """
        driver = FakeDriver(
            answers={"getPermission": "limited"}, platform=Platform.IOS
        )
        control = controller(platform=Platform.IOS, driver=driver)

        with mock.patch.object(capability_controller, "SimctlControl"), \
             self.assertRaises(ActionRejected) as raised:
            control.execute(PermissionCapability(permission="photos", grant=True))

        self.assertIn("limited to part of the library", str(raised.exception))
        self.assertIsNone(driver.asked("setPermission"))

    def test_a_file_that_is_not_there_is_refused(self):
        message = self._refuse(
            MediaInjectCapability(local_path="/no/such/file.png"),
            platform=Platform.ANDROID,
        )

        self.assertIn("/no/such/file.png", message)

    def test_a_failing_command_is_reported_as_a_refusal(self):
        """Nothing downstream may read the device as having changed."""
        control = controller(Platform.ANDROID)
        with mock.patch.object(capability_controller, "AdbControl") as adb:
            adb.set_battery_status.side_effect = RuntimeError("adb: device offline")
            with self.assertRaises(ActionRejected) as raised:
                control.execute(BatteryCapability(level=50))

        self.assertIn("device offline", str(raised.exception))


class VocabularyTests(unittest.TestCase):
    """What the planner is offered comes from the models themselves."""

    def test_the_prompt_lists_every_capability(self):
        from yewo.agent.actuation.actions import capability_vocabulary

        listing = capability_vocabulary()

        for kind in CapabilityKind:
            self.assertIn(kind.value, listing)

    def test_a_capability_s_choices_reach_the_prompt(self):
        """A mode the model cannot see is a mode it will not ask for."""
        from yewo.agent.actuation.actions import capability_vocabulary

        listing = capability_vocabulary()

        for mode in ("offline", "online", "wifi_only", "data_only", "airplane"):
            self.assertIn(mode, listing)

    def test_an_action_carries_a_typed_capability_not_a_dict(self):
        from yewo.agent.actuation.actions import CapabilityAction

        action = CapabilityAction.model_validate(
            {
                "type": "capability",
                "capability": {"kind": "network", "mode": "offline"},
                "reason": "the scenario is about losing connectivity",
            }
        )

        self.assertIsInstance(action.capability, NetworkCapability)
        self.assertEqual(action.capability.mode, "offline")

    def test_a_mode_no_capability_offers_is_refused(self):
        from pydantic import ValidationError

        from yewo.agent.actuation.actions import CapabilityAction

        with self.assertRaises(ValidationError):
            CapabilityAction.model_validate(
                {
                    "type": "capability",
                    "capability": {"kind": "network", "mode": "2g"},
                    "reason": "throttle",
                }
            )


class DeviceTruthTests(unittest.TestCase):
    """Where a tool's real contract differs from the obvious reading of it.

    A write can land somewhere the device does not read, a command can exit 0
    having done nothing, and an argument can be one the tool does not accept.
    A fixture agrees with whichever reading it was written from, so each of
    these pins the contract instead.
    """

    def test_the_timezone_is_read_from_the_property_the_system_keeps(self):
        """`settings global timezone` and the property disagree.

        Writing the setting leaves the property alone, so reading the setting
        back confirms the write and says nothing about the device.
        """
        from yewo.agent.device import receipts as module

        with mock.patch(
            "yewo.agent.device.adb_control.AdbControl.ask",
            return_value="Europe/London",
        ) as ask:
            self.assertEqual(module.timezone(SERIAL), "Europe/London")

        self.assertIn("persist.sys.timezone", ask.call_args[0])

    def test_a_permission_the_app_never_asked_for_is_not_merely_unreadable(self):
        """`pm grant` exits 0 for one of these and grants nothing."""
        from yewo.agent.device import receipts as module

        dump = "    requested permissions:\n      android.permission.INTERNET\n"
        with mock.patch(
            "yewo.agent.device.adb_control.AdbControl.ask", return_value=dump
        ):
            answer = module.permission_granted(
                SERIAL, PACKAGE, "android.permission.CAMERA"
            )

        self.assertEqual(answer, module.NOT_REQUESTED)


class BothWaysFailedTests(unittest.TestCase):
    """When the driver and the command line both fail, which reason survives.

    The driver names the constraint and the command line reports the symptom.
    Refused a call on a physical phone, the driver says the method is for
    emulators only, and `adb emu` exits non-zero with nothing to say. Losing
    the first message leaves the one that explains nothing.
    """

    def setUp(self):
        self._reads, self.reads = silent_reads()
        self.addCleanup(self._reads.stop)

    def test_the_driver_s_reason_leads_and_both_are_kept(self):
        from yewo.errors import DeviceUnavailable

        driver = FakeDriver(without=["gsmCall"])
        control = controller(driver=driver)
        with mock.patch.object(capability_controller, "AdbControl") as adb:
            adb.simulate_incoming_call.side_effect = DeviceUnavailable(
                "Failed to simulate incoming call: adb -s X emu gsm call 555"
            )
            with self.assertRaises(ActionRejected) as raised:
                control.execute(InterruptCapability(type="incoming_call"))

        message = str(raised.exception)
        self.assertIn("gsmCall", message)
        self.assertLess(message.index("driver refused"), message.index("then failed"))

    def test_a_command_line_that_works_is_not_an_error(self):
        driver = FakeDriver(without=["gsmCall"])
        control = controller(driver=driver)
        with mock.patch.object(capability_controller, "AdbControl") as adb:
            control.execute(InterruptCapability(type="incoming_call"))

        adb.simulate_incoming_call.assert_called_once()
