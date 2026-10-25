"""Reading a device state back, so a capability is evidence and not a claim.

A command exiting 0 says it ran. Whether the state changed is a different
question, and the only answer is to ask the device. The driver answers most of
them on both platforms, and the command line covers what it will not.
"""

import base64
import unittest
from types import SimpleNamespace
from unittest import mock

from yewo.agent.actuation.actions import (
    AccessibilityCapability,
    AppStateCapability,
    BatteryCapability,
    BiometricCapability,
    ClipboardCapability,
    DeeplinkCapability,
    LocaleCapability,
    LocationCapability,
    NetworkCapability,
    PermissionCapability,
)
from yewo.agent.device import capability_controller
from yewo.agent.device.platform import Platform
from yewo.errors import ActionRejected
from fakes import FakeDriver, controller, silent_reads


class ReceiptTests(unittest.TestCase):
    """A command exiting 0 says it ran, not that the state changed."""

    def setUp(self):
        self._reads, self.reads = silent_reads()
        self.addCleanup(self._reads.stop)

    def _apply(self, capability, platform=Platform.ANDROID, answers=None, **reads):
        for name, value in reads.items():
            getattr(self.reads, name).return_value = value
        driver = FakeDriver(answers=answers)
        control = controller(platform=platform, driver=driver)
        with mock.patch.object(capability_controller, "AdbControl"), \
             mock.patch.object(capability_controller, "SimctlControl"):
            return control.execute(capability)

    def test_a_state_the_driver_confirms_comes_back_confirmed(self):
        receipt = self._apply(
            NetworkCapability(mode="offline"),
            answers={
                "getConnectivity": {"wifi": False, "data": False, "airplaneMode": True}
            },
        )

        self.assertTrue(receipt.confirmed)
        self.assertEqual(receipt.source, "mobile: getConnectivity")

    def test_a_state_the_driver_contradicts_is_refused(self):
        """The command succeeded and the device did not change."""
        with self.assertRaises(ActionRejected) as raised:
            self._apply(
                NetworkCapability(mode="offline"),
                answers={
                    "getConnectivity": {
                        "wifi": True, "data": True, "airplaneMode": False
                    }
                },
            )

        self.assertIn("airplane=on", str(raised.exception))
        self.assertIn("airplane=off", str(raised.exception))

    def test_online_needs_one_radio_not_two(self):
        """An emulator with Wi-Fi and no cellular subscription is still online."""
        receipt = self._apply(
            NetworkCapability(mode="online"),
            answers={
                "getConnectivity": {
                    "wifi": True,
                    "data": False,
                    "airplaneMode": False,
                }
            },
        )

        self.assertTrue(receipt.confirmed)

    def test_airplane_mode_makes_a_stale_cellular_flag_irrelevant(self):
        receipt = self._apply(
            NetworkCapability(mode="offline"),
            answers={
                "getConnectivity": {
                    "wifi": False,
                    "data": True,
                    "airplaneMode": True,
                }
            },
        )

        self.assertTrue(receipt.confirmed)

    def test_online_refuses_airplane_off_with_no_active_radio(self):
        with self.assertRaisesRegex(ActionRejected, "observed offline"):
            self._apply(
                NetworkCapability(mode="online"),
                answers={
                    "getConnectivity": {
                        "wifi": False,
                        "data": False,
                        "airplaneMode": False,
                    }
                },
            )

    def test_a_state_that_cannot_be_read_is_not_a_mismatch(self):
        """Unconfirmed is not wrong, so the scenario continues."""
        receipt = self._apply(NetworkCapability(mode="offline"), airplane_mode=None)

        self.assertTrue(receipt.unreadable)
        self.assertFalse(receipt.confirmed)

    def test_the_receipt_falls_back_to_the_setting_when_the_driver_is_silent(self):
        receipt = self._apply(NetworkCapability(mode="offline"), airplane_mode="on")

        self.assertTrue(receipt.confirmed)
        self.assertIn("airplane_mode_on", receipt.source)

    def test_a_capability_with_no_read_reports_no_receipt(self):
        """Inventing one would claim evidence that does not exist."""
        self.assertIsNone(self._apply(BiometricCapability(action="match")))

    def test_a_clipboard_is_read_back_on_either_platform(self):
        """The driver carries clipboard content base64 encoded, both ways."""
        encoded = base64.b64encode(b"hello").decode()

        receipt = self._apply(
            ClipboardCapability(text="hello"), answers={"getClipboard": encoded}
        )

        self.assertTrue(receipt.confirmed)
        self.assertEqual(receipt.observed, "hello")

    def test_an_app_the_driver_still_reports_running_is_refused(self):
        with self.assertRaises(ActionRejected) as raised:
            self._apply(
                AppStateCapability(action="force_stop"), answers={"queryAppState": 4}
            )

        self.assertIn("running", str(raised.exception))

    def test_an_app_the_driver_reports_stopped_is_confirmed(self):
        receipt = self._apply(
            AppStateCapability(action="force_stop"), answers={"queryAppState": 1}
        )

        self.assertTrue(receipt.confirmed)

    def test_a_permission_the_driver_lists_is_confirmed(self):
        receipt = self._apply(
            PermissionCapability(permission="camera", grant=True),
            answers={"getPermissions": ["android.permission.CAMERA"]},
        )

        self.assertTrue(receipt.confirmed)
        self.assertEqual(receipt.source, "mobile: getPermissions")

    def test_a_permission_the_driver_does_not_list_is_refused(self):
        with self.assertRaises(ActionRejected):
            self._apply(
                PermissionCapability(permission="camera", grant=True),
                answers={"getPermissions": ["android.permission.INTERNET"]},
            )


class IosReceiptTests(unittest.TestCase):
    """simctl reads nothing back, and the driver in the same session reads plenty."""

    def setUp(self):
        self._reads, self.reads = silent_reads()
        self.addCleanup(self._reads.stop)

    def _apply(self, capability, answers=None):
        driver = FakeDriver(answers=answers)
        control = controller(platform=Platform.IOS, driver=driver)
        with mock.patch.object(capability_controller, "SimctlControl"), \
             mock.patch.object(capability_controller, "AdbControl"):
            return control.execute(capability)

    def test_the_screen_reader_is_read_back(self):
        """The driver answers with an object, and the bare value is inside it."""
        receipt = self._apply(
            AccessibilityCapability(enabled=True),
            answers={"isVoiceOverEnabled": {"enabled": True}},
        )

        self.assertTrue(receipt.confirmed)
        self.assertEqual(receipt.source, "mobile: isVoiceOverEnabled")

    def test_a_screen_reader_that_did_not_come_on_is_refused(self):
        with self.assertRaises(ActionRejected):
            self._apply(
                AccessibilityCapability(enabled=True),
                answers={"isVoiceOverEnabled": {"enabled": False}},
            )

    def test_a_screen_reader_answered_without_its_object_is_still_read(self):
        """The value comes from the runner on the device, through two wrappers."""
        receipt = self._apply(
            AccessibilityCapability(enabled=True), answers={"isVoiceOverEnabled": True}
        )

        self.assertTrue(receipt.confirmed)

    def test_a_battery_is_refused_rather_than_read_back(self):
        """The override changes the status bar, and the read answers the app."""
        with self.assertRaises(ActionRejected) as raised:
            self._apply(BatteryCapability(level=42))

        self.assertIn("status-bar", str(raised.exception))

    def test_a_permission_is_read_back(self):
        """`simctl privacy` reports nothing, so the read is the only evidence."""
        receipt = self._apply(
            PermissionCapability(permission="camera", grant=True),
            answers={"getPermission": "yes"},
        )

        self.assertTrue(receipt.confirmed)
        self.assertEqual(receipt.source, "mobile: getPermission")

    def test_a_permission_the_device_did_not_take_is_refused(self):
        with self.assertRaises(ActionRejected):
            self._apply(
                PermissionCapability(permission="camera", grant=True),
                answers={"getPermission": "no"},
            )

    def test_a_locale_is_refused_rather_than_read_back(self):
        """A confirmed device locale would say nothing about the running app.

        The localization command sets the simulator and reaches no app that
        already started, so a receipt on the device locale would report a
        change the scenario cannot see.
        """
        with self.assertRaises(ActionRejected) as raised:
            self._apply(LocaleCapability(language="fr", country="FR"))

        self.assertIn("locale", str(raised.exception))



    def test_a_capability_nothing_reads_reports_no_receipt(self):
        self.assertIsNone(self._apply(DeeplinkCapability(url="app://x")))


class SessionTruthTests(unittest.TestCase):
    """Contracts the driver states only when a live session asks it.

    Each of these reads as correct from the method name alone, and answers a
    different question than the caller means to ask.
    """

    def test_params_go_through_as_one_object(self):
        """Appium reads them as `args[0]`, so a list arrives where an object
        belongs and the driver refuses the call."""
        from yewo.agent.device.driver_commands import DriverCommands

        sent = []
        driver = SimpleNamespace(
            execute_script=lambda script, params=None: sent.append((script, params))
        )

        DriverCommands(driver).change("setConnectivity", {"wifi": True})

        self.assertEqual(sent[0][1], {"wifi": True})
        self.assertNotIsInstance(sent[0][1], list)

    def test_a_location_read_drops_the_cache_first(self):
        """`getGeolocation` reads a cache `setGeolocation` does not clear, so
        without this it reports the place before the change."""
        from yewo.agent.device.driver_commands import DriverCommands

        asked = []
        driver = SimpleNamespace(
            execute_script=lambda script, params=None: (
                asked.append(script.removeprefix("mobile: ")),
                {"latitude": 1.0},
            )[1]
        )

        DriverCommands(driver).android_location()

        self.assertEqual(asked, ["refreshGpsCache", "getGeolocation"])

    def test_a_coordinate_is_compared_at_four_places(self):
        """The device answers with the float it stored, which is not the float
        it was given: 35.6812 comes back as 35.6811983."""
        from yewo.agent.device.driver_commands import DriverCommands

        driver = SimpleNamespace(
            execute_script=lambda script, params=None: {
                "latitude": 35.6811983, "longitude": 139.7670983
            }
        )
        control = controller(driver=FakeDriver())
        control.commands = DriverCommands(driver)

        with mock.patch.object(capability_controller, "AdbControl"):
            receipt = control.execute(
                LocationCapability(latitude=35.6812, longitude=139.7671)
            )

        self.assertTrue(receipt.confirmed)

    def test_the_clipboard_is_carried_base64_both_ways(self):
        """Plain text reaches the driver as 'bad base-64'."""
        from yewo.agent.device.driver_commands import DriverCommands

        sent = []
        driver = SimpleNamespace(
            execute_script=lambda script, params=None: sent.append(params)
        )

        DriverCommands(driver).set_clipboard("coupon-123")

        self.assertEqual(
            base64.b64decode(sent[0]["content"]).decode(), "coupon-123"
        )
