"""Handing the device back the way a scenario found it.

Scenarios share one device. A scenario that goes offline and leaves it offline
decides what every later scenario was testing, and no verdict afterwards is
about the state its author declared.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from fakes import PACKAGE, SERIAL, session as device

from yewo.agent.device import restoration
from yewo.agent.device.platform import Platform
from yewo.agent.device.restoration import AndroidStateSnapshot
from yewo.errors import DeviceUnavailable


FOUND = {
    "airplane_mode": "off",
    "battery_level": "100",
    "timezone": "Europe/London",
    "app_locale": "en-GB",
    "accessibility_enabled": "off",
}


class RestorationTests(unittest.TestCase):
    """What is read on the way in is written back on the way out."""

    def setUp(self):
        self.reads = mock.patch.object(restoration, "receipts").start()
        self.adb = mock.patch.object(restoration, "AdbControl").start()
        self.addCleanup(mock.patch.stopall)
        self._answer(FOUND)

    def _answer(self, values):
        """What each read returns, for the snapshot and for the check after."""
        for name, value in values.items():
            getattr(self.reads, name).return_value = value

    def test_a_state_left_alone_is_not_written_back(self):
        """Restoring an untouched state is a write nothing asked for."""
        with restoration.preserved(device()):
            pass

        self.adb.set_airplane_mode.assert_not_called()
        self.adb.reset_battery.assert_not_called()
        self.adb.set_timezone.assert_not_called()
        self.adb.set_app_locale.assert_not_called()
        self.adb.set_accessibility_service.assert_not_called()

    def test_connectivity_a_scenario_took_away_is_given_back(self):
        with restoration.preserved(device()):
            self._answer({"airplane_mode": "on"})

        self.adb.set_airplane_mode.assert_called_once_with(SERIAL, False)

    def test_connectivity_a_scenario_restored_itself_is_left_alone(self):
        with restoration.preserved(device()):
            self._answer({"airplane_mode": "on"})
            self._answer({"airplane_mode": "off"})

        self.adb.set_airplane_mode.assert_not_called()

    def test_a_scenario_that_raised_still_hands_the_device_back(self):
        """The failure is why restoration matters most, not an excuse to skip it."""
        with self.assertRaises(RuntimeError), restoration.preserved(device()):
            self._answer({"airplane_mode": "on"})
            raise RuntimeError("the step failed")

        self.adb.set_airplane_mode.assert_called_once_with(SERIAL, False)

    def test_an_interrupted_scenario_still_hands_the_device_back(self):
        with self.assertRaises(KeyboardInterrupt), restoration.preserved(device()):
            self._answer({"airplane_mode": "on"})
            raise KeyboardInterrupt

        self.adb.set_airplane_mode.assert_called_once_with(SERIAL, False)

    def test_an_overridden_battery_is_handed_back_to_the_real_one(self):
        """The framework reports the hardware or the override, never a level
        it was asked to remember."""
        with restoration.preserved(device()):
            self._answer({"battery_level": "15"})

        self.adb.reset_battery.assert_called_once_with(SERIAL)

    def test_the_clock_is_put_back_where_it_was(self):
        with restoration.preserved(device()):
            self._answer({"timezone": "Asia/Tokyo"})

        self.adb.set_timezone.assert_called_once_with(SERIAL, "Europe/London")

    def test_the_whole_accessibility_list_is_put_back(self):
        """The setting is one list. A scenario that adds a screen reader must
        not cost the owner the others they had running."""
        both = "com.example/a:com.example/b"
        # The snapshot reads once, and the restore reads once to compare.
        self.adb.enabled_accessibility_services.side_effect = [both, "com.example/a"]

        with restoration.preserved(device()):
            pass

        self.adb.set_accessibility_services.assert_called_once_with(SERIAL, both)

    def test_an_accessibility_list_left_alone_is_not_rewritten(self):
        both = "com.example/a:com.example/b"
        self.adb.enabled_accessibility_services.side_effect = [both, both]

        with restoration.preserved(device()):
            pass

        self.adb.set_accessibility_services.assert_not_called()

    def test_a_locale_a_scenario_changed_is_put_back(self):
        with restoration.preserved(device()):
            self._answer({"app_locale": "fr-FR"})

        self.adb.set_app_locale.assert_called_once_with(SERIAL, PACKAGE, "en-GB")

    def test_a_state_that_could_not_be_read_is_not_guessed_at(self):
        """Writing back a value never observed would invent the state."""
        self._answer({"airplane_mode": None})

        with restoration.preserved(device()):
            self._answer({"airplane_mode": "on"})

        self.adb.set_airplane_mode.assert_not_called()

    def test_one_state_that_will_not_restore_does_not_stop_the_others(self):
        self.adb.set_airplane_mode.side_effect = DeviceUnavailable("adb: offline")

        with restoration.preserved(device()):
            self._answer({"airplane_mode": "on", "timezone": "Asia/Tokyo"})

        self.adb.set_timezone.assert_called_once_with(SERIAL, "Europe/London")

    def test_a_failed_restore_is_returned_to_the_scenario_reporter(self):
        self.adb.set_airplane_mode.side_effect = DeviceUnavailable("adb: offline")
        events = []

        with restoration.preserved(device(), events=events):
            self._answer({"airplane_mode": "on"})

        self.assertFalse(events[0].restored)
        self.assertIn("adb: offline", events[0].reason)

    def test_a_failed_restore_does_not_hide_why_the_scenario_ended(self):
        self.adb.set_airplane_mode.side_effect = DeviceUnavailable("adb: offline")

        with self.assertRaises(RuntimeError) as raised, restoration.preserved(device()):
            self._answer({"airplane_mode": "on"})
            raise RuntimeError("the step failed")

        self.assertIn("the step failed", str(raised.exception))

    def test_a_device_with_no_serial_is_left_untouched(self):
        with restoration.preserved(device(serial="")) as found:
            pass

        self.assertIsNone(found)
        self.reads.airplane_mode.assert_not_called()

    def test_an_ios_session_with_no_driver_reads_nothing(self):
        """Every iOS state is read through the driver, and adb reads none of them."""
        with restoration.preserved(device(platform=Platform.IOS)) as found:
            pass

        self.assertIsNone(found)
        self.reads.airplane_mode.assert_not_called()


class SnapshotTests(unittest.TestCase):
    """Reporting a snapshot without inventing what it does not hold."""

    def test_an_unread_state_is_absent_from_the_description(self):
        found = AndroidStateSnapshot(airplane="off", timezone=None)

        self.assertIn("airplane=off", found.describe())
        self.assertNotIn("timezone", found.describe())

    def test_a_snapshot_that_read_nothing_says_so(self):
        self.assertEqual(AndroidStateSnapshot().describe(), "nothing")


class RadioRestorationTests(unittest.TestCase):
    """Airplane mode alone cannot put back what a scenario moved.

    A scenario that asks for wifi_only turns mobile data off and leaves
    airplane mode where it was. A restore watching only airplane mode sees
    nothing move, and data stays off for every scenario after it.
    """

    def setUp(self):
        self.reads = mock.patch.object(restoration, "receipts").start()
        self.adb = mock.patch.object(restoration, "AdbControl").start()
        self.addCleanup(mock.patch.stopall)
        for name, value in FOUND.items():
            getattr(self.reads, name).return_value = value

    def _device(self, radios):
        """A device whose driver answers with these radios, in order, and
        records what it was asked to set."""
        answers = list(radios)
        sent = []

        def execute(script, params=None):
            method = script.removeprefix("mobile: ")
            if method == "getConnectivity":
                return answers.pop(0) if len(answers) > 1 else answers[0]
            if method == "setConnectivity":
                sent.append(params)
            return None

        driver = SimpleNamespace(execute_script=execute)
        device = SimpleNamespace(
            platform=Platform.ANDROID, serial=SERIAL, package_id=PACKAGE, driver=driver
        )
        return device, sent

    def test_mobile_data_a_scenario_turned_off_is_turned_back_on(self):
        on = {"wifi": True, "data": True, "airplaneMode": False}
        wifi_only = {"wifi": True, "data": False, "airplaneMode": False}
        device, sent = self._device([on, wifi_only])

        with restoration.preserved(device) as snap:
            self.assertEqual((snap.wifi, snap.data), ("on", "on"))

        self.assertEqual(
            sent, [{"wifi": True, "data": True, "airplaneMode": False}]
        )
        self.adb.set_airplane_mode.assert_not_called()

    def test_radios_left_alone_are_not_written_back(self):
        on = {"wifi": True, "data": True, "airplaneMode": False}
        device, sent = self._device([on])

        with restoration.preserved(device):
            pass

        self.assertEqual(sent, [])
        self.adb.set_airplane_mode.assert_not_called()

    def test_a_driver_that_cannot_read_the_radios_keeps_the_narrower_guarantee(self):
        """adb can set airplane mode and nothing else, so that is all it claims."""
        device = SimpleNamespace(
            platform=Platform.ANDROID,
            serial=SERIAL,
            package_id=PACKAGE,
            driver=SimpleNamespace(execute_script=lambda script, params=None: None),
        )

        with restoration.preserved(device) as snap:
            self.assertIsNone(snap.wifi)
            self.reads.airplane_mode.return_value = "on"

        self.adb.set_airplane_mode.assert_called_once_with(SERIAL, False)

    def test_a_snapshot_names_the_radios_it_read(self):
        found = AndroidStateSnapshot(airplane="off", wifi="on", data="off")

        self.assertIn("wifi=on", found.describe())
        self.assertIn("data=off", found.describe())
