import unittest
from unittest import mock

from pydantic import ValidationError

from fakes import BUNDLE, FakeClock, FakeDriver, controller, session

from yewo.agent.actuation.actions import (
    AppStateCapability,
    DeviceLockCapability,
    NetworkCapability,
    OrientationCapability,
)
from yewo.agent.device import restoration
from yewo.agent.device.capability_maps import active_ios_condition, ios_network_profile
from yewo.agent.device import emulator_network
from yewo.agent.device.emulator_network import EmulatorNetworkShape
from yewo.agent.device.driver_commands import DriverCommands
from yewo.agent.device.platform import Platform
from yewo.errors import ActionRejected


class SequencedDriver(FakeDriver):
    """A driver whose lifecycle reads advance through real transitions."""

    def __init__(self, sequences=None, **kwargs):
        super().__init__(**kwargs)
        self.sequences = {name: list(values) for name, values in (sequences or {}).items()}

    def execute_script(self, script, params=None):
        method = script.removeprefix("mobile: ")
        if method in self.sequences:
            self.calls.append((method, params or {}))
            values = self.sequences[method]
            return values.pop(0) if values else None
        return super().execute_script(script, params)


class OrientationTests(unittest.TestCase):
    def test_both_platforms_set_and_confirm_orientation(self):
        for platform in Platform:
            with self.subTest(platform=platform.value):
                driver = FakeDriver(platform=platform)
                receipt = controller(platform=platform, driver=driver).execute(
                    OrientationCapability(orientation="landscape")
                )

                self.assertEqual(driver.orientation, "LANDSCAPE")
                self.assertTrue(receipt.confirmed)

    def test_ios_orientation_is_restored_after_an_exception(self):
        driver = FakeDriver(platform=Platform.IOS, package=BUNDLE)
        device = session(platform=Platform.IOS, driver=driver)
        device.app_id = BUNDLE
        capability = OrientationCapability(orientation="landscape")

        with self.assertRaises(RuntimeError), restoration.preserved(
            device, capabilities=[capability]
        ):
            driver.orientation = "LANDSCAPE"
            raise RuntimeError("scenario ended")

        self.assertEqual(driver.orientation, "PORTRAIT")

    def test_android_orientation_is_restored(self):
        driver = FakeDriver()
        driver.orientation = "LANDSCAPE"
        found = restoration.AndroidStateSnapshot(orientation="portrait")

        restoration.restore_android(
            "emulator-5554", None, found, DriverCommands(driver)
        )

        self.assertEqual(driver.orientation, "PORTRAIT")


class AppLifecycleTests(unittest.TestCase):
    def test_background_resume_confirms_both_transitions(self):
        clock = FakeClock()
        driver = SequencedDriver(sequences={"queryAppState": [3, 4]})
        receipt = controller(driver=driver, sleep=clock.sleep).execute(
            AppStateCapability(action="background_resume", duration_seconds=2.5)
        )

        self.assertEqual(clock.elapsed, 2.5)
        self.assertTrue(receipt.confirmed)
        self.assertEqual(
            [name for name, _ in driver.calls if name in ("backgroundApp", "activateApp")],
            ["backgroundApp", "activateApp"],
        )

    def test_background_failure_still_recovers_the_app(self):
        driver = SequencedDriver(sequences={"queryAppState": [4]})

        with self.assertRaisesRegex(ActionRejected, "instead of a background"):
            controller(driver=driver, sleep=lambda _: None).execute(
                AppStateCapability(action="background_resume", duration_seconds=0)
            )

        self.assertIsNotNone(driver.asked("activateApp"))

    def test_interruption_during_background_wait_still_recovers_the_app(self):
        driver = SequencedDriver(sequences={"queryAppState": [3]})

        with self.assertRaises(KeyboardInterrupt):
            controller(
                driver=driver,
                sleep=mock.Mock(side_effect=KeyboardInterrupt),
            ).execute(AppStateCapability(action="background_resume"))

        self.assertIsNotNone(driver.asked("activateApp"))

    def test_relaunch_confirms_stop_then_foreground(self):
        driver = SequencedDriver(sequences={"queryAppState": [1, 4]})
        receipt = controller(driver=driver).execute(
            AppStateCapability(action="relaunch")
        )

        self.assertTrue(receipt.confirmed)
        self.assertIsNotNone(driver.asked("terminateApp"))
        self.assertIsNotNone(driver.asked("activateApp"))


class LockCycleTests(unittest.TestCase):
    def test_lock_cycle_is_atomic_and_confirmed(self):
        clock = FakeClock()
        driver = FakeDriver()
        receipt = controller(driver=driver, sleep=clock.sleep).execute(
            DeviceLockCapability(duration_seconds=3)
        )

        self.assertEqual(clock.elapsed, 3)
        self.assertTrue(receipt.confirmed)
        self.assertEqual(driver._locked, False)

    def test_an_unreadable_initial_lock_state_is_refused(self):
        driver = FakeDriver(answers={"isLocked": None})

        with self.assertRaisesRegex(ActionRejected, "unreadable"):
            controller(driver=driver).execute(DeviceLockCapability(duration_seconds=0))

        self.assertIsNone(driver.asked("lock"))

    def test_an_unreadable_locked_state_still_unlocks(self):
        driver = SequencedDriver(sequences={"isLocked": [False, None]})

        with self.assertRaisesRegex(ActionRejected, "not confirmed"):
            controller(driver=driver).execute(DeviceLockCapability(duration_seconds=0))

        self.assertIsNotNone(driver.asked("unlock"))

    def test_cycle_durations_are_bounded(self):
        with self.assertRaises(ValidationError):
            DeviceLockCapability(duration_seconds=61)
        with self.assertRaises(ValidationError):
            AppStateCapability(action="background_resume", duration_seconds=-1)


def network_conditions(active=""):
    return [
        {
            "name": "Network Link",
            "identifier": "SlowNetworkCondition",
            "isActive": bool(active),
            "activeProfile": active,
            "profiles": [
                {
                    "name": "Edge network",
                    "identifier": "SlowNetworkEdge",
                    "description": "A constrained 2G EDGE network",
                },
                {
                    "name": "3G network",
                    "identifier": "SlowNetwork3G",
                    "description": "A constrained 3G network",
                },
                {
                    "name": "High latency network",
                    "identifier": "SlowNetworkHighLatency",
                    "description": "Network with high latency",
                },
            ],
        }
    ]


class SlowNetworkTests(unittest.TestCase):
    def test_android_console_status_is_normalized_to_kbps_and_ms(self):
        status = """Current network status:
download speed: 14400 bits/s
upload speed: 14.4 kbits/s
minimum latency: 150 ms
maximum latency: 550 ms
"""
        with mock.patch.object(emulator_network, "network_status", return_value=status):
            observed = emulator_network.network_shape("emulator-5554")

        self.assertEqual(observed, EmulatorNetworkShape(14.4, 14.4, 150, 550))

    def test_ios_profiles_are_discovered_by_meaning(self):
        conditions = network_conditions()

        self.assertEqual(
            ios_network_profile(conditions, "slow_3g"),
            ("SlowNetworkCondition", "SlowNetwork3G"),
        )
        self.assertIsNone(active_ios_condition(conditions))

    def test_ios_enables_and_reads_back_the_discovered_profile(self):
        selected = "SlowNetwork3G"
        driver = SequencedDriver(
            platform=Platform.IOS,
            package=BUNDLE,
            answers={"enableConditionInducer": True},
            sequences={
                "listConditionInducers": [
                    network_conditions(),
                    network_conditions(selected),
                ]
            },
        )

        receipt = controller(platform=Platform.IOS, driver=driver).execute(
            NetworkCapability(mode="slow_3g")
        )

        self.assertTrue(receipt.confirmed)
        self.assertEqual(
            driver.asked("enableConditionInducer"),
            {"conditionID": "SlowNetworkCondition", "profileID": selected},
        )

    def test_android_sets_both_shape_dimensions_and_confirms_them(self):
        shape = EmulatorNetworkShape(14.4, 14.4, 150, 550)
        with mock.patch(
            "yewo.agent.device.resilience.set_network_shape"
        ) as set_shape, mock.patch(
            "yewo.agent.device.resilience.network_shape",
            return_value=shape,
        ):
            receipt = controller().execute(NetworkCapability(mode="slow_2g"))

        set_shape.assert_called_once_with("emulator-5554", "14.4:14.4", "150:550")
        self.assertTrue(receipt.confirmed)

    def test_ios_restores_no_active_condition_after_an_interruption(self):
        active = "SlowNetwork3G"
        driver = SequencedDriver(
            platform=Platform.IOS,
            package=BUNDLE,
            answers={"disableConditionInducer": True},
            sequences={
                "listConditionInducers": [
                    network_conditions(),
                    network_conditions(active),
                    network_conditions(),
                ]
            },
        )
        device = session(platform=Platform.IOS, driver=driver)
        device.app_id = BUNDLE

        with self.assertRaises(KeyboardInterrupt), restoration.preserved(
            device, capabilities=[NetworkCapability(mode="slow_3g")]
        ):
            raise KeyboardInterrupt()

        self.assertIsNotNone(driver.asked("disableConditionInducer"))


if __name__ == "__main__":
    unittest.main()
