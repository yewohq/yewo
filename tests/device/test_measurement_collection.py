"""Platform measurement sources produce bounded, scrub-ready samples."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from yewo.agent.device.launch import (
    SetupMode,
    measure_android_launch_ttid,
    parse_android_launch_ttid,
)
from yewo.agent.device.platform import Platform
from yewo.errors import MeasurementUnavailable
from yewo.metrics import MeasurementRelation, Metric
from yewo.testing.judge.capabilities import CapabilityTarget
from yewo.testing.measurement_collection import (
    ScenarioMeasurementCollector,
    accessibility_values,
    collector_refusal,
    janky_frame_percent,
)


class AndroidLaunchMeasurementTests(unittest.TestCase):
    def test_activity_manager_total_time_is_ttid(self):
        output = "Status: ok\nLaunchState: COLD\nTotalTime: 612\nWaitTime: 640\nComplete\n"

        self.assertEqual(parse_android_launch_ttid(output), 612)

    def test_missing_completed_timing_is_not_a_zero_sample(self):
        with self.assertRaises(MeasurementUnavailable):
            parse_android_launch_ttid("Status: timeout\nWaitTime: 640\n")

    def test_collector_force_stops_without_clearing_app_data(self):
        adb = Mock()
        adb.resolve_launch_activity.return_value = "com.example.MainActivity"
        adb.run_adb_command.return_value = SimpleNamespace(
            stdout="Status: ok\nTotalTime: 321\n"
        )
        device = SimpleNamespace(
            package_id="com.example",
            app_id="com.example",
            adb_control=adb,
            serial="emulator-5554",
            device_name="Pixel_8",
            app_path="app.apk",
        )

        with patch("yewo.agent.device.launch.await_first_paint"):
            timing = measure_android_launch_ttid(device)

        command = adb.run_adb_command.call_args.args[0]
        self.assertIn("-S", command)
        self.assertIn("-W", command)
        self.assertNotIn("pm", command)
        self.assertEqual(timing.duration_ms, 321)
        self.assertEqual(timing.launch_type, "cold_process")

    def test_frame_diagnostic_reads_the_worst_reported_window(self):
        output = "Janky frames: 2 (1.25%)\nJanky frames: 5 (3.50%)\n"

        self.assertEqual(janky_frame_percent(output), 3.5)


class AccessibilityMeasurementTests(unittest.TestCase):
    def test_audit_issues_produce_exact_count_and_minimum(self):
        payload = [
            {
                "auditType": "XCUIAccessibilityAuditTypeHitRegion",
                "elementAttributes": {"rect": {"width": 30, "height": 50}},
                "detailedDescription": "sensitive app content is not retained",
            },
            {
                "auditType": "XCUIAccessibilityAuditTypeSufficientElementDescription",
                "elementAttributes": {"type": "Button"},
            },
        ]

        values = accessibility_values(
            payload,
            [Metric.MIN_TOUCH_TARGET, Metric.UNLABELLED_ACTIONABLE_COUNT],
        )

        self.assertEqual(
            values[Metric.MIN_TOUCH_TARGET],
            (30.0, MeasurementRelation.EXACT),
        )
        self.assertEqual(
            values[Metric.UNLABELLED_ACTIONABLE_COUNT],
            (1.0, MeasurementRelation.EXACT),
        )

    def test_no_hit_region_issue_proves_only_the_platform_floor(self):
        values = accessibility_values([], [Metric.MIN_TOUCH_TARGET])

        self.assertEqual(
            values[Metric.MIN_TOUCH_TARGET],
            (44.0, MeasurementRelation.AT_LEAST),
        )

    def test_hit_region_without_a_rectangle_produces_no_invented_value(self):
        with self.assertRaises(MeasurementUnavailable):
            accessibility_values(
                [{"auditType": "XCUIAccessibilityAuditTypeHitRegion"}],
                [Metric.MIN_TOUCH_TARGET],
            )

    def test_one_audit_runs_per_observed_screen(self):
        driver = SimpleNamespace(
            execute_script=Mock(
                return_value=[
                    {
                        "auditType": "XCUIAccessibilityAuditTypeSufficientElementDescription",
                        "elementAttributes": {"type": "Button"},
                    }
                ]
            )
        )
        device = SimpleNamespace(
            platform=Platform.IOS,
            os_version="18.2",
            model="iPhone",
            device_name="iPhone 17 Pro",
            driver=driver,
        )
        collector = ScenarioMeasurementCollector(
            device,
            CapabilityTarget(platform=Platform.IOS, emulator=True, release=(18, 2)),
            "scenario",
            setup=SetupMode.CLEAR_DATA,
            required_metrics=[Metric.UNLABELLED_ACTIONABLE_COUNT],
        )

        collector.observe_screen("screen-a", 1)
        collector.observe_screen("screen-a", 2)

        driver.execute_script.assert_called_once()
        self.assertEqual(collector.samples[0].value, 1)
        self.assertEqual(collector.samples[0].screen, "screen-a")
        self.assertNotIn("detailedDescription", collector.samples[0].model_dump())


class CollectorSupportTests(unittest.TestCase):
    def test_ios_audit_needs_ios_17(self):
        reason = collector_refusal(
            Metric.MIN_TOUCH_TARGET,
            CapabilityTarget(platform=Platform.IOS, emulator=True, release=(16, 4)),
            setup=SetupMode.CLEAR_DATA,
        )

        self.assertIn("iOS 17", reason)

    def test_launch_measurement_cannot_break_preserve(self):
        reason = collector_refusal(
            Metric.APP_LAUNCH_TTID_MS,
            CapabilityTarget(platform=Platform.ANDROID, emulator=True),
            setup=SetupMode.PRESERVE,
        )

        self.assertIn("preserve", reason)

    def test_appium_wall_clock_is_not_accepted_as_ios_ttid(self):
        reason = collector_refusal(
            Metric.APP_LAUNCH_TTID_MS,
            CapabilityTarget(platform=Platform.IOS, emulator=True, release=(18,)),
            setup=SetupMode.CLEAR_DATA,
        )

        self.assertIn("first-frame", reason)


if __name__ == "__main__":
    unittest.main()
