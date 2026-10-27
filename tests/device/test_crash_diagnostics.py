"""App-scoped crash extraction from Android and iOS device logs."""

import unittest

from yewo.agent.device.crash_diagnostics import extract_crash_diagnostic


class CrashDiagnosticsExtractionTests(unittest.TestCase):
    def test_android_runtime_fatal_exception_is_attributed_by_package(self):
        log_lines = [
            "10-26 13:52:00.123 1000 1000 E AndroidRuntime: FATAL EXCEPTION: main",
            "10-26 13:52:00.124 1000 1000 E AndroidRuntime: Process: com.example.app, PID: 5183",
            "10-26 13:52:00.125 1000 1000 E AndroidRuntime: java.lang.NullPointerException: Submit failed",
            "10-26 13:52:00.126 1000 1000 E AndroidRuntime: \tat com.example.app.MainActivity.onClick(MainActivity.java:42)",
        ]

        diagnostic = extract_crash_diagnostic(
            log_lines, source="logcat", target_package="com.example.app"
        )

        self.assertIsNotNone(diagnostic)
        self.assertEqual(diagnostic.exception_type, "java.lang.NullPointerException")
        self.assertEqual(diagnostic.message, "Submit failed")
        self.assertEqual(len(diagnostic.stack_trace), 1)

    def test_android_runtime_crash_from_another_package_is_ignored(self):
        log_lines = [
            "10-26 13:52:00.123 1234 1234 E AndroidRuntime: FATAL EXCEPTION: main",
            "10-26 13:52:00.124 1234 1234 E AndroidRuntime: Process: com.example.app.helper, PID: 1234",
            "10-26 13:52:00.125 1234 1234 E AndroidRuntime: java.lang.IllegalStateException: background noise",
        ]

        diagnostic = extract_crash_diagnostic(
            log_lines, source="logcat", target_package="com.example.app"
        )

        self.assertIsNone(diagnostic)

    def test_android_subprocess_crash_belongs_to_the_package(self):
        log_lines = [
            "10-26 13:52:00.123 1234 1234 E AndroidRuntime: FATAL EXCEPTION: worker",
            "10-26 13:52:00.124 1234 1234 E AndroidRuntime: Process: com.example.app:worker, PID: 1234",
            "10-26 13:52:00.125 1234 1234 E AndroidRuntime: java.lang.IllegalStateException: worker stopped",
        ]

        diagnostic = extract_crash_diagnostic(
            log_lines, source="logcat", target_package="com.example.app"
        )

        self.assertIsNotNone(diagnostic)

    def test_ios_uncaught_exception_is_attributed_by_process_name(self):
        log_lines = [
            "Oct 26 iPhone Runner[1234]: bundle com.example.app",
            "Oct 26 iPhone Runner[1234]: *** Terminating app due to uncaught exception 'NSInvalidArgumentException', reason: 'Unknown selector'",
        ]

        diagnostic = extract_crash_diagnostic(
            log_lines,
            source="syslog",
            target_app_id="com.example.app",
            target_process_name="Runner",
        )

        self.assertIsNotNone(diagnostic)
        self.assertEqual(diagnostic.exception_type, "NSInvalidArgumentException")
        self.assertEqual(diagnostic.message, "Unknown selector")

    def test_ios_exception_without_the_target_identity_is_ignored(self):
        diagnostic = extract_crash_diagnostic(
            [
                "Other[1234]: *** Terminating app due to uncaught exception "
                "'NSException', reason: 'noise'"
            ],
            source="syslog",
            target_process_name="Runner",
        )

        self.assertIsNone(diagnostic)
