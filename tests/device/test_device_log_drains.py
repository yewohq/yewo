"""Draining a session's log repeatedly, and numbering what each drain wrote.

Appium holds 10,000 lines and evicts the oldest whether or not anybody
reads, so a session drained once keeps its tail and loses its start. Reads
are incremental, so the drains append rather than repeat. Session recovery
reopens on the same id, so the numbering is what tells two sessions apart.
"""

import logging
import tempfile
import unittest

from device_log_case import CaptureCase
from fakes import LoggingDriver, log_entries

from yewo.agent.device import device_logs
from yewo.agent.device.platform import Platform

SESSION = "s1"


class DrainingTests(CaptureCase):
    """Draining at every scenario boundary, not once at the end.

    Appium holds 10,000 lines and evicts the oldest whether or not anybody
    reads. A run drained only at the end keeps its tail and loses its
    beginning, which is usually where a problem started.
    """

    def test_every_drain_is_appended_in_order(self):
        driver = LoggingDriver(offers=("logcat",), logs={"logcat": log_entries("scenario one")})
        recorder = self.recorder(Platform.ANDROID)

        recorder.drain(driver)
        driver.arrive("logcat", "scenario two")
        recorder.drain(driver)
        driver.arrive("logcat", "scenario three")
        captures = recorder.finish(driver)

        written = self.layout.device_log("logcat", 1).read_text().splitlines()
        self.assertEqual(
            [line.split(" INFO ")[-1] for line in written],
            ["scenario one", "scenario two", "scenario three"],
        )
        self.assertEqual(captures[0].line_count, 3)

    def test_nothing_is_written_twice(self):
        """Appium's read drains, so a second drain holds only what is new.
        A capture that re-read the buffer would duplicate every line."""
        driver = LoggingDriver(offers=("logcat",), logs={"logcat": log_entries("once")})
        recorder = self.recorder(Platform.ANDROID)

        recorder.drain(driver)
        recorder.drain(driver)
        recorder.finish(driver)

        self.assertEqual(self.layout.device_log("logcat", 1).read_text().count("once"), 1)

    def test_a_quiet_boundary_says_nothing(self):
        """Most boundaries find nothing new. A warning at each one would
        bury the absence that matters."""
        driver = LoggingDriver(offers=("logcat",), logs={"logcat": log_entries("a")})
        recorder = self.recorder(Platform.ANDROID)
        recorder.drain(driver)

        with self.assertNoLogs("yewo.agent.device.device_logs", level=logging.WARNING):
            recorder.drain(driver)
            recorder.drain(driver)

    def test_a_log_that_arrived_before_a_quiet_end_is_still_reported(self):
        """The last drain finds nothing, and the session still captured."""
        driver = LoggingDriver(offers=("logcat",), logs={"logcat": log_entries("a", "b")})
        recorder = self.recorder(Platform.ANDROID)
        recorder.drain(driver)

        captures = recorder.finish(driver)

        self.assertTrue(captures[0].is_written)
        self.assertEqual(captures[0].line_count, 2)
        self.assertEqual(captures[0].reason, "")

    def test_the_driver_is_asked_for_its_log_types_once(self):
        """They do not change for the life of a session, and asking is a
        round trip to the server at every boundary."""
        driver = LoggingDriver(offers=("logcat",), logs={"logcat": log_entries("a")})
        recorder = self.recorder(Platform.ANDROID)

        recorder.drain(driver)
        recorder.drain(driver)
        recorder.finish(driver)

        self.assertEqual(driver.type_reads, 1)

    def test_a_drain_that_fills_the_buffer_reports_lines_lost(self):
        """Evicted between drains, which means drain more often."""
        full = log_entries(*(f"line {n}" for n in range(device_logs.BUFFERED_LINES)))
        driver = LoggingDriver(offers=("logcat",), logs={"logcat": full})
        recorder = self.recorder(Platform.ANDROID)

        recorder.drain(driver)
        driver.arrive("logcat", "after the flood")
        captures = recorder.finish(driver)

        self.assertTrue(captures[0].truncated)
        self.assertEqual(captures[0].line_count, device_logs.BUFFERED_LINES + 1)


class RecoveryTests(CaptureCase):
    """Session recovery opens a second session on the same session id."""

    def test_a_second_capture_does_not_overwrite_the_first(self):
        """The failed session's log is the one worth reading, and it is
        written first. The healthy replacement closes after it."""
        for message in ("the session that died", "the session that replaced it"):
            driver = LoggingDriver(offers=("logcat",), logs={"logcat": log_entries(message)})
            self.recorder(Platform.ANDROID).finish(driver)

        self.assertEqual(
            self.files(),
            [f"device_logcat_{SESSION}_001.log", f"device_logcat_{SESSION}_002.log"],
        )
        first = self.layout.device_log("logcat", 1).read_text()
        self.assertIn("the session that died", first)


class SegmentPairingTests(CaptureCase):
    """One segment number per capture, so a number names a session."""

    def _ios(self, **logs):
        driver = LoggingDriver(offers=("syslog", "crashlog"), logs=logs)
        return self.recorder(Platform.IOS).finish(driver)

    def test_a_crashlog_is_numbered_with_the_syslog_of_its_own_session(self):
        """A crash is read against the syslog beside it. Numbering each log
        on its own pairs the second session's crash with the first
        session's syslog whenever one source was silent."""
        self._ios(syslog=log_entries("session one"))
        self._ios(
            syslog=log_entries("session two"),
            crashlog=log_entries("the crash in session two"),
        )

        self.assertIn("session two", self.layout.device_log("syslog", 2).read_text())
        self.assertIn(
            "the crash in session two", self.layout.device_log("crashlog", 2).read_text()
        )
        self.assertFalse(self.layout.device_log("crashlog", 1).exists())

    def test_a_segment_is_skipped_rather_than_reused_by_either_log(self):
        """The number advances for both sources together, so a reader can
        take two files with one number as one session."""
        self._ios(crashlog=log_entries("only a crash"))
        self._ios(syslog=log_entries("only a log"))

        self.assertEqual(
            self.files(),
            [f"device_crashlog_{SESSION}_001.log", f"device_syslog_{SESSION}_002.log"],
        )


class ExhaustionTests(CaptureCase):
    """Every segment taken, which bounds the search rather than hanging."""

    def test_a_session_with_no_free_segment_is_refused_with_a_reason(self):
        for segment in range(1, device_logs.MAX_SEGMENTS + 1):
            self.layout.device_log("logcat", segment).touch()
        driver = LoggingDriver(offers=("logcat",), logs={"logcat": log_entries("a")})

        capture = self.recorder(Platform.ANDROID).finish(driver)[0]

        self.assertFalse(capture.is_written)
        self.assertIn(str(device_logs.MAX_SEGMENTS), capture.reason)

    def test_a_refused_session_asks_the_driver_for_nothing(self):
        for segment in range(1, device_logs.MAX_SEGMENTS + 1):
            self.layout.device_log("logcat", segment).touch()
        driver = LoggingDriver(offers=("logcat",), logs={"logcat": log_entries("a")})

        self.recorder(Platform.ANDROID).finish(driver)

        self.assertEqual(driver.asked, [])


class RunnerWiringTests(unittest.TestCase):
    """The runner asking for a drain, which is what makes draining happen.

    The recorder above is held to its own contract. These hold the runner to
    using it, so a wiring change fails here rather than on a device.
    """

    def _asked(self, scenario_count):
        from fakes import suite_run, walk

        from yewo.testing.models import GeneratedTestCase

        scenarios = [
            GeneratedTestCase(
                test_id=f"TS_{n}",
                scenario_name=f"Scenario {n}",
                cell="functional.happy_path.0",
                category="functional",
                scenario_type="happy_path",
                priority="P0",
                steps=walk("Open the app."),
                expected_outcomes=["ok"],
            )
            for n in range(1, scenario_count + 1)
        ]
        asked = []
        with tempfile.TemporaryDirectory() as directory:
            suite_run(scenarios, directory, session_calls=asked)
        return asked

    def test_the_device_log_is_drained_at_every_scenario_boundary(self):
        asked = self._asked(2)

        self.assertEqual(asked.count("checkpoint_device_logs"), 2)

    def test_the_drain_happens_before_the_session_closes(self):
        """A drain after the close would read a quit driver."""
        asked = self._asked(1)

        self.assertLess(
            asked.index("checkpoint_device_logs"), asked.index("close"), asked
        )
