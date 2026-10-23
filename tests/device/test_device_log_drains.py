"""Draining a session's log repeatedly, and numbering what each drain wrote.

Appium holds 10,000 lines and evicts the oldest whether or not anybody
reads, so a session drained once keeps its tail and loses its start. Reads
are incremental, so the drains append rather than repeat. Session recovery
reopens on the same id, so the numbering is what tells two sessions apart.
"""

import logging
import tempfile
import threading
import time
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


class TickerTests(CaptureCase):
    """Draining on a timer, because a step is too coarse for the rate.

    A step takes about twelve seconds. The buffer has been seen to take
    1,100 lines in one second, which fills it in nine, so waiting for a
    step boundary can lose a burst that happened inside the step.
    """

    def _ticking(self, driver, interval=0.01):
        recorder = device_logs.DeviceLogRecorder(
            self.layout, Platform.ANDROID, interval=interval
        )
        self.addCleanup(recorder.finish, driver)
        recorder.start(driver)
        return recorder

    def _until(self, condition, limit=2.0):
        deadline = time.monotonic() + limit
        while time.monotonic() < deadline:
            if condition():
                return True
            time.sleep(0.01)
        return False

    def test_lines_are_taken_without_anyone_asking(self):
        driver = LoggingDriver(offers=("logcat",), logs={"logcat": log_entries("mid-step")})
        self._ticking(driver)

        self.assertTrue(
            self._until(lambda: self.layout.device_log("logcat", 1).exists()),
            "the timer never drained",
        )

    def test_the_timer_stops_when_the_session_is_finished(self):
        driver = LoggingDriver(offers=("logcat",), logs={"logcat": log_entries("a")})
        recorder = self._ticking(driver)
        self.assertTrue(self._until(lambda: driver.asked))

        recorder.finish(driver)
        settled = len(driver.asked)
        self.assertFalse(any(t.name == "yewo-device-logs" for t in threading.enumerate()))
        self.assertEqual(len(driver.asked), settled, "the timer kept reading after finish")

    def test_no_thread_outlives_the_session(self):
        driver = LoggingDriver(offers=("logcat",), logs={"logcat": log_entries("a")})
        recorder = self._ticking(driver)
        self.assertTrue(self._until(lambda: driver.asked))

        recorder.finish(driver)

        self.assertNotIn(
            "yewo-device-logs", [thread.name for thread in threading.enumerate()]
        )

    def test_a_dead_session_stops_the_timer_rather_than_failing_forever(self):
        """Polling a gone session only repeats one failure into the run log."""
        driver = LoggingDriver(offers=("logcat",), logs={"logcat": log_entries("a")})
        self._ticking(driver)
        self.assertTrue(self._until(lambda: driver.asked))

        driver.quit()
        at_death = len(driver.asked)
        self.assertTrue(
            self._until(
                lambda: not any(
                    thread.name == "yewo-device-logs"
                    for thread in threading.enumerate()
                )
            )
        )

        self.assertLessEqual(
            len(driver.asked) - at_death,
            device_logs.CONSECUTIVE_FAILURES_BEFORE_DISABLING_SOURCE,
            "the timer kept polling a dead session",
        )

    def test_a_timer_that_stopped_says_so(self):
        driver = LoggingDriver(offers=("logcat",), logs={"logcat": log_entries("a")})
        self._ticking(driver)
        self.assertTrue(self._until(lambda: driver.asked))

        with self.assertLogs("yewo.agent.device.device_logs", logging.WARNING) as logged:
            driver.quit()
            self.assertTrue(
                self._until(lambda: any("draining stopped" in line for line in logged.output))
            )

    def test_a_boundary_drain_and_a_tick_never_read_at_once(self):
        """Two reads in flight would append one session's lines out of order.

        The driver holds each read open, so a lock that was not taken shows
        up as two callers inside it at the same time.
        """

        class Colliding(LoggingDriver):
            def __init__(self, **named):
                super().__init__(**named)
                self.inside = 0
                self.most_inside = 0
                self._counting = threading.Lock()

            def get_log(self, log_type):
                with self._counting:
                    self.inside += 1
                    self.most_inside = max(self.most_inside, self.inside)
                time.sleep(0.02)
                try:
                    return super().get_log(log_type)
                finally:
                    with self._counting:
                        self.inside -= 1

        driver = Colliding(offers=("logcat",), logs={"logcat": log_entries("a")})
        recorder = self._ticking(driver)

        for number in range(3):
            driver.arrive("logcat", f"line {number}")
            recorder.drain(driver)
        recorder.finish(driver)

        self.assertEqual(driver.most_inside, 1, "two drains read the log at once")
        written = [
            line.split(" INFO ")[-1]
            for line in self.layout.device_log("logcat", 1).read_text().splitlines()
        ]
        self.assertEqual(written, ["a"] + [f"line {n}" for n in range(3)])

    def test_a_second_start_does_not_raise_a_second_timer(self):
        """Nothing calls it twice today, and two timers on one session would
        read the same buffer against each other."""
        driver = LoggingDriver(offers=("logcat",), logs={"logcat": log_entries("a")})
        recorder = self._ticking(driver)

        recorder.start(driver)

        running = [t for t in threading.enumerate() if t.name == "yewo-device-logs"]
        self.assertEqual(len(running), 1)

    def test_a_platform_with_no_logs_runs_no_timer(self):
        recorder = device_logs.DeviceLogRecorder(self.layout, None, interval=0.01)
        self.addCleanup(recorder.finish, LoggingDriver())

        recorder.start(LoggingDriver())

        self.assertNotIn(
            "yewo-device-logs", [t.name for t in threading.enumerate()]
        )
