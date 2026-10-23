"""Timed device-log draining through slow, partial, and failed sessions."""

import threading
import time

from device_log_case import CaptureCase
from fakes import LoggingDriver, log_entries

from yewo.agent.device import device_logs
from yewo.agent.device.platform import Platform


class BlockingDriver(LoggingDriver):
    """A log read held by the test until its lifecycle assertion is complete."""

    def __init__(self, **configuration):
        super().__init__(**configuration)
        self.read_started = threading.Event()
        self.release_read = threading.Event()
        self._has_blocked = False

    def get_log(self, log_type):
        if not self._has_blocked:
            self._has_blocked = True
            self.read_started.set()
            self.release_read.wait(2)
        return super().get_log(log_type)


class LifecycleTests(CaptureCase):
    """The timer must preserve evidence without holding the run open."""

    def _until(self, condition, limit=2.0):
        deadline = time.monotonic() + limit
        while time.monotonic() < deadline:
            if condition():
                return True
            time.sleep(0.01)
        return False

    @staticmethod
    def _timer_is_running():
        return any(thread.name == "yewo-device-logs" for thread in threading.enumerate())

    def test_what_is_buffered_is_taken_before_the_first_wait(self):
        driver = LoggingDriver(offers=("logcat",), logs={"logcat": log_entries("boot")})
        recorder = device_logs.DeviceLogRecorder(
            self.layout, Platform.ANDROID, interval=30
        )
        self.addCleanup(recorder.finish, driver)

        recorder.start(driver)

        self.assertTrue(
            self._until(lambda: self.layout.device_log("logcat", 1).exists()),
            "nothing was taken before the first interval",
        )

    def test_start_returns_before_the_first_read_finishes(self):
        driver = BlockingDriver(
            offers=("logcat",), logs={"logcat": log_entries("boot")}
        )
        recorder = device_logs.DeviceLogRecorder(
            self.layout, Platform.ANDROID, interval=0.01
        )

        began = time.monotonic()
        recorder.start(driver)
        elapsed = time.monotonic() - began

        self.assertLess(elapsed, 1.0, "start() waited for the read")
        self.assertTrue(driver.read_started.wait(1), "the timer never read")
        driver.release_read.set()
        recorder.finish(driver)

    def test_finish_seals_evidence_when_a_read_is_still_in_flight(self):
        driver = BlockingDriver(
            offers=("logcat",), logs={"logcat": log_entries("late")}
        )
        recorder = device_logs.DeviceLogRecorder(
            self.layout, Platform.ANDROID, interval=0.01
        )
        recorder.start(driver)
        self.assertTrue(driver.read_started.wait(1), "the timer never read")

        boundary_started = time.monotonic()
        self.assertFalse(recorder.drain(driver, timeout=0))
        self.assertLess(
            time.monotonic() - boundary_started,
            1.0,
            "a boundary queued behind the stuck timer read",
        )

        began = time.monotonic()
        captures = recorder.finish(driver)
        elapsed = time.monotonic() - began

        self.assertLess(elapsed, 1.0, "finish() waited for the stuck read")
        self.assertIn("still in flight", captures[0].reason)
        driver.release_read.set()
        self.assertTrue(self._until(lambda: not self._timer_is_running()))
        self.assertFalse(self.layout.device_log("logcat", 1).exists())

    def test_a_queued_boundary_observes_that_the_recorder_closed(self):
        driver = BlockingDriver(offers=("logcat",))
        recorder = device_logs.DeviceLogRecorder(
            self.layout, Platform.ANDROID, interval=0.01
        )
        recorder.start(driver)
        self.assertTrue(driver.read_started.wait(1), "the timer never read")
        boundary_result = []
        boundary = threading.Thread(
            target=lambda: boundary_result.append(recorder.drain(driver))
        )
        boundary.start()

        recorder.finish(driver)
        driver.release_read.set()
        boundary.join(timeout=1)

        self.assertEqual(boundary_result, [False])
        self.assertTrue(self._until(lambda: not self._timer_is_running()))

    def test_a_late_read_failure_does_not_change_finalized_evidence(self):
        class FailingAfterRelease(BlockingDriver):
            def get_log(self, log_type):
                if not self._has_blocked:
                    self._has_blocked = True
                    self.read_started.set()
                    self.release_read.wait(2)
                    raise RuntimeError("late failure")
                return super().get_log(log_type)

        driver = FailingAfterRelease(offers=("logcat",))
        recorder = device_logs.DeviceLogRecorder(
            self.layout, Platform.ANDROID, interval=0.01
        )
        recorder.start(driver)
        self.assertTrue(driver.read_started.wait(1), "the timer never read")

        capture = recorder.finish(driver)[0]
        driver.release_read.set()
        self.assertTrue(self._until(lambda: not self._timer_is_running()))

        finalized = recorder.finish(driver)[0]
        self.assertEqual(finalized.failed_drains, capture.failed_drains)

    def test_a_late_render_failure_does_not_change_finalized_evidence(self):
        rendering = threading.Event()
        release_render = threading.Event()

        class LateFailure:
            def __str__(self):
                rendering.set()
                release_render.wait(2)
                raise RuntimeError("late failure")

        driver = LoggingDriver(offers=("logcat",), logs={"logcat": [LateFailure()]})
        recorder = device_logs.DeviceLogRecorder(
            self.layout, Platform.ANDROID, interval=0.01
        )
        recorder.start(driver)
        self.assertTrue(rendering.wait(1), "the timer never rendered the entry")

        capture = recorder.finish(driver)[0]
        release_render.set()
        self.assertTrue(self._until(lambda: not self._timer_is_running()))

        finalized = recorder.finish(driver)[0]
        self.assertEqual(finalized.failed_drains, capture.failed_drains)

    def test_success_resets_the_consecutive_failure_count(self):
        driver = LoggingDriver(offers=("logcat",))
        recorder = self.recorder(Platform.ANDROID)

        for number in range(
            device_logs.CONSECUTIVE_FAILURES_BEFORE_DISABLING_SOURCE
        ):
            driver.fail_next()
            recorder.drain(driver)
            driver.arrive("logcat", f"recovered {number}")
            recorder.drain(driver)

        capture = recorder.finish(driver)[0]
        written = capture.path.read_text()
        self.assertEqual(capture.failed_drains, 3)
        self.assertIn("recovered 2", written)

    def test_a_quiet_read_counts_as_a_success(self):
        """A device is often silent between drains. A read that answers
        nothing still answered, so it clears the run of failures the same
        way a read with lines in it does."""
        driver = LoggingDriver(offers=("logcat",))
        recorder = self.recorder(Platform.ANDROID)

        for _ in range(device_logs.CONSECUTIVE_FAILURES_BEFORE_DISABLING_SOURCE):
            driver.fail_next()
            recorder.drain(driver)
            recorder.drain(driver)

        driver.arrive("logcat", "still listening")
        capture = recorder.finish(driver)[0]

        self.assertTrue(capture.is_written, capture.reason)
        self.assertIn("still listening", capture.path.read_text())

    def test_one_disabled_log_does_not_disable_a_healthy_one(self):
        driver = LoggingDriver(
            offers=("syslog", "crashlog"),
            failing=("crashlog",),
            logs={"syslog": log_entries("alive")},
        )
        recorder = device_logs.DeviceLogRecorder(self.layout, Platform.IOS, interval=0.02)
        self.addCleanup(recorder.finish, driver)
        recorder.start(driver)
        self.assertTrue(
            self._until(
                lambda: driver.asked.count("crashlog")
                == device_logs.CONSECUTIVE_FAILURES_BEFORE_DISABLING_SOURCE
            )
        )

        driver.arrive("syslog", "after the crashlog gave out")

        self.assertTrue(
            self._until(
                lambda: "after the crashlog gave out"
                in self.layout.device_log("syslog", 1).read_text()
            ),
            "the healthy log stopped with the broken one",
        )
        self.assertEqual(
            driver.asked.count("crashlog"),
            device_logs.CONSECUTIVE_FAILURES_BEFORE_DISABLING_SOURCE,
        )

    def test_a_driver_that_will_not_list_its_logs_stops_asking(self):
        driver = LoggingDriver(offers=("logcat",), list_fails=True)
        recorder = device_logs.DeviceLogRecorder(
            self.layout, Platform.ANDROID, interval=0.02
        )
        self.addCleanup(recorder.finish, driver)

        recorder.start(driver)

        self.assertTrue(self._until(lambda: not self._timer_is_running()))
        self.assertEqual(
            driver.type_reads,
            device_logs.CONSECUTIVE_FAILURES_BEFORE_DISABLING_SOURCE,
        )
