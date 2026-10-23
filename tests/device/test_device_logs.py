"""What each target answers when asked for its own log, and what it refuses.

A run that fails on the device is the run whose log is wanted, so the
capture must not fail with it. What it must never do is leave an empty file
or an ordinary-looking one, because a reader cannot see a gap that nothing
reports.
"""

import logging
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from device_log_case import CaptureCase
from fakes import LOG_EPOCH_MS, LoggingDriver, log_entries

from yewo.agent.artifacts.layout import RunLayout
from yewo.agent.device import device_logs
from yewo.agent.device.platform import Platform

SESSION = "s1"


def _moment(offset_seconds):
    """The host receipt time `log_entries` stamps, as the capture renders it."""
    stamped = datetime.fromtimestamp((LOG_EPOCH_MS + offset_seconds * 1000) / 1000)
    return stamped.isoformat(sep=" ", timespec="milliseconds")


class AndroidTests(CaptureCase):
    """Android answers the same way on an emulator and on a physical phone."""

    def test_logcat_is_written_under_the_session(self):
        driver = LoggingDriver(
            offers=("logcat", "bugreport", "server"),
            logs={"logcat": log_entries("lowmemorykiller: killing com.app", "adbd: error -32")},
        )

        captures = self.capture(driver, Platform.ANDROID)

        self.assertEqual(len(captures), 1)
        written = captures[0]
        self.assertTrue(written.is_written)
        self.assertEqual(written.line_count, 2)
        self.assertEqual(written.path.name, f"device_logcat_{SESSION}_001.log")
        self.assertEqual(
            written.path.read_text().splitlines(),
            [
                f"{_moment(0)} INFO lowmemorykiller: killing com.app",
                f"{_moment(1)} INFO adbd: error -32",
            ],
        )

    def test_only_the_wanted_log_is_asked_for(self):
        """A bugreport takes a minute, so a run does not take one every time."""
        driver = LoggingDriver(
            offers=("logcat", "bugreport", "server"), logs={"logcat": log_entries("a")}
        )

        self.capture(driver, Platform.ANDROID)

        self.assertEqual(driver.asked, ["logcat"])


class IosTests(CaptureCase):
    """A simulator and a physical iPhone do not answer the same way."""

    def test_a_simulator_answers_for_both_of_its_logs(self):
        driver = LoggingDriver(
            offers=("syslog", "crashlog", "server"),
            logs={
                "syslog": log_entries("SpringBoard: launched"),
                "crashlog": log_entries("Exception Type: EXC_BAD_ACCESS"),
            },
        )

        captures = self.capture(driver, Platform.IOS)

        self.assertTrue(all(one.is_written for one in captures))
        self.assertEqual(
            self.files(),
            [f"device_crashlog_{SESSION}_001.log", f"device_syslog_{SESSION}_001.log"],
        )

    def test_a_physical_iphone_that_answers_nothing_is_said_to_have(self):
        """XCUITest lists both logs whatever the target is, and reads a real
        device in-process. So the empty answer is the refusal here, and it
        must not be written out as an empty file."""
        driver = LoggingDriver(
            offers=("syslog", "crashlog"),
            logs={"crashlog": log_entries("Exception Type: X")},
        )

        captures = self.by_name(self.capture(driver, Platform.IOS))

        self.assertFalse(captures["syslog"].is_written)
        self.assertIn("returned nothing", captures["syslog"].reason)
        self.assertTrue(captures["crashlog"].is_written)

    def test_a_target_offering_no_logs_writes_no_file(self):
        self.capture(LoggingDriver(offers=()), Platform.IOS)

        self.assertEqual(self.files(), [])


class AbsenceTests(CaptureCase):
    """Four ways a log can be missing, each said rather than written empty."""

    def test_a_log_the_driver_does_not_list_is_refused_with_what_it_has(self):
        captures = self.capture(LoggingDriver(offers=("server",)), Platform.ANDROID)

        self.assertFalse(captures[0].is_written)
        self.assertIn("It offers: server", captures[0].reason)

    def test_a_log_offered_and_returned_empty_writes_nothing(self):
        """An empty answer is not an empty file. The device may simply have
        said nothing, and a file cannot tell a reader which it was."""
        captures = self.capture(LoggingDriver(offers=("logcat",)), Platform.ANDROID)

        self.assertFalse(captures[0].is_written)
        self.assertIn("returned nothing", captures[0].reason)
        self.assertEqual(self.files(), [])

    def test_a_log_offered_and_then_refused_is_a_read_failure(self):
        """Distinct from a log the driver never listed, because the fault is
        in the read and not in what this target can do."""
        driver = LoggingDriver(offers=("syslog", "crashlog"), failing=("syslog",))

        syslog = self.by_name(self.capture(driver, Platform.IOS))["syslog"]

        self.assertFalse(syslog.is_written)
        self.assertIn("the read failed", syslog.reason)

    def test_a_driver_that_cannot_list_its_logs_refuses_every_one(self):
        captures = self.capture(LoggingDriver(list_fails=True), Platform.IOS)

        self.assertEqual(len(captures), 2)
        self.assertFalse(any(one.is_written for one in captures))

    def test_a_quit_driver_reads_nothing_and_raises_no_further(self):
        """The ordering hazard, from the capture's side. It runs before the
        quit, and a caller that got that wrong gets a reason, not a crash."""
        driver = LoggingDriver(offers=("logcat",), logs={"logcat": log_entries("a")})
        driver.quit()

        captures = self.capture(driver, Platform.ANDROID)

        self.assertFalse(captures[0].is_written)
        self.assertIn("the read failed", captures[0].reason)


class GapTests(CaptureCase):
    """A drain that did not land, reported beside the file it is missing from.

    Reads are incremental. Lines a failed read would have returned can be
    gone for good, so a later drain that succeeds does not make the capture
    whole again. A file that looks ordinary is the one way a reader would
    never know.
    """

    def _flaky(self, losing):
        driver = LoggingDriver(offers=("logcat",), logs={"logcat": log_entries("first")})
        recorder = self.recorder(Platform.ANDROID)
        recorder.drain(driver)
        driver.arrive("logcat", "during the failed drain")
        driver.fail_next(losing=losing)
        recorder.drain(driver)
        driver.arrive("logcat", "last")
        return recorder.finish(driver)[0], driver

    def test_a_failed_middle_drain_is_reported_though_the_file_exists(self):
        capture, _ = self._flaky(losing=False)

        self.assertTrue(capture.is_written)
        self.assertFalse(capture.is_whole)
        self.assertEqual(capture.failed_drains, 1)
        self.assertIn("did not land", capture.describe())

    def test_a_lossy_failed_drain_leaves_a_gap_the_file_cannot_show(self):
        """The server read and the response never arrived, so those lines
        are gone. The count is the only trace left of them."""
        capture, _ = self._flaky(losing=True)

        written = capture.path.read_text()
        self.assertNotIn("during the failed drain", written)
        self.assertEqual(capture.failed_drains, 1)
        self.assertFalse(capture.is_whole)

    def test_a_session_with_no_failures_is_whole(self):
        driver = LoggingDriver(offers=("logcat",), logs={"logcat": log_entries("a")})

        capture = self.recorder(Platform.ANDROID).finish(driver)[0]

        self.assertTrue(capture.is_whole)
        self.assertEqual(capture.reason, "")

    def test_a_failed_drain_is_reported_where_it_happened(self):
        """Which boundary failed is what tells a reader which part of the
        log to distrust, and finish() is too late to say it."""
        driver = LoggingDriver(offers=("logcat",), logs={"logcat": log_entries("a")})
        recorder = self.recorder(Platform.ANDROID)
        recorder.drain(driver)
        driver.fail_next()

        with self.assertLogs("yewo.agent.device.device_logs", logging.WARNING) as logged:
            recorder.drain(driver)

        self.assertTrue(any("drain failed" in line for line in logged.output), logged.output)

    def test_an_incomplete_capture_is_a_warning_at_the_end(self):
        driver = LoggingDriver(offers=("logcat",), logs={"logcat": log_entries("a")})
        recorder = self.recorder(Platform.ANDROID)
        recorder.drain(driver)
        driver.fail_next()
        recorder.drain(driver)

        with self.assertLogs("yewo.agent.device.device_logs", logging.WARNING) as logged:
            recorder.finish(driver)

        self.assertTrue(any("incomplete" in line for line in logged.output), logged.output)


class UnbreakableTests(CaptureCase):
    """A run that failed on the device is the run whose log is wanted, so the
    capture must never be what takes the rest of the teardown down."""

    def test_a_line_holding_an_unpaired_surrogate_is_still_written(self):
        """Real logcat output is bytes. utf-8 refuses a lone surrogate, and
        that refusal is a ValueError rather than an OSError."""
        driver = LoggingDriver(
            offers=("logcat",), logs={"logcat": [{"message": "boom \ud800 here"}]}
        )

        captures = self.capture(driver, Platform.ANDROID)

        self.assertTrue(captures[0].is_written)
        self.assertIn("boom", captures[0].path.read_text())

    def test_an_entry_that_cannot_be_rendered_is_reported_not_raised(self):
        class Unrenderable:
            def __str__(self):
                raise RuntimeError("this entry will not be read")

        driver = LoggingDriver(offers=("logcat",), logs={"logcat": [Unrenderable()]})

        captures = self.capture(driver, Platform.ANDROID)

        self.assertFalse(captures[0].is_written)
        self.assertIn("the write failed", captures[0].reason)

    def test_a_directory_that_is_gone_is_reported_not_raised(self):
        driver = LoggingDriver(offers=("logcat",), logs={"logcat": log_entries("a")})
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.layout = RunLayout.under(Path(self.directory.name), SESSION)

        captures = self.recorder(Platform.ANDROID).finish(driver)

        self.assertFalse(captures[0].is_written)
        self.assertIn("the write failed", captures[0].reason)


class TruncationTests(CaptureCase):
    """A drain at the buffer limit may have arrived after an eviction."""

    def test_a_full_buffer_is_reported_as_possibly_missing_lines(self):
        entries = ["line"] * device_logs.BUFFERED_LINES
        driver = LoggingDriver(offers=("logcat",), logs={"logcat": entries})
        recorder = self.recorder(Platform.ANDROID)

        recorder.drain(driver)
        driver.arrive("logcat", "after the full buffer")
        captures = recorder.finish(driver)

        self.assertTrue(captures[0].truncated)
        self.assertEqual(captures[0].line_count, device_logs.BUFFERED_LINES + 1)
        self.assertIn("older lines may be missing", captures[0].describe())
        self.assertFalse(captures[0].is_whole)

    def test_a_buffer_under_the_cap_is_not(self):
        entries = ["line"] * (device_logs.BUFFERED_LINES - 1)
        driver = LoggingDriver(offers=("logcat",), logs={"logcat": entries})

        captures = self.capture(driver, Platform.ANDROID)

        self.assertFalse(captures[0].truncated)


class RunLogTests(CaptureCase):
    """What a reader is told, in the artifact they open first."""

    def test_a_refusal_is_logged_as_a_warning(self):
        with self.assertLogs("yewo.agent.device.device_logs", level=logging.WARNING) as logged:
            self.capture(LoggingDriver(offers=()), Platform.IOS)

        self.assertTrue(
            all("not captured" in line for line in logged.output), logged.output
        )

    def test_a_written_log_is_not_a_warning(self):
        driver = LoggingDriver(offers=("logcat",), logs={"logcat": log_entries("a")})

        with self.assertLogs("yewo.agent.device.device_logs", level=logging.INFO) as logged:
            self.capture(driver, Platform.ANDROID)

        self.assertFalse(any("WARNING" in line for line in logged.output), logged.output)


class EntryShapeTests(CaptureCase):
    """A driver that answers in an unexpected shape still has lines in it."""

    def test_a_plain_string_entry_is_kept(self):
        driver = LoggingDriver(offers=("logcat",), logs={"logcat": ["a bare line"]})

        captures = self.capture(driver, Platform.ANDROID)

        self.assertEqual(captures[0].path.read_text(), "a bare line\n")

    def test_an_entry_with_no_message_keeps_what_it_does_have(self):
        """Dropping it would renumber every line after it, and a log is read
        against the lines around it."""
        driver = LoggingDriver(
            offers=("logcat",),
            logs={"logcat": [{"level": "INFO"}, {"message": "after"}]},
        )

        captures = self.capture(driver, Platform.ANDROID)

        self.assertEqual(captures[0].line_count, 2)
        self.assertEqual(captures[0].path.read_text().splitlines(), ["INFO", "after"])


if __name__ == "__main__":
    unittest.main()


class TimestampTests(unittest.TestCase):
    """A stamp a reader can line up against Yewo's own log, or nothing."""

    def test_an_epoch_millisecond_value_is_rendered(self):
        line = device_logs._format_line({"timestamp": LOG_EPOCH_MS, "message": "x"})

        self.assertTrue(line.endswith(" x"), line)
        self.assertIn(":", line.split(" x")[0])

    def test_a_value_that_is_not_a_time_is_left_out(self):
        for value in ("not a time", None, True, float("nan"), 10**20):
            with self.subTest(value=value):
                line = device_logs._format_line({"timestamp": value, "message": "x"})

                self.assertEqual(line, "x")
