"""Taking the device's own log, on each of the four kinds of target.

A run that fails on the device is the run whose log is wanted, so the capture
must not fail with it. What it must never do is leave an empty file, because
zero bytes reads as a silent device rather than as a target nothing could ask.
"""

import logging
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from fakes import LOG_EPOCH_MS, LoggingDriver, log_entries

from yewo.agent.artifacts.layout import RunLayout
from yewo.agent.device import device_logs
from yewo.agent.device.platform import Platform

SESSION = "s1"


def _moment(offset_seconds):
    """The host receipt time `log_entries` stamps, as the capture renders it."""
    stamped = datetime.fromtimestamp((LOG_EPOCH_MS + offset_seconds * 1000) / 1000)
    return stamped.isoformat(sep=" ", timespec="milliseconds")


class CaptureCase(unittest.TestCase):
    """Runs one capture against a target, under its own artifact tree."""

    def capture(self, driver, platform):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.layout = RunLayout.under(Path(self.directory.name), SESSION)
        self.layout.create()
        return device_logs.capture(driver, platform, self.layout)

    def files(self):
        return sorted(path.name for path in self.layout.logs.iterdir())

    def by_name(self, captures):
        return {one.driver_name: one for one in captures}


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


class RecoveryTests(CaptureCase):
    """Session recovery opens a second session on the same session id."""

    def test_a_second_capture_does_not_overwrite_the_first(self):
        """The failed session's log is the one worth reading, and it is
        written first. The healthy replacement closes after it."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.layout = RunLayout.under(Path(self.directory.name), SESSION)
        self.layout.create()

        for message in ("the session that died", "the session that replaced it"):
            driver = LoggingDriver(offers=("logcat",), logs={"logcat": log_entries(message)})
            device_logs.capture(driver, Platform.ANDROID, self.layout)

        self.assertEqual(
            self.files(),
            [f"device_logcat_{SESSION}_001.log", f"device_logcat_{SESSION}_002.log"],
        )
        first = self.layout.device_log("logcat", 1).read_text()
        self.assertIn("the session that died", first)


class SegmentPairingTests(CaptureCase):
    """One segment number per capture, so a number names a session."""

    def _tree(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.layout = RunLayout.under(Path(self.directory.name), SESSION)
        self.layout.create()

    def _ios(self, **logs):
        driver = LoggingDriver(offers=("syslog", "crashlog"), logs=logs)
        return device_logs.capture(driver, Platform.IOS, self.layout)

    def test_a_crashlog_is_numbered_with_the_syslog_of_its_own_session(self):
        """A crash is read against the syslog beside it. Numbering each log
        on its own pairs the second session's crash with the first
        session's syslog whenever one source was silent."""
        self._tree()
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
        self._tree()
        self._ios(crashlog=log_entries("only a crash"))
        self._ios(syslog=log_entries("only a log"))

        self.assertEqual(
            self.files(),
            [f"device_crashlog_{SESSION}_001.log", f"device_syslog_{SESSION}_002.log"],
        )


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

        captures = device_logs.capture(driver, Platform.ANDROID, self.layout)

        self.assertFalse(captures[0].is_written)
        self.assertIn("the write failed", captures[0].reason)


class TruncationTests(CaptureCase):
    """Appium discards from the front, so a full buffer has lost the start."""

    def test_a_full_buffer_is_reported_as_having_dropped_lines(self):
        entries = log_entries(*(f"line {n}" for n in range(device_logs.BUFFERED_LINES)))
        driver = LoggingDriver(offers=("logcat",), logs={"logcat": entries})

        captures = self.capture(driver, Platform.ANDROID)

        self.assertTrue(captures[0].truncated)
        self.assertIn("the oldest lines were dropped", captures[0].describe())

    def test_a_buffer_under_the_cap_is_not(self):
        entries = log_entries(*(f"line {n}" for n in range(device_logs.BUFFERED_LINES - 1)))
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
