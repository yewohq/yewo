"""The device's own log, taken through the driver that holds the session.

Yewo's log says what Yewo asked for. The Appium log says what the server did
with it. Neither can see inside the device, so a run that ends
`ERROR (device)` cannot say what the device was doing.

`driver.get_log` is the seam, because both drivers implement it and `adb`
does not reach iOS.

Two properties of that buffer shape everything here. It holds 10,000 lines
and evicts the oldest whether or not anybody reads, so a session drained
once keeps its tail and loses its start. And a read returns only what
arrived since the last one, so drains append rather than repeat.

A log nothing could ask for records a reason instead of an empty file. An
empty file says the device was silent, which is a different answer.
"""

import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Protocol, Sequence

from yewo.agent.artifacts.layout import RunLayout
from yewo.agent.device.platform import Platform

logger = logging.getLogger(__name__)

BUFFERED_LINES = 10000
"""Appium's buffer size, per log. A drain this full reached a buffer at its
limit, so older lines may already have been evicted."""

DRAIN_INTERVAL_SECONDS = 5.0
"""How often to drain while a scenario runs.

A step takes about twelve seconds, so a step boundary is too coarse. The
buffer has been seen to take 1,100 lines in one second, which fills it in
nine. Five seconds holds about half the buffer at that rate, and three
percent of it at the ordinary one.
"""

CONSECUTIVE_FAILURES_BEFORE_DISABLING_SOURCE = 3
"""When to give up on one log. A source that refuses this many reads in a
row is gone, and polling it only repeats one failure into the run log.

Counted per source. One iOS log can fail while the other answers, and
disabling the healthy one would lose the log somebody will read.
"""

MAX_SEGMENTS = 999
"""The highest segment a session's logs are numbered to. Three digits,
matching the filename, and the bound on the search for a free one."""


@dataclass(frozen=True)
class LogSource:
    """One log worth asking a platform for, named as `get_log` takes it."""

    log_type: str
    file_stem: str


LOG_SOURCES_BY_PLATFORM: dict[Platform, tuple[LogSource, ...]] = {
    Platform.ANDROID: (LogSource("logcat", "logcat"),),
    Platform.IOS: (LogSource("syslog", "syslog"), LogSource("crashlog", "crashlog")),
}
"""What to ask each platform for. Whether a target can answer is left to
`log_types`, which is the open session answering it.

XCUITest lists both of its logs whatever the target is, and reads a physical
device in process over RemoteXPC. So on iOS a silent log is a log that
answered nothing, never a log the target could not be asked for.
"""


@dataclass(frozen=True)
class LogCapture:
    """What one log came to over a session.

    `reason` says why the capture is less than whole, whether that means no
    file at all or a file with gaps in it. A written log is still evidence
    about what it holds, so a gap is reported beside it rather than instead
    of it.
    """

    log_type: str
    path: Optional[Path] = None
    line_count: int = 0
    truncated: bool = False
    failed_drains: int = 0
    reason: str = ""

    @property
    def is_written(self) -> bool:
        return self.path is not None

    @property
    def is_whole(self) -> bool:
        """Whether this log holds everything the session produced."""
        return self.is_written and not self.reason

    def describe(self) -> str:
        if not self.is_written:
            return f"{self.log_type}: not captured, {self.reason}"
        if self.reason:
            return f"{self.log_type}: {self.line_count} lines, incomplete. {self.reason}"
        return f"{self.log_type}: {self.line_count} lines"


class LogReadingDriver(Protocol):
    """The part of a driver this module uses."""

    @property
    def log_types(self) -> Sequence[str]: ...

    def get_log(self, log_type: str) -> Sequence[object]: ...


@dataclass
class _SourceState:
    """What one log has come to so far, across this session's drains.

    A failed drain is counted whether or not a file already exists. Reads
    are incremental, so the lines a failed read would have returned can be
    gone for good. A later drain that succeeds does not bring them back.
    """

    path: Optional[Path] = None
    line_count: int = 0
    truncated: bool = False
    failed_drains: int = 0
    consecutive_failures: int = 0
    is_polling_disabled: bool = False
    last_failure: str = ""
    unwritten_reason: str = "this session was never drained"


def _resolve_free_segment(
    layout: RunLayout, sources: Sequence[LogSource]
) -> Optional[int]:
    """The number this session's logs are written under.

    Session recovery opens a new session on the same id. One filename would
    let the healthy replacement overwrite the failed session's log.

    One number covers every source. A reader takes a crashlog together with
    the syslog beside it, and numbering each alone would pair a crash with
    another session's log.

    None when every segment is taken. Bounded, because a search that cannot
    end would hang the run instead of reporting.
    """
    for segment in range(1, MAX_SEGMENTS + 1):
        if not any(layout.device_log(source.file_stem, segment).exists() for source in sources):
            return segment
    return None


def _format_timestamp(value: object) -> str:
    """The host's receipt time, on the clock Yewo's own log runs on.

    The device stamps its own clock inside the line, and the two drift. Both
    are kept, because one orders the log against Yewo's steps and the other
    against the device's.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return ""
    try:
        return datetime.fromtimestamp(value / 1000).isoformat(sep=" ", timespec="milliseconds")
    except (OverflowError, OSError, ValueError):
        return ""


def _format_line(entry: object) -> str:
    """One line, from an entry carrying a timestamp, a level and a message.

    A driver that answers in some other shape still has the line in it, so
    the entry is rendered whole rather than dropped.
    """
    if not isinstance(entry, dict):
        return str(entry)
    parts = (
        _format_timestamp(entry.get("timestamp")),
        str(entry.get("level") or ""),
        str(entry.get("message") or ""),
    )
    return " ".join(part for part in parts if part)


def _gaps_in(state: _SourceState) -> str:
    """Why a written log may be missing lines, or empty when it is whole."""
    gaps = []
    if state.failed_drains:
        drains = "drain" if state.failed_drains == 1 else "drains"
        gaps.append(
            f"{state.failed_drains} {drains} did not land, the last with "
            f"{state.last_failure}"
        )
    if state.truncated:
        gaps.append(
            f"a drain reached Appium's {BUFFERED_LINES}-line buffer limit, "
            "so older lines may be missing"
        )
    return ". ".join(gaps)


class DeviceLogRecorder:
    """Drains one session's device logs into one file each.

    Built per session. Its segment number is what tells this session's logs
    from those of the session that replaced it.

    No method raises. A device log is wanted most on the run that failed on
    the device, and a capture that failed with it would take the evidence.
    """

    def __init__(
        self,
        layout: RunLayout,
        platform: Platform,
        interval: float = DRAIN_INTERVAL_SECONDS,
    ):
        self._layout = layout
        self._sources = LOG_SOURCES_BY_PLATFORM.get(platform, ())
        self._segment = _resolve_free_segment(layout, self._sources)
        self._state: Dict[str, _SourceState] = {
            source.log_type: _SourceState() for source in self._sources
        }
        self._available: Optional[frozenset[str]] = None
        self._interval = interval
        # One drain at a time. The ticker and the scenario boundary both
        # call it, and two reads interleaved would write out of order.
        self._draining = threading.Lock()
        self._stop = threading.Event()
        self._closed = threading.Event()
        self._ticker: Optional[threading.Thread] = None

    def start(self, driver: LogReadingDriver) -> None:
        """Takes what is already buffered, then drains on a timer.

        Returns at once. The first read happens on the timer thread
        before it waits. So a session several seconds old gives up its
        earliest lines straight away. A read that hangs cannot hold up
        the session that asked for it.

        A daemon thread, so a read still in flight at interpreter exit
        cannot hold the process open.
        """
        if self._ticker is not None or self._closed.is_set() or not self._sources:
            return
        self._ticker = threading.Thread(
            target=self._tick, args=(driver,), name="yewo-device-logs", daemon=True
        )
        self._ticker.start()

    def _tick(self, driver: LogReadingDriver) -> None:
        while True:
            # The first drain happens before any wait. When finish wakes the
            # worker, the same branch performs the final drain before exit.
            self.drain(driver)
            if all(state.is_polling_disabled for state in self._state.values()):
                logger.warning("Device log draining stopped: every source was disabled.")
                return
            if self._stop.wait(self._interval):
                self.drain(driver)
                return

    def drain(self, driver: LogReadingDriver, timeout: Optional[float] = None) -> bool:
        """Appends what arrived since the last drain.

        Called on the timer, and at each scenario boundary. Finding nothing
        new is the ordinary case between two drains, and reports nothing.

        Returns whether the drain ran. With a timeout it gives up rather
        than queue behind a read that is still in flight.
        """
        if self._closed.is_set():
            return False
        if not self._draining.acquire(timeout=-1 if timeout is None else timeout):
            return False
        try:
            if self._closed.is_set():
                return False
            live = [
                source
                for source in self._sources
                if not self._state[source.log_type].is_polling_disabled
            ]
            if not live:
                return True
            available = self._log_types(driver)
            if available is None:
                for source in live:
                    self._record_failure(
                        self._state[source.log_type],
                        source,
                        "the driver would not list its log types",
                    )
                return True
            for source in live:
                self._drain_source(driver, source, available)
            return True
        finally:
            self._draining.release()

    def finish(self, driver: LogReadingDriver) -> tuple[LogCapture, ...]:
        """Stops the timer, drains once more, then says what each log came to.

        Called before the session is quit, because a quit driver reads
        nothing.
        """
        if self._closed.is_set():
            return tuple(self._capture_of(source) for source in self._sources)

        self._stop.set()
        ticker = self._ticker
        if ticker is None:
            # A recorder used without start still supports a one-shot capture.
            self.drain(driver)
        else:
            ticker.join(timeout=self._interval * 2)
        if ticker is not None and ticker.is_alive():
            # Seal the state before reporting it. If the request eventually
            # returns, it must not rewrite evidence after the session closed.
            self._closed.set()
            for source in self._sources:
                self._record_failure(
                    self._state[source.log_type],
                    source,
                    "the last drain could not run, a read was still in flight",
                )
        else:
            self._closed.set()
            self._ticker = None
        captures = tuple(self._capture_of(source) for source in self._sources)
        for capture in captures:
            # The run log is what a reader opens first, so anything short of
            # a whole capture has to reach it. An absence nobody is told of
            # reads as silence from the device, and so does a gap.
            logger.log(
                logging.INFO if capture.is_whole else logging.WARNING,
                "Device log %s",
                capture.describe(),
            )
        return captures

    def stop(self) -> None:
        """Stop background reads when an interruption has ended the driver."""
        self._stop.set()
        self._closed.set()
        ticker = self._ticker
        if ticker is not None:
            ticker.join(timeout=self._interval * 2)
        self._ticker = None

    def _capture_of(self, source: LogSource) -> LogCapture:
        state = self._state[source.log_type]
        return LogCapture(
            log_type=source.log_type,
            path=state.path,
            line_count=state.line_count,
            truncated=state.truncated,
            failed_drains=state.failed_drains,
            reason=state.unwritten_reason if state.path is None else _gaps_in(state),
        )

    def _log_types(self, driver: LogReadingDriver) -> Optional[frozenset[str]]:
        """What this session answers for, asked once.

        A driver lists the same types for its whole life, and the question
        costs a round trip to the server.

        None when the driver would not say. A refusal counts against
        every source. So a driver that never answers stops the timer,
        rather than repeating one warning for the length of the run.
        """
        if self._available is not None:
            return self._available
        try:
            self._available = frozenset(f"{name}" for name in driver.log_types)
        except Exception:  # noqa: BLE001 - a log capture must not fail a run
            return None
        return self._available

    @staticmethod
    def _record_failure(state: _SourceState, source: LogSource, detail: str) -> None:
        """Counts a drain that did not land, and says where it happened.

        Said now rather than only at the end, because which boundary failed
        is what tells a reader which part of the log to distrust.
        """
        state.failed_drains += 1
        state.consecutive_failures += 1
        state.last_failure = detail
        if state.path is None:
            state.unwritten_reason = detail
        logger.warning("Device log %s drain failed: %s", source.log_type, detail)
        if (
            state.consecutive_failures
            >= CONSECUTIVE_FAILURES_BEFORE_DISABLING_SOURCE
        ):
            state.is_polling_disabled = True
            logger.warning(
                "Device log %s disabled after %d failed reads in a row.",
                source.log_type,
                state.consecutive_failures,
            )

    def _drain_source(
        self, driver: LogReadingDriver, source: LogSource, available: frozenset[str]
    ) -> None:
        state = self._state[source.log_type]
        if self._segment is None:
            state.unwritten_reason = (
                f"{MAX_SEGMENTS} device logs are already written here"
            )
            return

        if source.log_type not in available:
            state.consecutive_failures = 0
            listed = ", ".join(sorted(available)) or "nothing"
            state.unwritten_reason = f"the driver does not offer it. It offers: {listed}"
            return

        try:
            entries = list(driver.get_log(source.log_type))
        except Exception as exc:  # noqa: BLE001 - a log capture must not fail a run
            if not self._closed.is_set():
                self._record_failure(
                    state, source, f"the read failed: {type(exc).__name__}"
                )
            return

        if self._closed.is_set():
            return

        if not entries:
            state.consecutive_failures = 0
            if state.path is None:
                state.unwritten_reason = "the driver offered it and returned nothing"
            return

        try:
            lines = [_format_line(entry) for entry in entries]
            path = self._layout.device_log(source.file_stem, self._segment)
            # A device log is bytes rather than text. One unpaired surrogate
            # must not cost the drain, and the escape keeps the byte visible.
            with path.open("a", encoding="utf-8", errors="backslashreplace") as opened:
                opened.write("\n".join(lines) + "\n")
        except Exception as exc:  # noqa: BLE001 - a log capture must not fail a run
            if not self._closed.is_set():
                self._record_failure(
                    state, source, f"the write failed: {type(exc).__name__}"
                )
            return

        state.path = path
        state.line_count += len(lines)
        state.consecutive_failures = 0
        # A full drain arrived at a buffer that had already evicted.
        state.truncated = state.truncated or len(lines) >= BUFFERED_LINES
