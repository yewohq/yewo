"""The device's own log, taken through the driver that holds the session.

Yewo's log says what Yewo asked for and the Appium log says what the server
did with it. Neither can see inside the device, so a run that ends
`ERROR (device)` cannot say what the device was doing.

`driver.get_log` is the seam. Both drivers implement it, `adb` does not exist
on iOS, and the driver buffers from session start. A device's own ring buffer
survives an emulator reboot, so reading that instead returns another run's
lines.

A target that cannot be asked records a reason, never an empty file. An empty
file says the device was silent. A missing capture says nobody could ask it.
A reader has to be able to tell those two apart.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Protocol, Sequence

from yewo.agent.artifacts.layout import RunLayout
from yewo.agent.device.platform import Platform

logger = logging.getLogger(__name__)

BUFFERED_LINES = 10000
"""Appium's cap per log. It discards from the front, so a full buffer has lost
the beginning of the run rather than the end."""

MAX_SEGMENTS = 999
"""How many times one session may be recovered before its logs stop being
numbered. The width of the number in the filename."""


@dataclass(frozen=True)
class LogSource:
    """One log a platform is worth asking for, under the driver's name for it."""

    driver_name: str
    stem: str


LOG_SOURCES_BY_PLATFORM: dict[Platform, tuple[LogSource, ...]] = {
    Platform.ANDROID: (LogSource("logcat", "logcat"),),
    Platform.IOS: (LogSource("syslog", "syslog"), LogSource("crashlog", "crashlog")),
}
"""What to ask each platform for. Whether this target can answer is not
tabulated, because `log_types` is the open session answering it.

XCUITest lists `syslog` and `crashlog` statically, for a simulator and for a
physical device alike, and reads a real device in process over RemoteXPC. So
on iOS an empty answer is the refusal, rather than an absent log type.
"""


@dataclass(frozen=True)
class LogCapture:
    """A written log, or the reason there is none."""

    driver_name: str
    path: Optional[Path] = None
    line_count: int = 0
    truncated: bool = False
    reason: str = ""

    @property
    def is_written(self) -> bool:
        return self.path is not None

    def describe(self) -> str:
        if not self.is_written:
            return f"{self.driver_name}: not captured, {self.reason}"
        lost = ", and the oldest lines were dropped" if self.truncated else ""
        return f"{self.driver_name}: {self.line_count} lines{lost}"


class LogReadingDriver(Protocol):
    @property
    def log_types(self) -> Sequence[str]: ...

    def get_log(self, log_type: str) -> Sequence[object]: ...


def _resolve_free_segment(
    layout: RunLayout, sources: Sequence[LogSource]
) -> Optional[int]:
    """One number for the whole capture, so a segment names a session.

    Numbered because session recovery opens a new session on the same id.
    One name would let the healthy replacement overwrite the failed
    session's log.

    Chosen once for every source rather than per source. A crashlog is read
    against the syslog beside it. Numbering each on its own pairs a crash
    with another session's log whenever one source is silent.

    None when every segment is taken. Bounded rather than open, because this
    runs during teardown, and a search that cannot end would hang the run
    instead of reporting.
    """
    for segment in range(1, MAX_SEGMENTS + 1):
        if not any(layout.device_log(source.stem, segment).exists() for source in sources):
            return segment
    return None


def _format_timestamp(value: object) -> str:
    """The host's receipt time, which is the clock Yewo's own log runs on.

    A logcat line carries the device's clock in its text, and the two drift.
    Keeping both lines a device failure up against the step that provoked it.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return ""
    try:
        return datetime.fromtimestamp(value / 1000).isoformat(sep=" ", timespec="milliseconds")
    except (OverflowError, OSError, ValueError):
        return ""


def _format_line(entry: object) -> str:
    """Selenium's contract is a mapping holding a message, a level and a
    timestamp. A driver that returns anything else still has a line in it."""
    if not isinstance(entry, dict):
        return str(entry)
    parts = (
        _format_timestamp(entry.get("timestamp")),
        str(entry.get("level") or ""),
        str(entry.get("message") or ""),
    )
    return " ".join(part for part in parts if part)


def _read_log_types(driver: LogReadingDriver) -> frozenset[str]:
    """What this session says it will answer for. Empty on failure, which
    refuses each log with a reason rather than asking for one it would
    reject."""
    try:
        return frozenset(f"{name}" for name in driver.log_types)
    except Exception as exc:  # noqa: BLE001 - a log capture must not fail a run
        logger.warning("The driver would not list its log types: %s", type(exc).__name__)
        return frozenset()


def _capture_log(
    driver: LogReadingDriver, source: LogSource, layout: RunLayout, segment: int
) -> LogCapture:
    try:
        entries = list(driver.get_log(source.driver_name))
    except Exception as exc:  # noqa: BLE001 - a log capture must not fail a run
        return LogCapture(source.driver_name, reason=f"the read failed: {type(exc).__name__}")

    if not entries:
        return LogCapture(
            source.driver_name, reason="the driver offered it and returned nothing"
        )

    try:
        lines = [_format_line(entry) for entry in entries]
        path = layout.device_log(source.stem, segment)
        # A device log is bytes rather than text. One unpaired surrogate must
        # not cost the whole capture, and the escape keeps the byte visible.
        path.write_text("\n".join(lines) + "\n", encoding="utf-8", errors="backslashreplace")
    except Exception as exc:  # noqa: BLE001 - a log capture must not fail a run
        return LogCapture(source.driver_name, reason=f"the write failed: {type(exc).__name__}")

    return LogCapture(
        source.driver_name,
        path=path,
        line_count=len(lines),
        truncated=len(lines) >= BUFFERED_LINES,
    )


def capture(
    driver: LogReadingDriver, platform: Platform, layout: RunLayout
) -> tuple[LogCapture, ...]:
    """Every log worth taking from this target, each written or explained.

    Call before the session is quit, because a quit driver reads nothing.
    Nothing here raises. The run whose log is wanted is the run that failed
    on the device, and a capture that failed it would take the evidence too.
    """
    sources = LOG_SOURCES_BY_PLATFORM.get(platform, ())
    available_logs = _read_log_types(driver)
    listed = ", ".join(sorted(available_logs)) or "nothing"
    segment = _resolve_free_segment(layout, sources)

    captures: List[LogCapture] = []
    for source in sources:
        if segment is None:
            captures.append(
                LogCapture(
                    source.driver_name,
                    reason=f"{MAX_SEGMENTS} device logs are already written here",
                )
            )
        elif source.driver_name in available_logs:
            captures.append(_capture_log(driver, source, layout, segment))
        else:
            captures.append(
                LogCapture(
                    source.driver_name,
                    reason=f"the driver does not offer it. It offers: {listed}",
                )
            )

    for log_capture in captures:
        # A refusal goes to the run log, which is what a reader opens first.
        # An absence nobody is told about reads as silence from the device.
        logger.log(
            logging.INFO if log_capture.is_written else logging.WARNING,
            "Device log %s",
            log_capture.describe(),
        )
    return tuple(captures)
