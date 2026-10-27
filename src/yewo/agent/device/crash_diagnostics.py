"""App-scoped crash facts from device-log formats Yewo explicitly supports.

Appium exposes device logs as text. This module first parses that transport
text into typed records. Source-specific extractors then classify only records
that can be attributed to the app under test. Unknown formats produce no
diagnostic instead of a guessed failure.
"""

import re
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

from pydantic import BaseModel, Field


class CrashDiagnostic(BaseModel):
    """A fatal crash, uncaught exception, or ANR attributed to the app."""

    exception_type: str = Field(default="Crash", description="Exception class or crash kind.")
    message: str = Field(default="", description="The exception message or failure reason.")
    stack_trace: list[str] = Field(
        default_factory=list,
        description="At most ten stack frames from the app-scoped log window.",
    )
    source: str = Field(description="The device log that supplied the diagnostic.")

    @property
    def summary(self) -> str:
        if self.exception_type and self.message:
            return f"{self.exception_type}: {self.message}"
        return self.message or self.exception_type or "Uncaught exception"


class CrashDiagnosticSource(Protocol):
    """A session boundary that can attribute captured failures to its app."""

    def collect_crash_diagnostic(self) -> Optional[CrashDiagnostic]: ...


@dataclass(frozen=True)
class LogcatRecord:
    """The fields crash classification reads from Android threadtime logs."""

    tag: str
    message: str


LOGCAT_THREADTIME_RE = re.compile(
    r"(?:^|\s)\d+\s+\d+\s+[VDIWEF]\s+"
    r"(?P<tag>[^:]+?)\s*:\s*(?P<message>.*)$"
)
EXCEPTION_WITH_MESSAGE_RE = re.compile(
    r"(?P<type>[A-Za-z0-9_.$]+(?:Exception|Error)):\s*(?P<message>.*)"
)
IOS_UNCAUGHT_RE = re.compile(
    r"\*\*\* Terminating app due to uncaught exception '([^']+)', reason: '([^']*)'",
    re.IGNORECASE,
)

MAX_INSPECTED_LINES = 5000
ANDROID_RUNTIME_TAG = "AndroidRuntime"
ANDROID_ACTIVITY_TAG = "ActivityManager"


def _parse_logcat_records(lines: Sequence[str]) -> list[LogcatRecord]:
    records = []
    for line in lines:
        match = LOGCAT_THREADTIME_RE.search(line.rstrip())
        if match is None:
            continue
        records.append(
            LogcatRecord(
                tag=match.group("tag").strip(),
                message=match.group("message").strip(),
            )
        )
    return records


def _process_matches(package: str, target_package: Optional[str]) -> bool:
    return target_package is None or package == target_package or package.startswith(
        f"{target_package}:"
    )


def _android_process_from_window(records: Sequence[LogcatRecord]) -> Optional[str]:
    for record in records:
        if not record.message.startswith("Process:"):
            continue
        process_text = record.message.removeprefix("Process:").split(",", 1)[0].strip()
        return process_text or None
    return None


def _extract_android_runtime_diagnostic(
    records: Sequence[LogcatRecord], target_package: Optional[str]
) -> Optional[CrashDiagnostic]:
    for index in range(len(records) - 1, -1, -1):
        record = records[index]
        if record.tag != ANDROID_RUNTIME_TAG or not record.message.startswith(
            "FATAL EXCEPTION:"
        ):
            continue
        window = records[index : min(len(records), index + 25)]
        process = _android_process_from_window(window[:4])
        if process is None or not _process_matches(process, target_package):
            continue
        exception_type = "FatalException"
        message = ""
        stack_trace = []
        for window_record in window[1:]:
            if window_record.tag != ANDROID_RUNTIME_TAG:
                continue
            exception_match = EXCEPTION_WITH_MESSAGE_RE.search(window_record.message)
            if not message and exception_match:
                exception_type = exception_match.group("type")
                message = exception_match.group("message").strip()
            elif window_record.message.startswith("at "):
                stack_trace.append(window_record.message)
            elif stack_trace:
                break
        return CrashDiagnostic(
            exception_type=exception_type,
            message=message or f"Process {process} crashed",
            stack_trace=stack_trace[:10],
            source="logcat",
        )
    return None


def _extract_android_anr(
    records: Sequence[LogcatRecord], target_package: Optional[str]
) -> Optional[CrashDiagnostic]:
    for record in reversed(records):
        if record.tag != ANDROID_ACTIVITY_TAG or not record.message.startswith("ANR in "):
            continue
        process = record.message.removeprefix("ANR in ").split(" ", 1)[0]
        if _process_matches(process, target_package):
            return CrashDiagnostic(
                exception_type="ANR",
                message=f"Application Not Responding in {process}",
                source="logcat",
            )
    return None


def _extract_ios_diagnostic(
    lines: Sequence[str],
    source: str,
    target_app_id: Optional[str],
    target_process_name: Optional[str],
) -> Optional[CrashDiagnostic]:
    identities = tuple(value for value in (target_app_id, target_process_name) if value)
    for index in range(len(lines) - 1, -1, -1):
        line = lines[index]
        window = lines[max(0, index - 4) : min(len(lines), index + 5)]
        if identities and not any(identity in row for identity in identities for row in window):
            continue
        uncaught_match = IOS_UNCAUGHT_RE.search(line)
        if uncaught_match:
            return CrashDiagnostic(
                exception_type=uncaught_match.group(1),
                message=uncaught_match.group(2),
                source=source,
            )
        if "Fatal error:" in line:
            return CrashDiagnostic(
                exception_type="FatalError",
                message=line.split("Fatal error:", 1)[1].strip(),
                source=source,
            )
    return None


def extract_crash_diagnostic(
    lines: Sequence[str],
    *,
    source: str,
    target_package: Optional[str] = None,
    target_app_id: Optional[str] = None,
    target_process_name: Optional[str] = None,
) -> Optional[CrashDiagnostic]:
    """Return a diagnostic only for a supported, app-attributed log format."""
    inspected_lines = lines[-MAX_INSPECTED_LINES:]
    if source == "logcat":
        records = _parse_logcat_records(inspected_lines)
        return (
            _extract_android_runtime_diagnostic(records, target_package)
            or _extract_android_anr(records, target_package)
        )
    return _extract_ios_diagnostic(
        inspected_lines, source, target_app_id, target_process_name
    )
