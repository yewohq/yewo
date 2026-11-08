"""Classify iOS diagnostic reports without model inference."""

from collections.abc import Iterator, Mapping
from datetime import datetime, timezone
import json

from yewo.devices.diagnostics import ApplicationFailureKind
from yewo.devices.logs.errors import DeviceLogCaptureError
from yewo.devices.logs.reader import ApplicationDiagnosticReport


_NONFATAL_MARKERS = (
    "simulated (this is not a crash)",
    "non-fatal condition (this is not a crash)",
)
_WATCHDOG_CODE = "0x8badf00d"
_WATCHDOG_DECIMAL_CODE = str(int(_WATCHDOG_CODE, 16))
_APPLICATION_IDENTITY_KEYS = frozenset({"bundleid", "bundleidentifier", "cfbundleidentifier"})


def parse_ios_diagnostic_report(
    source_name: str,
    content: str,
) -> ApplicationDiagnosticReport:
    """Normalize one `.ips` report and preserve its platform explanation."""
    values = tuple(_json_values(content))
    if not values:
        raise DeviceLogCaptureError("The iOS diagnostic report is empty.")
    fields = _collect_fields(values)
    searchable_text = " ".join(str(value) for value in fields.values()).casefold()
    termination_code = _first_field(fields, "terminationcode")
    exception_type = _first_field(fields, "exceptiontype")
    termination_reason = _first_field(
        fields,
        "terminationreason",
        "terminationdescription",
        "terminationindicator",
    )
    if any(marker in searchable_text for marker in _NONFATAL_MARKERS):
        failure_kind = ApplicationFailureKind.UNCLASSIFIED
    elif _WATCHDOG_CODE in searchable_text or _WATCHDOG_DECIMAL_CODE in searchable_text:
        failure_kind = ApplicationFailureKind.ANR
    elif exception_type is not None or termination_reason is not None:
        failure_kind = ApplicationFailureKind.CRASH
    else:
        failure_kind = ApplicationFailureKind.UNCLASSIFIED
    return ApplicationDiagnosticReport(
        source_name=source_name,
        text=content,
        failure_kind=failure_kind,
        occurred_at=_report_time(fields),
        platform_reason=_string_value(exception_type or termination_reason),
        platform_code=_string_value(termination_code),
    )


def ios_report_contains_application(content: str, application_id: str) -> bool:
    """Return whether a valid report identifies the application under test."""
    values = tuple(_json_values(content))
    if not values:
        raise DeviceLogCaptureError("The iOS diagnostic report is empty.")

    def contains(value: object) -> bool:
        if isinstance(value, Mapping):
            return any(
                (
                    isinstance(key, str)
                    and _field_key(key) in _APPLICATION_IDENTITY_KEYS
                    and nested_value == application_id
                )
                or contains(nested_value)
                for key, nested_value in value.items()
            )
        if isinstance(value, list):
            return any(contains(item) for item in value)
        return False

    return any(contains(value) for value in values)


def _json_values(content: str) -> Iterator[object]:
    decoder = json.JSONDecoder()
    position = 0
    while position < len(content):
        while position < len(content) and content[position].isspace():
            position += 1
        if position == len(content):
            return
        try:
            value, position = decoder.raw_decode(content, position)
        except json.JSONDecodeError as error:
            raise DeviceLogCaptureError("The iOS diagnostic report is invalid.") from error
        yield value


def _collect_fields(values: tuple[object, ...]) -> dict[str, object]:
    fields: dict[str, object] = {}

    def collect(value: object, parent_key: str = "") -> None:
        if isinstance(value, Mapping):
            for key, nested_value in value.items():
                normalized_key = _field_key(key) if isinstance(key, str) else ""
                combined_key = f"{parent_key}{normalized_key}"
                if normalized_key and not isinstance(nested_value, Mapping | list):
                    fields.setdefault(normalized_key, nested_value)
                    fields.setdefault(combined_key, nested_value)
                collect(nested_value, combined_key)
        elif isinstance(value, list):
            for item in value:
                collect(item, parent_key)

    for value in values:
        collect(value)
    return fields


def _field_key(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _first_field(fields: Mapping[str, object], *keys: str) -> object | None:
    return next((fields[key] for key in keys if key in fields), None)


def _report_time(fields: Mapping[str, object]) -> datetime | None:
    value = _first_field(fields, "timestamp", "capturetime", "datetime")
    if isinstance(value, int | float) and not isinstance(value, bool):
        seconds = value / 1000 if value > 10_000_000_000 else value
        try:
            return datetime.fromtimestamp(seconds, timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None
    if not isinstance(value, str):
        return None
    try:
        report_time = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        try:
            report_time = _parse_apple_report_time(value)
        except ValueError as error:
            raise DeviceLogCaptureError(
                "The iOS diagnostic report has an invalid timestamp."
            ) from error
    return report_time if report_time.utcoffset() is not None else None


def _parse_apple_report_time(value: str) -> datetime:
    for date_format in (
        "%Y-%m-%d %H:%M:%S.%f %z",
        "%Y-%m-%d %H:%M:%S %z",
    ):
        try:
            return datetime.strptime(value, date_format)
        except ValueError:
            continue
    raise ValueError("The Apple diagnostic timestamp has an unsupported shape.")


def _string_value(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text[:256] if text else None
