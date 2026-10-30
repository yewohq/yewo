"""Bound redacted device-log text without losing every oversized entry."""

from yewo.devices.logs.errors import DeviceLogCaptureError
from yewo.devices.logs.records import DeviceLogOmission
from yewo.devices.logs.redaction import DeviceLogRedactionResult


def redaction_policy_id(results: tuple[DeviceLogRedactionResult, ...]) -> str:
    """Return the one policy shared by all entries in an artifact."""
    policy_ids = {result.policy_id for result in results}
    if len(policy_ids) != 1:
        raise DeviceLogCaptureError("One device-log artifact cannot mix redaction policies.")
    return next(iter(policy_ids))


def bound_text_entries(
    entries: tuple[DeviceLogRedactionResult, ...],
    maximum_bytes: int,
) -> tuple[bytes, int, DeviceLogOmission]:
    """Retain the newest entries and preserve a prefix of one oversized entry."""
    retained_lines: list[bytes] = []
    retained_bytes = 0
    omission = DeviceLogOmission.NONE
    for entry_index, entry in reversed(tuple(enumerate(entries))):
        encoded_text = entry.text.encode("utf-8", errors="backslashreplace")
        line = encoded_text + b"\n"
        if retained_bytes + len(line) <= maximum_bytes:
            retained_lines.append(line)
            retained_bytes += len(line)
            continue
        if not retained_lines:
            prefix = (
                utf8_prefix(encoded_text, maximum_bytes - 1)
                if maximum_bytes > 1
                else b""
            )
            if prefix:
                retained_lines.append(prefix + b"\n")
            omission = DeviceLogOmission.LATER
            if entry_index:
                omission = DeviceLogOmission.BOTH
        else:
            omission = combine_omission(omission, DeviceLogOmission.EARLIER)
        break
    retained_lines.reverse()
    return b"".join(retained_lines), len(retained_lines), omission


def utf8_prefix(content: bytes, maximum_bytes: int) -> bytes:
    """Return the longest valid UTF-8 prefix within one byte bound."""
    prefix = content[:maximum_bytes]
    while prefix:
        try:
            prefix.decode("utf-8")
            return prefix
        except UnicodeDecodeError:
            prefix = prefix[:-1]
    if maximum_bytes > 0:
        return b""
    raise DeviceLogCaptureError("The device-log size bound excluded every byte.")


def combine_omission(
    current: DeviceLogOmission,
    added: DeviceLogOmission,
) -> DeviceLogOmission:
    """Combine directional omissions without hiding either side."""
    if current is DeviceLogOmission.NONE:
        return added
    if added is DeviceLogOmission.NONE or current is added:
        return current
    return DeviceLogOmission.BOTH
