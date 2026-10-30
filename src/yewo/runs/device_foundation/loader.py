"""Bounded loading for caller-owned device-foundation scripts."""

import json
from pathlib import Path

from pydantic import ValidationError

from yewo.runs.device_foundation.records import DeviceFoundationScript


MAXIMUM_DEVICE_FOUNDATION_SCRIPT_BYTES = 1024 * 1024


class DeviceFoundationScriptError(ValueError):
    """A script file cannot produce a trusted deterministic workflow."""


def load_device_foundation_script(path: Path) -> DeviceFoundationScript:
    """Load one bounded JSON script without retaining invalid input text."""
    try:
        if path.is_symlink() or not path.is_file():
            raise DeviceFoundationScriptError(
                "The device-foundation script must be a regular file."
            )
        size = path.stat().st_size
        if size > MAXIMUM_DEVICE_FOUNDATION_SCRIPT_BYTES:
            raise DeviceFoundationScriptError(
                "The device-foundation script exceeds the one-megabyte limit."
            )
        with path.open("rb") as script_file:
            content = script_file.read(MAXIMUM_DEVICE_FOUNDATION_SCRIPT_BYTES + 1)
    except DeviceFoundationScriptError:
        raise
    except OSError as error:
        raise DeviceFoundationScriptError("Could not read the device-foundation script.") from error
    if len(content) > MAXIMUM_DEVICE_FOUNDATION_SCRIPT_BYTES:
        raise DeviceFoundationScriptError(
            "The device-foundation script exceeds the one-megabyte limit."
        )
    if len(content) != size:
        raise DeviceFoundationScriptError("The device-foundation script changed while it was read.")
    try:
        payload = json.loads(content)
        return DeviceFoundationScript.model_validate(payload)
    except (UnicodeError, json.JSONDecodeError, ValidationError) as error:
        raise DeviceFoundationScriptError("The device-foundation script is invalid.") from error
