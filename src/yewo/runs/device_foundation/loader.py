"""Bounded loading for caller-owned device-foundation scripts."""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

from pydantic import ValidationError

from yewo.runs.device_foundation.records import DeviceFoundationScript


MAXIMUM_DEVICE_FOUNDATION_SCRIPT_BYTES = 1024 * 1024


class DeviceFoundationScriptError(ValueError):
    """A script file cannot produce a trusted deterministic workflow."""


@dataclass(frozen=True)
class LoadedDeviceFoundationScript:
    """A validated script bound to the digest of its exact source bytes."""

    script: DeviceFoundationScript
    source_sha256: str


def load_device_foundation_script_source(path: Path) -> LoadedDeviceFoundationScript:
    """Load one bounded JSON script and retain its source identity."""
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
        script = DeviceFoundationScript.model_validate(payload)
    except (UnicodeError, json.JSONDecodeError, ValidationError) as error:
        raise DeviceFoundationScriptError("The device-foundation script is invalid.") from error
    return LoadedDeviceFoundationScript(script=script, source_sha256=sha256(content).hexdigest())


def load_device_foundation_script(path: Path) -> DeviceFoundationScript:
    """Load one validated script when its source identity is not required."""
    return load_device_foundation_script_source(path).script
