"""Shared validation for Android command values."""

from yewo.devices.errors import DeviceCommandError


def validate_remote_value(value: str, name: str) -> None:
    """Reject an empty value or a value that can split a remote command."""
    if not value or any(character in value for character in ("\x00", "\n", "\r")):
        raise DeviceCommandError(f"The {name} is empty or contains a control character.")
