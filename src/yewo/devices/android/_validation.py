"""Shared validation for Android command values."""

import re
import shlex
from urllib.parse import urlsplit

from yewo.devices.errors import DeviceCommandError


PACKAGE_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)*$")
ACTIVITY_PATTERN = re.compile(
    r"^\.?[A-Za-z][A-Za-z0-9_.$]*(?:\.[A-Za-z][A-Za-z0-9_.$]*)*$"
)
LOCALE_PATTERN = re.compile(r"^[A-Za-z0-9]{1,8}(?:-[A-Za-z0-9]{1,8})*$")
TIMEZONE_PATTERN = re.compile(r"^[A-Za-z0-9._+-]+(?:/[A-Za-z0-9._+-]+)*$")


def validate_remote_value(value: str, name: str) -> None:
    """Reject an empty value or a value that can split a remote command."""
    if not value or any(character in value for character in ("\x00", "\n", "\r")):
        raise DeviceCommandError(f"The {name} is empty or contains a control character.")


def quote_remote_arguments(arguments: tuple[str, ...]) -> tuple[str, ...]:
    """Quote validated Android shell arguments as one remote command."""
    for argument in arguments:
        if any(character in argument for character in ("\x00", "\n", "\r")):
            raise DeviceCommandError("An Android shell argument contains a control character.")
    return tuple(shlex.quote(argument) for argument in arguments)


def validate_pattern(value: str, pattern: re.Pattern[str], name: str) -> None:
    """Require one value to match its complete Android command vocabulary."""
    if pattern.fullmatch(value) is None:
        raise DeviceCommandError(f"The {name} has an invalid format.")


def validate_package_id(value: str, *, name: str = "package ID") -> None:
    """Validate one Android package-shaped identifier."""
    validate_pattern(value, PACKAGE_PATTERN, name)


def validate_component_name(value: str) -> None:
    """Validate one package/activity component name."""
    try:
        package_id, activity = value.split("/", maxsplit=1)
    except ValueError as error:
        raise DeviceCommandError(
            "The accessibility service must use package/activity format."
        ) from error
    validate_package_id(package_id)
    validate_pattern(activity, ACTIVITY_PATTERN, "accessibility service activity")


def validate_deep_link(url: str) -> None:
    """Require a safe deep link with an explicit URI scheme."""
    validate_remote_value(url, "deep link URL")
    if not urlsplit(url).scheme:
        raise DeviceCommandError("The deep link URL must include a scheme.")
