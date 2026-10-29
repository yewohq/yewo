"""Android input preferences for Yewo's touch-only device sessions."""

from dataclasses import dataclass
from typing import Optional

from yewo.agent.device.adb_control import AdbControl


@dataclass(frozen=True)
class TouchInputSettings:
    """The handwriting preference present before an automated session."""

    stylus_handwriting: Optional[bool]


def read_touch_input(serial: str) -> Optional[TouchInputSettings]:
    """Read the handwriting preference, including its unset default state."""
    value = AdbControl.ask(
        serial, "settings", "get", "secure", "stylus_handwriting_enabled"
    )
    if value == "1":
        return TouchInputSettings(stylus_handwriting=True)
    if value == "0":
        return TouchInputSettings(stylus_handwriting=False)
    if value == "null":
        return TouchInputSettings(stylus_handwriting=None)
    return None


def prepare_touch_input(serial: str, settings: TouchInputSettings) -> None:
    """Disable handwriting because Yewo injects touch input, not stylus input."""
    if settings.stylus_handwriting is False:
        return
    AdbControl.shell(
        serial,
        "settings",
        "put",
        "secure",
        "stylus_handwriting_enabled",
        "0",
        doing="disable stylus handwriting for touch input",
    )


def restore_touch_input(serial: str, settings: TouchInputSettings) -> None:
    """Restore the exact handwriting preference present before the session."""
    if settings.stylus_handwriting is None:
        AdbControl.shell(
            serial,
            "settings",
            "delete",
            "secure",
            "stylus_handwriting_enabled",
            doing="restore the unset stylus handwriting preference",
        )
        return
    AdbControl.shell(
        serial,
        "settings",
        "put",
        "secure",
        "stylus_handwriting_enabled",
        "1" if settings.stylus_handwriting else "0",
        doing="restore the stylus handwriting preference",
    )
