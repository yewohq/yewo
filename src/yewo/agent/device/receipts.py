"""Reading back the device state a capability asked for.

An adb command exiting 0 says the command ran, not that the state changed. A scenario about being offline is evidence of something only if the device
really was offline. So each capability that can be read is read, and the answer
travels with the request that produced it.

Where no read exists the receipt carries `observed=None`. That is not a
mismatch and does not refuse the capability. It says the state could not be confirmed, which is not the same as the state
being wrong.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

from yewo.agent.device.adb_control import AdbControl
from yewo.errors import DeviceUnavailable

logger = logging.getLogger(__name__)

BATTERY_LEVEL = re.compile(r"^\s*level:\s*(\d+)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Receipt:
    """What a capability asked for, and what the device reports.

    Carries the source command, because a receipt nothing can be traced to is
    an assertion rather than evidence.
    """

    capability: str
    requested: str
    observed: Optional[str]
    target: str
    source: str
    private: bool = False
    """Whether the states here are content rather than a device setting.

    A clipboard carries whatever was copied, which can be a secret or a
    token. Such a receipt still decides whether the capability worked, and it
    reports that without putting the value in a log line or a report.
    """

    @property
    def confirmed(self) -> bool:
        """Whether the device reports the state that was asked for."""
        return self.observed is not None and self.observed == self.requested

    @property
    def unreadable(self) -> bool:
        """Whether the state could not be read, rather than read as wrong."""
        return self.observed is None

    def _shown(self, value: Optional[str]) -> str:
        """One state as it may be written down."""
        if value is None:
            return "unreadable"
        if not self.private:
            return value
        return f"<{len(value)} characters>"

    def __str__(self) -> str:
        return (
            f"{self.capability} on {self.target}: "
            f"requested {self._shown(self.requested)}, "
            f"observed {self._shown(self.observed)} (via {self.source})"
        )


def airplane_mode(serial: str) -> Optional[str]:
    """"on" or "off", as the system setting reports it."""
    value = AdbControl.ask(serial, "settings", "get", "global", "airplane_mode_on")
    if value not in ("0", "1"):
        return None
    return "on" if value == "1" else "off"


def timezone(serial: str) -> Optional[str]:
    """The zone the system actually keeps time in.

    Read from the property and not from `settings global timezone`. The two disagree. A write to the setting leaves the property alone, so
    reading the setting back confirms the write and says nothing about the
    device.
    """
    value = AdbControl.ask(serial, "getprop", "persist.sys.timezone")
    return None if value in (None, "null") else value


def app_locale(serial: str, package: str) -> Optional[str]:
    """The locale set for one app, which is not the system locale."""
    value = AdbControl.ask(serial, "cmd", "locale", "get-app-locales", package)
    if not value:
        return None
    # The command answers "Locales for com.example are [fr-FR]".
    found = re.search(r"\[([^\]]*)\]", value)
    return found.group(1).strip() or None if found else None


def battery_level(serial: str) -> Optional[str]:
    """The battery level the framework reports, as a string of digits."""
    value = AdbControl.ask(serial, "dumpsys", "battery")
    if not value:
        return None
    found = BATTERY_LEVEL.search(value)
    return found.group(1) if found else None


AC_POWERED = re.compile(r"^\s*AC powered:\s*(true|false)\s*$", re.MULTILINE)


def battery_state(serial: str) -> Optional[str]:
    """Level and charging together, as one state to compare against.

    A level on its own confirms half of what a battery capability asked for. A
    scenario about a discharging phone is not satisfied by the right number
    while the device charges.
    """
    value = AdbControl.ask(serial, "dumpsys", "battery")
    if not value:
        return None
    level = BATTERY_LEVEL.search(value)
    powered = AC_POWERED.search(value)
    if not level or not powered:
        return None
    return f"level={level.group(1)},charging={powered.group(1)}"


NOT_REQUESTED = "not requested"


def permission_granted(serial: str, package: str, permission: str) -> Optional[str]:
    """How one package holds one permission, or None when nothing can be read.

    A permission the app never asked for reports NOT_REQUESTED rather than
    "denied" or nothing. `pm grant` exits 0 for one of those and grants nothing. Without this, a
    request to grant it reads as a state that could not be confirmed. It is
    really a state that can never be reached.
    """
    value = AdbControl.ask(serial, "dumpsys", "package", package)
    if not value:
        return None
    for line in value.splitlines():
        if permission in line and "granted=" in line:
            return "granted" if "granted=true" in line else "denied"
    return NOT_REQUESTED if permission not in value else None


def accessibility_enabled(serial: str, service: str) -> Optional[str]:
    """"on" or "off" for one accessibility service.

    Reads the list of enabled services rather than the master switch, because
    the switch can be on with no service running.
    """
    value = AdbControl.ask(serial, "settings", "get", "secure", "enabled_accessibility_services")
    if value is None:
        return None
    return "on" if service in value else "off"


def app_running(serial: str, package: str) -> Optional[str]:
    """"running" or "stopped", read from whether the app holds a pid."""
    try:
        pid = AdbControl.get_app_pid(serial, package)
    except DeviceUnavailable as exc:
        logger.warning("Could not read whether %s is running: %s", package, exc)
        return None
    return "running" if pid else "stopped"
