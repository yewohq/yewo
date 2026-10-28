"""Android accessibility-service setting operations."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def enabled_services(serial: str) -> Optional[str]:
    """Return the complete enabled-service setting."""
    from yewo.agent.device.adb_control import AdbControl

    listed = AdbControl.ask(
        serial, "settings", "get", "secure", "enabled_accessibility_services"
    )
    return "" if listed in (None, "null") else listed


def write_services(serial: str, services: str) -> None:
    """Write the complete enabled-service setting and its master switch."""
    from yewo.agent.device.adb_control import AdbControl

    AdbControl.shell(
        serial,
        "settings",
        "put",
        "secure",
        "enabled_accessibility_services",
        services or '""',
        doing="set the accessibility services",
    )
    AdbControl.shell(
        serial,
        "settings",
        "put",
        "secure",
        "accessibility_enabled",
        "1" if services else "0",
        doing="set the accessibility master switch",
    )


def set_service(serial: str, service: str, enabled: bool) -> None:
    """Add or remove one service without changing the other services."""
    services = [name for name in (enabled_services(serial) or "").split(":") if name]
    if enabled and service not in services:
        services.append(service)
    elif not enabled and service in services:
        services.remove(service)

    logger.info(
        "%s accessibility service %s",
        "Registering" if enabled else "Deregistering",
        service,
    )
    write_services(serial, ":".join(services))
