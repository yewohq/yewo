"""Live session health checks used by the journey runner."""

import logging

from yewo.agent.device.platform import Platform

logger = logging.getLogger(__name__)


def is_session_responsive(device_control) -> bool:
    if not device_control or not device_control.driver:
        return False
    try:
        if device_control.platform == Platform.ANDROID:
            _ = device_control.driver.current_package
        else:
            _ = device_control.driver.orientation
        return True
    except Exception as exc:  # noqa: BLE001 - driver boundary
        logger.warning("Session responsiveness check failed: %s", exc)
        return False
