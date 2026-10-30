"""Automation infrastructure for mobile driver sessions."""

from yewo.automation.appium_runtime import AppiumRuntime, resolve_appium_runtime
from yewo.automation.appium_server import (
    AppiumServerConfig,
    AppiumServerLease,
    AppiumServerManager,
)
from yewo.automation.driver_options import DriverOptionsBuilder
from yewo.automation.driver_session import (
    DriverClient,
    DriverFactory,
    DriverSessionLease,
    DriverSessionManager,
)
from yewo.automation.errors import (
    AppiumRuntimeError,
    AppiumServerError,
    AppiumStartupTimeoutError,
    AutomationError,
    AutomationSessionError,
    DriverOptionsError,
    DriverSessionError,
)
from yewo.automation.session import AutomationSession, AutomationSessionManager

__all__ = [
    "AppiumRuntime",
    "AppiumRuntimeError",
    "AppiumServerConfig",
    "AppiumServerError",
    "AppiumServerLease",
    "AppiumServerManager",
    "AppiumStartupTimeoutError",
    "AutomationError",
    "AutomationSession",
    "AutomationSessionError",
    "AutomationSessionManager",
    "DriverOptionsBuilder",
    "DriverOptionsError",
    "DriverClient",
    "DriverFactory",
    "DriverSessionError",
    "DriverSessionLease",
    "DriverSessionManager",
    "resolve_appium_runtime",
]
