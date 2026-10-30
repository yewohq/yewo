"""Failures reported by mobile automation infrastructure."""


class AutomationError(RuntimeError):
    """Mobile automation infrastructure could not complete an operation."""


class AppiumServerError(AutomationError):
    """An Appium server could not be acquired or released."""


class AppiumRuntimeError(AppiumServerError):
    """No compatible local Appium runtime was available."""


class AppiumStartupTimeoutError(AppiumServerError):
    """A started Appium process did not become ready before its deadline."""


class DriverOptionsError(AutomationError):
    """Driver capabilities could not represent the selected target and application."""


class DriverSessionError(AutomationError):
    """An Appium driver session could not be opened or closed."""


class AutomationSessionError(AutomationError):
    """A composed device, server, driver, and application session failed."""
