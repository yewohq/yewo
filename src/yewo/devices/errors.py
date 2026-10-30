"""Failures reported by device platform boundaries."""


class DeviceError(RuntimeError):
    """A device operation could not complete."""


class DeviceCommandError(DeviceError):
    """A platform command could not complete successfully."""


class DeviceInventoryError(DeviceError):
    """Device discovery could not produce a trustworthy inventory."""


class DeviceLifecycleError(DeviceError):
    """A selected device could not become ready for interaction."""


class DeviceInteractionError(DeviceError):
    """A device interaction was invalid or could not complete."""


class DeviceStartupTimeoutError(DeviceLifecycleError):
    """A selected device did not become ready before its deadline."""


class DeviceSelectionError(DeviceError):
    """A device query did not identify exactly one target."""


class DeviceNotFoundError(DeviceSelectionError):
    """No discovered target matched a device query."""


class AmbiguousDeviceError(DeviceSelectionError):
    """More than one discovered target matched a device query."""
