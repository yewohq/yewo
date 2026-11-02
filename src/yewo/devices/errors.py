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


class DeviceActionRejectedError(DeviceInteractionError):
    """A requested action was invalid or did not reach its verified outcome."""


class DeviceInputRejectedError(DeviceActionRejectedError):
    """Entered text conflicts with a secret-free platform input constraint."""


class DeviceTargetRejectedError(DeviceActionRejectedError):
    """A grounded target did not accept the requested interaction."""


class ApplicationFirstPaintError(DeviceInteractionError):
    """The application did not produce a usable first frame."""


class DeviceStartupTimeoutError(DeviceLifecycleError):
    """A selected device did not become ready before its deadline."""


class DeviceSelectionError(DeviceError):
    """A device query did not identify exactly one target."""


class DeviceNotFoundError(DeviceSelectionError):
    """No discovered target matched a device query."""


class AmbiguousDeviceError(DeviceSelectionError):
    """More than one discovered target matched a device query."""
