"""Failures raised by device-log capture boundaries."""


class DeviceLogCaptureError(RuntimeError):
    """A device-log boundary or artifact operation failed."""


class DeviceLogRollbackError(DeviceLogCaptureError):
    """Incomplete device-log artifacts could not be removed safely."""
