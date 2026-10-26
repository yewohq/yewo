"""What a device session owns and keeps in the foreground."""

from enum import Enum


class DeviceScope(str, Enum):
    """The boundary a session is allowed to control."""

    APP = "app"
    """Install or reuse one app and recover unexpected navigation to it."""

    DEVICE = "device"
    """Attach to an existing device and allow navigation between its apps."""
