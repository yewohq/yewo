"""Platform-neutral observation of application installation state."""

from __future__ import annotations

from typing import Protocol

from yewo.devices.inventory import DeviceDescriptor


class ApplicationPresence(Protocol):
    """Observe whether one application is installed without changing it."""

    def is_installed(
        self,
        device: DeviceDescriptor,
        application_id: str,
    ) -> bool | None:
        """Return installed, absent, or unknown when inspection fails."""
        ...
