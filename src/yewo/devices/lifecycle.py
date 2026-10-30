"""Platform-neutral ownership for a prepared device."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from yewo.devices.inventory import DeviceDescriptor
from yewo.resources import ResourceLease, ResourceOwnership


class DeviceLease(ResourceLease):
    """A prepared device and its idempotent release operation."""

    def __init__(
        self,
        descriptor: DeviceDescriptor,
        ownership: ResourceOwnership,
        release: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(ownership, release)
        self.descriptor = descriptor


class DeviceLifecycle(Protocol):
    """Prepare one selected device and return its ownership handle."""

    def prepare(self, device: DeviceDescriptor) -> DeviceLease:
        """Make the selected device ready for interaction."""
        ...
