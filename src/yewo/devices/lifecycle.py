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
        restore_environment: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(ownership, release)
        self.descriptor = descriptor
        self._restore_environment = restore_environment

    def close(self) -> None:
        """Restore run-owned state, then release an owned device."""
        if self.closed:
            return
        restore_error: Exception | None = None
        if self._restore_environment is not None:
            try:
                self._restore_environment()
            except Exception as error:  # noqa: BLE001 - device cleanup must continue
                restore_error = error
        try:
            super().close()
        except Exception as release_error:
            if restore_error is not None:
                release_error.add_note(
                    f"Device environment restoration also failed: {restore_error}"
                )
            raise
        if restore_error is not None:
            raise restore_error


class DeviceLifecycle(Protocol):
    """Prepare one selected device and return its ownership handle."""

    def prepare(self, device: DeviceDescriptor) -> DeviceLease:
        """Make the selected device ready for interaction."""
        ...
