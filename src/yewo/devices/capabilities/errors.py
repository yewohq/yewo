"""Errors from device-capability transaction execution."""

from yewo.devices.capabilities.records import CapabilityTransactionReceipt


class DeviceCapabilityError(RuntimeError):
    """Base error for transactional device capabilities."""


class CapabilityApplicationError(DeviceCapabilityError):
    """A capability could not be applied and verified."""

    def __init__(self, receipt: CapabilityTransactionReceipt) -> None:
        super().__init__(receipt.application.reason)
        self.receipt = receipt


class CapabilityRestorationError(DeviceCapabilityError):
    """A capability transaction could not restore its original state."""

    def __init__(self, receipt: CapabilityTransactionReceipt) -> None:
        super().__init__(receipt.restoration.reason)
        self.receipt = receipt
