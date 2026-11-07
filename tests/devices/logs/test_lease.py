"""Run-owned device-log checkpoint lease tests."""

from typing import cast
from unittest.mock import Mock

from yewo.devices.logs import (
    DeviceLogBoundary,
    DeviceLogCheckpointKind,
    DeviceLogDrainReceipt,
)
from yewo.devices.logs.lease import DeviceLogCaptureLease
from yewo.devices.logs.service import DeviceLogCaptureService


def test_application_setup_drain_consumes_its_own_checkpoint() -> None:
    service = Mock(spec=DeviceLogCaptureService)
    receipt = Mock(spec=DeviceLogDrainReceipt)
    service.drain.return_value = receipt
    lease = DeviceLogCaptureLease(
        service,
        cast(DeviceLogBoundary, Mock()),
        cast(DeviceLogDrainReceipt, Mock()),
    )

    returned = lease.drain_application_setup("setup_1")

    assert returned is receipt
    service.drain.assert_called_once_with(
        lease.boundary,
        DeviceLogCheckpointKind.APPLICATION_SETUP,
        "setup_1",
    )
    assert lease.receipts[-1] is receipt
