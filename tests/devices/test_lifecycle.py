"""Platform-neutral device lease restoration."""

from yewo.devices import (
    DeviceDescriptor,
    DeviceKind,
    DeviceLease,
    DevicePlatform,
    DeviceReadiness,
)
from yewo.resources import ResourceOwnership


def test_external_device_lease_restores_run_environment_once() -> None:
    events: list[str] = []
    device = DeviceDescriptor(
        platform=DevicePlatform.ANDROID,
        target_id="Pixel_9",
        connection_id="emulator-5554",
        display_name="Pixel 9",
        kind=DeviceKind.VIRTUAL,
        readiness=DeviceReadiness.READY,
        native_state="device",
    )
    lease = DeviceLease(
        device,
        ResourceOwnership.EXTERNAL,
        restore_environment=lambda: events.append("restore"),
    )

    lease.close()
    lease.close()

    assert events == ["restore"]
