"""Tests for the device inventory boundary."""

import pytest
from pydantic import ValidationError

from yewo.devices.inventory import (
    DeviceDescriptor,
    DeviceInventory,
    DeviceKind,
    DevicePlatform,
    DeviceQuery,
    DeviceReadiness,
)


def _android_device() -> DeviceDescriptor:
    return DeviceDescriptor(
        platform=DevicePlatform.ANDROID,
        target_id="Pixel_9_API_35",
        connection_id="emulator-5554",
        display_name="Pixel 9 API 35",
        kind=DeviceKind.VIRTUAL,
        readiness=DeviceReadiness.READY,
        native_state="device",
        operating_system_version="15",
        device_model="Pixel 9",
        architecture="arm64-v8a",
        screen_width_px=1080,
        screen_height_px=2424,
    )


def test_query_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        DeviceQuery.model_validate({"platform": "android", "serial": "emulator-5554"})


def test_descriptor_round_trip_preserves_observed_facts() -> None:
    descriptor = _android_device()

    restored = DeviceDescriptor.model_validate_json(descriptor.model_dump_json())

    assert restored == descriptor


def test_descriptor_requires_both_screen_dimensions() -> None:
    with pytest.raises(ValidationError, match="Screen width and height"):
        DeviceDescriptor(
            platform=DevicePlatform.IOS,
            target_id="A1B2C3",
            display_name="iPhone 16",
            kind=DeviceKind.VIRTUAL,
            readiness=DeviceReadiness.READY,
            native_state="Booted",
            screen_width_px=1179,
        )


def test_inventory_uses_structural_implementation() -> None:
    class InventoryStub:
        def list_devices(self) -> tuple[DeviceDescriptor, ...]:
            return (_android_device(),)

    inventory: DeviceInventory = InventoryStub()

    assert inventory.list_devices() == (_android_device(),)
