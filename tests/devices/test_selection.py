"""Tests for deterministic device selection."""

import pytest

from yewo.devices import (
    DeviceDescriptor,
    DeviceKind,
    DevicePlatform,
    DeviceQuery,
    DeviceReadiness,
    select_device,
)
from yewo.devices.errors import AmbiguousDeviceError, DeviceNotFoundError


def _device(
    target_id: str,
    display_name: str,
    *,
    platform: DevicePlatform = DevicePlatform.ANDROID,
    connection_id: str | None = None,
) -> DeviceDescriptor:
    return DeviceDescriptor(
        platform=platform,
        target_id=target_id,
        connection_id=connection_id,
        display_name=display_name,
        kind=DeviceKind.VIRTUAL,
        readiness=(
            DeviceReadiness.READY if connection_id is not None else DeviceReadiness.STARTABLE
        ),
        native_state="device" if connection_id is not None else "stopped",
    )


def test_target_id_selects_a_running_android_emulator() -> None:
    emulator = _device(
        "Pixel_9_API_35",
        "Pixel 9",
        connection_id="emulator-5554",
    )

    selected = select_device(DeviceQuery(target_id="Pixel_9_API_35"), (emulator,))

    assert selected.target_id == "Pixel_9_API_35"
    assert selected.connection_id == "emulator-5554"


def test_stopped_target_can_be_selected_for_lifecycle_preparation() -> None:
    emulator = _device("Pixel_8", "Pixel 8")

    selected = select_device(DeviceQuery(target_id="Pixel_8"), (emulator,))

    assert selected.readiness is DeviceReadiness.STARTABLE
    assert selected.connection_id is None


def test_exact_target_id_wins_over_another_display_name() -> None:
    target = _device("TARGET-A", "Pixel 9")
    name_collision = _device("TARGET-B", "TARGET-A")

    selected = select_device(DeviceQuery(target_id="TARGET-A"), (name_collision, target))

    assert selected is target


def test_platform_limits_a_shared_display_name() -> None:
    android = _device("ANDROID", "QA phone")
    ios = _device("IOS", "QA phone", platform=DevicePlatform.IOS)

    selected = select_device(
        DeviceQuery(platform=DevicePlatform.IOS, display_name="QA phone"),
        (android, ios),
    )

    assert selected is ios


def test_shared_display_name_is_ambiguous() -> None:
    devices = (_device("A", "QA phone"), _device("B", "QA phone"))

    with pytest.raises(AmbiguousDeviceError, match="A.*B"):
        select_device(DeviceQuery(display_name="QA phone"), devices)


def test_query_without_criteria_requires_one_available_target() -> None:
    devices = (_device("A", "First"), _device("B", "Second"))

    with pytest.raises(AmbiguousDeviceError, match="matched 2 devices"):
        select_device(DeviceQuery(), devices)


def test_missing_target_lists_available_candidates() -> None:
    devices = (_device("AVAILABLE", "Pixel 9"),)

    with pytest.raises(DeviceNotFoundError, match="AVAILABLE"):
        select_device(DeviceQuery(target_id="MISSING"), devices)


def test_all_query_criteria_must_match_the_same_target() -> None:
    devices = (_device("A", "First"), _device("B", "Second"))

    with pytest.raises(DeviceNotFoundError):
        select_device(DeviceQuery(target_id="A", display_name="Second"), devices)
