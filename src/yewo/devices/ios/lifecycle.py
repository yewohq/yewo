"""Prepare iOS targets and preserve simulator ownership."""

from __future__ import annotations

from contextlib import suppress
from typing import Protocol

from yewo.devices.errors import DeviceCommandError, DeviceLifecycleError
from yewo.devices.inventory import (
    DeviceDescriptor,
    DeviceKind,
    DevicePlatform,
    DeviceReadiness,
)
from yewo.devices.lifecycle import DeviceLease
from yewo.resources import ResourceOwnership


class _IosSimulatorControl(Protocol):
    def boot(self, connection_id: str) -> None: ...

    def wait_until_booted(self, connection_id: str, *, timeout_seconds: float = 120) -> None: ...

    def shutdown(self, connection_id: str) -> None: ...


class IosDeviceLifecycle:
    """Prepare iOS hardware or a simulator for one run."""

    def __init__(
        self,
        simctl: _IosSimulatorControl,
        *,
        timeout_seconds: float = 120,
    ) -> None:
        self._simctl = simctl
        self._timeout_seconds = timeout_seconds

    def prepare(self, device: DeviceDescriptor) -> DeviceLease:
        """Return a ready iOS target and stop only a simulator started here."""
        self._validate(device)
        if device.kind is DeviceKind.PHYSICAL:
            return DeviceLease(device, ResourceOwnership.EXTERNAL)
        connection_id = device.connection_id
        if connection_id is None:
            raise DeviceLifecycleError("An iOS simulator requires a connection ID.")
        if device.readiness is DeviceReadiness.READY:
            return DeviceLease(device, ResourceOwnership.EXTERNAL)
        if device.readiness is DeviceReadiness.TRANSITIONING:
            ready = self._wait(device, connection_id)
            return DeviceLease(ready, ResourceOwnership.EXTERNAL)
        if device.readiness is not DeviceReadiness.STARTABLE:
            raise DeviceLifecycleError(f"iOS target '{device.target_id}' is {device.native_state}.")
        self._simctl.boot(connection_id)
        try:
            ready = self._wait(device, connection_id)
        except BaseException:
            self._shutdown_after_failed_start(connection_id)
            raise
        return DeviceLease(
            ready,
            ResourceOwnership.RUN,
            lambda: self._simctl.shutdown(connection_id),
        )

    @staticmethod
    def _validate(device: DeviceDescriptor) -> None:
        if device.platform is not DevicePlatform.IOS:
            raise DeviceLifecycleError("The iOS lifecycle requires an iOS target.")
        if device.kind is DeviceKind.PHYSICAL and device.readiness is not DeviceReadiness.READY:
            raise DeviceLifecycleError(
                f"Physical iOS target '{device.target_id}' is {device.native_state}."
            )

    def _wait(self, device: DeviceDescriptor, connection_id: str) -> DeviceDescriptor:
        try:
            self._simctl.wait_until_booted(
                connection_id,
                timeout_seconds=self._timeout_seconds,
            )
        except DeviceCommandError as error:
            raise DeviceLifecycleError(
                f"iOS target '{device.target_id}' did not finish startup."
            ) from error
        return device.model_copy(
            update={
                "readiness": DeviceReadiness.READY,
                "native_state": "Booted",
            }
        )

    def _shutdown_after_failed_start(self, connection_id: str) -> None:
        with suppress(DeviceCommandError):
            self._simctl.shutdown(connection_id)
