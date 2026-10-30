"""Platform-neutral contract and validation for Appium driver options."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, TypeVar

from appium.options.common.base import AppiumOptions

from yewo.applications.records import (
    ApplicationDescriptor,
    ApplicationInstallMode,
    ApplicationRequest,
)
from yewo.automation.errors import DriverOptionsError
from yewo.devices.inventory import DeviceDescriptor, DevicePlatform, DeviceReadiness


OptionsT_co = TypeVar("OptionsT_co", bound=AppiumOptions, covariant=True)


class DriverOptionsBuilder(Protocol[OptionsT_co]):
    """Build platform options without opening a driver session."""

    def build(
        self,
        device: DeviceDescriptor,
        request: ApplicationRequest,
        application: ApplicationDescriptor,
    ) -> OptionsT_co:
        """Return validated Appium options for the prepared target."""
        ...


def validate_driver_inputs(
    expected_platform: DevicePlatform,
    device: DeviceDescriptor,
    request: ApplicationRequest,
    application: ApplicationDescriptor,
) -> tuple[str, Path | None]:
    """Return the connection and installation artifact after shared validation."""
    if {
        device.platform,
        request.platform,
        application.platform,
    } != {expected_platform}:
        raise DriverOptionsError(
            f"Driver options require matching {expected_platform} device and application records."
        )
    if device.readiness is not DeviceReadiness.READY or device.connection_id is None:
        raise DriverOptionsError("Driver options require a prepared device connection.")
    if request.application_id and request.application_id != application.application_id:
        raise DriverOptionsError(
            "The requested application ID does not match the observed application ID."
        )
    if request.install_mode is ApplicationInstallMode.REUSE:
        return device.connection_id, None
    artifact_path = application.artifact_path
    if artifact_path is None:
        raise DriverOptionsError("Application installation requires an inspected artifact.")
    if request.artifact_path != artifact_path:
        raise DriverOptionsError(
            "The requested artifact does not match the inspected application artifact."
        )
    return device.connection_id, artifact_path
