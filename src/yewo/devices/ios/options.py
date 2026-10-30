"""Build XCUITest options for a prepared iOS target."""

from __future__ import annotations

from appium.options.ios import XCUITestOptions

from yewo.applications.records import (
    ApplicationDescriptor,
    ApplicationInstallMode,
    ApplicationRequest,
)
from yewo.automation.driver_options import validate_driver_inputs
from yewo.devices.inventory import DeviceDescriptor, DevicePlatform


class IosDriverOptionsBuilder:
    """Translate validated records into XCUITest capabilities."""

    def __init__(
        self,
        *,
        new_command_timeout_seconds: int = 300,
        use_new_wda: bool = True,
    ) -> None:
        if new_command_timeout_seconds <= 0:
            raise ValueError("The new-command timeout must be greater than zero.")
        self._new_command_timeout_seconds = new_command_timeout_seconds
        self._use_new_wda = use_new_wda

    def build(
        self,
        device: DeviceDescriptor,
        request: ApplicationRequest,
        application: ApplicationDescriptor,
    ) -> XCUITestOptions:
        """Return options bound to one prepared iOS connection."""
        connection_id, artifact_path = validate_driver_inputs(
            DevicePlatform.IOS,
            device,
            request,
            application,
        )
        options = XCUITestOptions()
        options.device_name = device.display_name
        options.udid = connection_id
        options.new_command_timeout = self._new_command_timeout_seconds
        options.set_capability("appium:useNewWDA", self._use_new_wda)

        if request.install_mode is ApplicationInstallMode.REUSE:
            options.bundle_id = application.application_id
            options.no_reset = True
        else:
            options.app = str(artifact_path)
            options.set_capability(
                "appium:enforceAppInstall",
                request.install_mode is ApplicationInstallMode.INSTALL,
            )
        return options
