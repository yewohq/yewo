"""Build UiAutomator2 options for a prepared Android target."""

from __future__ import annotations

from appium.options.android import UiAutomator2Options

from yewo.applications.records import (
    ApplicationDescriptor,
    ApplicationInstallMode,
    ApplicationRequest,
)
from yewo.automation.driver_options import validate_driver_inputs
from yewo.automation.errors import DriverOptionsError
from yewo.devices.inventory import DeviceDescriptor, DevicePlatform


class AndroidDriverOptionsBuilder:
    """Translate validated records into UiAutomator2 capabilities."""

    def __init__(self, *, new_command_timeout_seconds: int = 300) -> None:
        if new_command_timeout_seconds <= 0:
            raise ValueError("The new-command timeout must be greater than zero.")
        self._new_command_timeout_seconds = new_command_timeout_seconds

    def build(
        self,
        device: DeviceDescriptor,
        request: ApplicationRequest,
        application: ApplicationDescriptor,
    ) -> UiAutomator2Options:
        """Return options bound to one prepared Android connection."""
        connection_id, artifact_path = validate_driver_inputs(
            DevicePlatform.ANDROID,
            device,
            request,
            application,
        )
        launch_target = application.launch_target
        if launch_target is None:
            raise DriverOptionsError("Android driver options require a launch target.")

        options = UiAutomator2Options()
        options.device_name = device.display_name
        options.udid = connection_id
        options.new_command_timeout = self._new_command_timeout_seconds
        options.app_package = application.application_id
        options.app_activity = launch_target
        options.set_capability("appium:disableSuppressAccessibilityService", True)

        if request.install_mode is ApplicationInstallMode.REUSE:
            options.no_reset = True
        else:
            options.app = str(artifact_path)
            options.set_capability(
                "appium:enforceAppInstall",
                request.install_mode is ApplicationInstallMode.INSTALL,
            )
        return options
