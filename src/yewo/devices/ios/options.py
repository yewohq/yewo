"""Build XCUITest options for a prepared iOS target."""

from __future__ import annotations

from appium.options.ios import XCUITestOptions
from pydantic import BaseModel, ConfigDict, Field

from yewo.applications.records import (
    ApplicationDescriptor,
    ApplicationInstallMode,
    ApplicationRequest,
)
from yewo.automation.driver_options import (
    DEFAULT_DRIVER_IDLE_TIMEOUT_SECONDS,
    validate_driver_inputs,
)
from yewo.devices.inventory import DeviceDescriptor, DeviceKind, DevicePlatform


class IosWebDriverAgentSigning(BaseModel):
    """Signing identity used when Xcode builds WebDriverAgent for hardware."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    team_id: str = Field(pattern=r"^[A-Z0-9]{10}$")
    bundle_id: str = Field(pattern=r"^[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+$")
    signing_identity: str = Field(default="Apple Development", min_length=1)
    allow_device_registration: bool = False


class IosDriverOptionsBuilder:
    """Translate validated records into XCUITest capabilities."""

    def __init__(
        self,
        *,
        new_command_timeout_seconds: int = DEFAULT_DRIVER_IDLE_TIMEOUT_SECONDS,
        use_new_wda: bool = True,
        wda_signing: IosWebDriverAgentSigning | None = None,
        show_xcode_log: bool = False,
    ) -> None:
        if new_command_timeout_seconds <= 0:
            raise ValueError("The new-command timeout must be greater than zero.")
        self._new_command_timeout_seconds = new_command_timeout_seconds
        self._use_new_wda = use_new_wda
        self._wda_signing = wda_signing
        self._show_xcode_log = show_xcode_log

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
        if device.operating_system_version is not None:
            options.platform_version = device.operating_system_version
        options.new_command_timeout = self._new_command_timeout_seconds
        options.set_capability("appium:useNewWDA", self._use_new_wda)
        if device.kind is DeviceKind.PHYSICAL and self._wda_signing is not None:
            options.set_capability("appium:xcodeOrgId", self._wda_signing.team_id)
            options.set_capability(
                "appium:xcodeSigningId",
                self._wda_signing.signing_identity,
            )
            options.set_capability(
                "appium:updatedWDABundleId",
                self._wda_signing.bundle_id,
            )
            if self._wda_signing.allow_device_registration:
                options.set_capability(
                    "appium:allowProvisioningDeviceRegistration",
                    True,
                )
        if self._show_xcode_log:
            options.set_capability("appium:showXcodeLog", True)

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
