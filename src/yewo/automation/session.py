"""Compose device, Appium, driver, and application ownership."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Self

from appium.options.common.base import AppiumOptions

from yewo.applications.presence import ApplicationPresence
from yewo.applications.records import (
    ApplicationDescriptor,
    ApplicationInstallationChange,
    ApplicationInstallMode,
    ApplicationRequest,
)
from yewo.automation.appium_server import AppiumServerLease
from yewo.automation.driver_options import DriverOptionsBuilder
from yewo.automation.driver_session import DriverSessionLease, DriverSessionManager
from yewo.automation.errors import AutomationSessionError
from yewo.devices.inventory import DevicePlatform
from yewo.devices.lifecycle import DeviceLease
from yewo.resources import ResourceOwnership


class AutomationSession:
    """All runtime resources needed to automate one application."""

    def __init__(
        self,
        device: DeviceLease,
        server: AppiumServerLease,
        driver: DriverSessionLease,
        application: ApplicationDescriptor,
        installation_ownership: ResourceOwnership,
        installation_change: ApplicationInstallationChange,
        *,
        retain_installation: bool,
    ) -> None:
        self.device = device
        self.server = server
        self.driver = driver
        self.application = application
        self.installation_ownership = installation_ownership
        self.installation_change = installation_change
        self.retain_installation = retain_installation
        self._closed = False

    @property
    def closed(self) -> bool:
        """Return whether all cleanup stages were attempted."""
        return self._closed

    def close(self) -> None:
        """Release application, driver, server, and device in dependency order."""
        if self._closed:
            return
        errors: list[Exception] = []
        if self.installation_ownership is ResourceOwnership.RUN and not self.retain_installation:
            try:
                self.driver.driver.remove_app(self.application.application_id)
            except Exception as error:  # noqa: BLE001 - external driver boundary
                errors.append(error)
        for resource in (self.driver, self.server, self.device):
            try:
                resource.close()
            except Exception as error:  # noqa: BLE001 - cleanup must continue
                errors.append(error)
        self._closed = True
        if errors:
            failure = AutomationSessionError(
                f"Automation session cleanup failed in {len(errors)} stage(s)."
            )
            failure.add_note("; ".join(str(error) for error in errors))
            raise failure from errors[0]

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *unused: object) -> None:
        self.close()


class AutomationSessionManager:
    """Create one composed automation session with failure rollback."""

    def __init__(
        self,
        driver_sessions: DriverSessionManager,
        builders: Mapping[DevicePlatform, DriverOptionsBuilder[AppiumOptions]],
        presence: Mapping[DevicePlatform, ApplicationPresence],
    ) -> None:
        self._driver_sessions = driver_sessions
        self._builders = builders
        self._presence = presence

    @classmethod
    def default(cls) -> AutomationSessionManager:
        """Create standard Android and iOS adapters."""
        from yewo.devices.android.adb import AdbClient
        from yewo.devices.android.application import AndroidApplicationPresence
        from yewo.devices.android.options import AndroidDriverOptionsBuilder
        from yewo.devices.ios.application import IosApplicationPresence
        from yewo.devices.ios.devicectl import DevicectlClient
        from yewo.devices.ios.options import IosDriverOptionsBuilder
        from yewo.devices.ios.simctl import SimctlClient

        return cls(
            DriverSessionManager(),
            {
                DevicePlatform.ANDROID: AndroidDriverOptionsBuilder(),
                DevicePlatform.IOS: IosDriverOptionsBuilder(),
            },
            {
                DevicePlatform.ANDROID: AndroidApplicationPresence(AdbClient()),
                DevicePlatform.IOS: IosApplicationPresence(
                    SimctlClient(),
                    DevicectlClient(),
                ),
            },
        )

    def open(
        self,
        device: DeviceLease,
        server: AppiumServerLease,
        request: ApplicationRequest,
        application: ApplicationDescriptor,
    ) -> AutomationSession:
        """Open a driver and transfer the input leases into one session."""
        driver: DriverSessionLease | None = None
        try:
            if device.closed or server.closed:
                raise AutomationSessionError("Automation requires open device and server leases.")
            platform = device.descriptor.platform
            builder = self._builders.get(platform)
            presence = self._presence.get(platform)
            if builder is None or presence is None:
                raise AutomationSessionError(
                    f"No automation adapters are configured for {platform}."
                )
            installed_before = presence.is_installed(
                device.descriptor,
                application.application_id,
            )
            if (
                request.install_mode is ApplicationInstallMode.REUSE
                and installed_before is not True
            ):
                raise AutomationSessionError(
                    "Reuse requires positive evidence that the application is installed."
                )
            options = builder.build(device.descriptor, request, application)
            driver = self._driver_sessions.open(server, options)
            try:
                installed_after = driver.driver.is_app_installed(application.application_id)
            except Exception as inspection_error:  # noqa: BLE001 - external driver boundary
                raise AutomationSessionError(
                    "Could not verify the application after the driver opened."
                ) from inspection_error
            if not installed_after:
                raise AutomationSessionError(
                    "The application is not installed after the driver session opened."
                )
            installation_ownership = (
                ResourceOwnership.RUN if installed_before is False else ResourceOwnership.EXTERNAL
            )
            installation_change = _installation_change(request, installed_before)
            return AutomationSession(
                device,
                server,
                driver,
                application,
                installation_ownership,
                installation_change,
                retain_installation=request.retain_installation,
            )
        except BaseException as error:
            cleanup_errors = self._rollback(driver, server, device)
            if cleanup_errors:
                error.add_note(
                    "Rollback also failed: "
                    + "; ".join(str(cleanup_error) for cleanup_error in cleanup_errors)
                )
            raise

    @staticmethod
    def _rollback(
        driver: DriverSessionLease | None,
        server: AppiumServerLease,
        device: DeviceLease,
    ) -> list[Exception]:
        errors: list[Exception] = []
        for resource in (driver, server, device):
            if resource is None:
                continue
            try:
                resource.close()
            except Exception as error:  # noqa: BLE001 - rollback must continue
                errors.append(error)
        return errors


def _installation_change(
    request: ApplicationRequest,
    installed_before: bool | None,
) -> ApplicationInstallationChange:
    if request.install_mode is ApplicationInstallMode.REUSE:
        return ApplicationInstallationChange.NONE
    if installed_before is None:
        return ApplicationInstallationChange.UNVERIFIED
    if installed_before is False:
        return ApplicationInstallationChange.INSTALLED
    if request.install_mode is ApplicationInstallMode.INSTALL:
        return ApplicationInstallationChange.REINSTALLED
    return ApplicationInstallationChange.NONE
