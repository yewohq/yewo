"""Production composition for run-coordinator platform boundaries."""

from __future__ import annotations

from collections.abc import Mapping

from yewo.artifacts import RUNTIME_DIRECTORY
from yewo.applications import RequestedApplicationResolver
from yewo.automation import AppiumServerConfig, AppiumServerManager, AutomationSessionManager
from yewo.applications.inspection import ApplicationArtifactInspector
from yewo.devices import DeviceInventory, DeviceLifecycle, DevicePlatform
from yewo.devices.android import (
    AdbDeviceInventory,
    AndroidDeviceLifecycle,
    MacOsAndroidEmulatorEnvironment,
)
from yewo.devices.android.adb import AdbClient
from yewo.devices.android.apk import ApkInspector
from yewo.devices.ios import (
    IosApplicationInspector,
    IosDeviceInventory,
    IosDeviceLifecycle,
    IosPhysicalDeviceInventory,
    SimctlDeviceInventory,
)
from yewo.devices.ios.simctl import SimctlClient
from yewo.devices.logs import DeviceLogTextScrubber
from yewo.host import (
    supports_android_emulator_clipboard_control,
    supports_apple_device_tooling,
)
from yewo.runs.coordinator import RunCoordinator, RunWorkflow
from yewo.runs.records import RunMode
from yewo.runs.store import RunWorkspace, RunWorkspaceFactory


def build_default_run_coordinator(
    workflows: Mapping[RunMode, RunWorkflow],
    scrub_device_log_text: DeviceLogTextScrubber,
) -> RunCoordinator:
    """Compose standard host adapters without discovering or changing devices."""
    android_inventory = AdbDeviceInventory()
    android_control = AdbClient()
    android_lifecycle = AndroidDeviceLifecycle(
        android_inventory,
        android_control,
        host_environment=(
            MacOsAndroidEmulatorEnvironment()
            if supports_android_emulator_clipboard_control()
            else None
        ),
    )
    inventories: dict[DevicePlatform, DeviceInventory] = {
        DevicePlatform.ANDROID: android_inventory
    }
    lifecycles: dict[DevicePlatform, DeviceLifecycle] = {
        DevicePlatform.ANDROID: android_lifecycle
    }
    inspectors: dict[DevicePlatform, ApplicationArtifactInspector] = {
        DevicePlatform.ANDROID: ApkInspector(android_control)
    }
    if supports_apple_device_tooling():
        inventories[DevicePlatform.IOS] = IosDeviceInventory(
            SimctlDeviceInventory(),
            IosPhysicalDeviceInventory(),
        )
        lifecycles[DevicePlatform.IOS] = IosDeviceLifecycle(SimctlClient())
        inspectors[DevicePlatform.IOS] = IosApplicationInspector()
    return RunCoordinator(
        RunWorkspaceFactory(),
        inventories,
        lifecycles,
        RequestedApplicationResolver(inspectors),
        AppiumServerManager(),
        AutomationSessionManager.default(),
        workflows,
        _server_config,
        scrub_device_log_text,
    )


def _server_config(workspace: RunWorkspace) -> AppiumServerConfig:
    settings = workspace.start.request.automation_server
    return AppiumServerConfig(
        log_path=workspace.root / RUNTIME_DIRECTORY / "appium.log",
        server_url=settings.server_url,
        start_if_unavailable=settings.start_if_unavailable,
        startup_timeout_seconds=settings.startup_timeout_seconds,
        probe_timeout_seconds=settings.probe_timeout_seconds,
        poll_interval_seconds=settings.poll_interval_seconds,
    )
