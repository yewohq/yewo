"""Translate validated command-line values into a run request."""

from __future__ import annotations

import argparse
from pathlib import Path

from yewo.applications import ApplicationInstallMode, ApplicationRequest
from yewo.devices import DevicePlatform, DeviceQuery
from yewo.runs import AutomationServerRequest, RunMode, RunRequest


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Add only arguments consumed by the current run-coordinator boundary."""
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--platform", choices=tuple(DevicePlatform), required=True)
    device = parser.add_mutually_exclusive_group(required=True)
    device.add_argument("--device-id")
    device.add_argument("--device-name")
    parser.add_argument("--app", type=Path)
    parser.add_argument("--application-id")
    parser.add_argument(
        "--install-mode",
        choices=tuple(ApplicationInstallMode),
    )
    parser.add_argument("--retain-installation", action="store_true")
    parser.add_argument("--appium-url", default="http://127.0.0.1:4723")
    parser.add_argument("--no-start-appium", action="store_true")


def run_request_from_namespace(values: argparse.Namespace) -> RunRequest:
    """Build one typed request or raise an argument error for an invalid source."""
    if values.app is None and values.application_id is None:
        raise argparse.ArgumentTypeError(
            "An application artifact path or application ID is required."
        )
    platform = DevicePlatform(values.platform)
    install_mode = _install_mode(values)
    try:
        return RunRequest(
            mode=RunMode.DEVICE_FOUNDATION,
            output_root=values.output_root,
            workflow_input_path=values.script,
            device=DeviceQuery(
                platform=platform,
                target_id=values.device_id,
                display_name=values.device_name,
            ),
            application=ApplicationRequest(
                platform=platform,
                install_mode=install_mode,
                artifact_path=values.app,
                application_id=values.application_id,
                retain_installation=values.retain_installation,
            ),
            automation_server=AutomationServerRequest(
                server_url=values.appium_url,
                start_if_unavailable=not values.no_start_appium,
            ),
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _install_mode(values: argparse.Namespace) -> ApplicationInstallMode:
    if values.install_mode is not None:
        return ApplicationInstallMode(values.install_mode)
    if values.app is not None:
        return ApplicationInstallMode.INSTALL
    return ApplicationInstallMode.REUSE
