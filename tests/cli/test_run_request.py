"""Tests for CLI-to-run-request translation."""

import argparse
from pathlib import Path

import pytest

from yewo.applications import ApplicationInstallMode
from yewo.cli.run_request import add_run_arguments, run_request_from_namespace
from yewo.devices import DevicePlatform


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_run_arguments(parser)
    return parser


def test_artifact_arguments_create_an_install_request() -> None:
    values = _parser().parse_args(
        [
            "--output-root",
            "runs",
            "--script",
            "workflow.json",
            "--platform",
            "android",
            "--device-id",
            "Pixel_8",
            "--app",
            "application.apk",
        ]
    )

    request = run_request_from_namespace(values)

    assert request.output_root == Path("runs")
    assert request.workflow_input_path == Path("workflow.json")
    assert request.device.platform is DevicePlatform.ANDROID
    assert request.device.target_id == "Pixel_8"
    assert request.application.install_mode is ApplicationInstallMode.INSTALL
    assert request.application.artifact_path == Path("application.apk")


def test_application_id_arguments_create_a_reuse_request() -> None:
    values = _parser().parse_args(
        [
            "--output-root",
            "runs",
            "--script",
            "workflow.json",
            "--platform",
            "ios",
            "--device-name",
            "My iPhone",
            "--application-id",
            "com.example.app",
            "--no-start-appium",
        ]
    )

    request = run_request_from_namespace(values)

    assert request.application.install_mode is ApplicationInstallMode.REUSE
    assert request.application.application_id == "com.example.app"
    assert not request.automation_server.start_if_unavailable


def test_cli_requires_one_device_selector() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(
            [
                "--output-root",
                "runs",
                "--script",
                "workflow.json",
                "--platform",
                "android",
                "--application-id",
                "com.example.app",
            ]
        )


def test_cli_rejects_a_request_without_an_application_source() -> None:
    values = _parser().parse_args(
        [
            "--output-root",
            "runs",
            "--script",
            "workflow.json",
            "--platform",
            "android",
            "--device-id",
            "Pixel_8",
        ]
    )

    with pytest.raises(argparse.ArgumentTypeError, match="artifact path"):
        run_request_from_namespace(values)
