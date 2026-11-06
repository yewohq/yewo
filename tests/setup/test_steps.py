"""Tests for previewable Setup mutations."""

from pathlib import Path
import subprocess

import pytest

from yewo.automation.appium_runtime import AppiumRuntime
from yewo.setup import SetupContext, SetupStatus, SetupStep, run_setup
from yewo.setup.steps import _appium_step, _driver_step


def _replace_tool_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "yewo.setup.steps._appium_step",
        lambda context, environment: (
            SetupStep(name="Appium", status=SetupStatus.OK, detail="ready"),
            None,
        ),
    )
    monkeypatch.setattr(
        "yewo.setup.steps._driver_step",
        lambda context, platform, runtime: SetupStep(
            name=f"driver: {platform}",
            status=SetupStatus.OK,
            detail="ready",
        ),
    )


def test_creates_a_starter_yaml_and_env_without_overwriting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _replace_tool_steps(monkeypatch)
    context = SetupContext(
        root=tmp_path,
        requested_platform="android",
        dry_run=False,
        assume_yes=True,
        confirm=lambda prompt: False,
    )

    report = run_setup(context)

    yaml = (tmp_path / "yewo.yaml").read_text()
    env = (tmp_path / ".env").read_text()
    assert "platform: android" in yaml
    assert "protocol: source_pixels" in yaml
    assert "# OPENROUTER_API_KEY=" in env
    assert (tmp_path / ".gitignore").read_text() == ".env\n"
    assert report.failures == 0

    (tmp_path / "yewo.yaml").write_text("platform: ios\n")
    (tmp_path / ".env").write_text("KEEP=1\n")
    run_setup(context)
    assert (tmp_path / "yewo.yaml").read_text() == "platform: ios\n"
    assert (tmp_path / ".env").read_text() == "KEEP=1\n"


def test_dry_run_previews_files_without_writing_them(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _replace_tool_steps(monkeypatch)

    report = run_setup(
        SetupContext(
            root=tmp_path,
            requested_platform="ios",
            dry_run=True,
            assume_yes=False,
            confirm=lambda prompt: pytest.fail("Dry-run must not prompt."),
        )
    )

    assert not (tmp_path / "yewo.yaml").exists()
    assert not (tmp_path / ".env").exists()
    assert not (tmp_path / ".gitignore").exists()
    assert [step.status for step in report.steps[:3]] == [
        SetupStatus.PLANNED,
        SetupStatus.PLANNED,
        SetupStatus.PLANNED,
    ]


def test_driver_install_uses_the_verified_node_and_appium_pair(tmp_path: Path) -> None:
    runtime = AppiumRuntime(
        node_executable=tmp_path / "node",
        appium_script=tmp_path / "appium.js",
        appium_version="3.1.0",
        environment={},
    )
    commands: list[tuple[str, ...]] = []

    def runner(command, *, timeout_seconds, capture_output):
        commands.append(tuple(command))
        stdout = '{"uiautomator2":{"installed":false}}' if capture_output else ""
        return subprocess.CompletedProcess(command, 0, stdout, "")

    step = _driver_step(
        SetupContext(
            root=tmp_path,
            requested_platform="android",
            dry_run=False,
            assume_yes=True,
            confirm=lambda prompt: False,
            runner=runner,
        ),
        "android",
        runtime,
    )

    assert step.status is SetupStatus.OK
    assert commands == [
        (
            str(runtime.node_executable),
            str(runtime.appium_script),
            "driver",
            "list",
            "--installed",
            "--json",
        ),
        (
            str(runtime.node_executable),
            str(runtime.appium_script),
            "driver",
            "install",
            "uiautomator2",
        ),
    ]


def test_setup_preserves_gitignore_content_when_adding_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _replace_tool_steps(monkeypatch)
    (tmp_path / ".gitignore").write_text("build/\n", encoding="utf-8")

    run_setup(
        SetupContext(
            root=tmp_path,
            requested_platform="android",
            dry_run=False,
            assume_yes=True,
            confirm=lambda prompt: False,
        )
    )

    assert (tmp_path / ".gitignore").read_text() == "build/\n.env\n"


def test_appium_installer_timeout_is_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("yewo.setup.steps._resolved_runtime", lambda environment: None)
    monkeypatch.setattr("yewo.setup.steps.shutil.which", lambda name, path=None: "/usr/bin/npm")

    def runner(command, *, timeout_seconds, capture_output):
        raise subprocess.TimeoutExpired(command, timeout_seconds)

    step, runtime = _appium_step(
        SetupContext(
            root=tmp_path,
            requested_platform="android",
            dry_run=False,
            assume_yes=True,
            confirm=lambda prompt: False,
            runner=runner,
        ),
        {"PATH": "/usr/bin"},
    )

    assert runtime is None
    assert step.status is SetupStatus.FAIL
    assert "timed out" in step.detail
