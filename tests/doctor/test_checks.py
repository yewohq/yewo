"""Tests for read-only Doctor aggregation."""

from pathlib import Path
import subprocess

import pytest

from yewo.doctor import DoctorCheck, DoctorStatus, run_doctor
from yewo.doctor.checks import _avd_check


def _ok(name: str) -> tuple[DoctorCheck, ...]:
    return (DoctorCheck(name=name, status=DoctorStatus.OK, detail="ready"),)


def _replace_environment_checks(
    monkeypatch: pytest.MonkeyPatch,
    visited: list[str],
) -> None:
    monkeypatch.setattr("yewo.doctor.checks._system_checks", lambda start: _ok("system"))
    monkeypatch.setattr(
        "yewo.doctor.checks._appium_checks",
        lambda platforms, runner, environment: _ok("appium"),
    )
    monkeypatch.setattr(
        "yewo.doctor.checks._android_checks",
        lambda device, runner, environment: visited.append("android") or _ok("android"),
    )
    monkeypatch.setattr(
        "yewo.doctor.checks._ios_checks",
        lambda device, runner: visited.append("ios") or _ok("ios"),
    )


def test_uses_project_platform_and_checks_every_configured_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = tmp_path / "app.apk"
    app.write_bytes(b"apk")
    (tmp_path / "yewo.yaml").write_text(
        """
device: Pixel_8
platform: android
app: app.apk
model: openrouter/example/semantic
grounder:
  model: openrouter/example/grounder
  protocol: source_pixels
"""
    )
    visited: list[str] = []
    checked_models: list[str] = []
    _replace_environment_checks(monkeypatch, visited)

    report = run_doctor(
        tmp_path,
        credential_check=lambda profile: checked_models.append(profile.model) or (),
    )

    assert report.blocking == 0
    assert visited == ["android"]
    assert checked_models == [
        "openrouter/example/semantic",
        "openrouter/example/grounder",
    ]
    project_checks = report.sections[0].checks
    assert [check.name for check in project_checks] == [
        "yewo.yaml",
        "application artifact",
        "run configuration",
        "model credentials: openrouter/example/semantic",
        "model credentials: openrouter/example/grounder",
    ]


def test_reports_missing_project_and_honours_an_explicit_platform(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    visited: list[str] = []
    _replace_environment_checks(monkeypatch, visited)

    report = run_doctor(tmp_path, "ios")

    assert report.blocking == 1
    assert visited == ["ios"]
    assert "No yewo.yaml was found" in report.sections[0].checks[0].detail


def test_missing_project_defaults_to_the_same_android_platform_as_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    visited: list[str] = []
    _replace_environment_checks(monkeypatch, visited)

    run_doctor(tmp_path)

    assert visited == ["android"]


def test_project_inspection_contains_an_unexpected_loader_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    visited: list[str] = []
    _replace_environment_checks(monkeypatch, visited)
    monkeypatch.setattr(
        "yewo.doctor.checks.load_nearest_yewo_project",
        lambda start: (_ for _ in ()).throw(RuntimeError("private detail")),
    )

    report = run_doctor(tmp_path)

    project_check = report.sections[0].checks[0]
    assert project_check.status is DoctorStatus.FAIL
    assert project_check.detail == "could not inspect the project: RuntimeError"


def test_real_credential_check_rejects_a_blank_openrouter_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = tmp_path / "app.apk"
    app.write_bytes(b"apk")
    (tmp_path / "yewo.yaml").write_text(
        """
device: Pixel_8
platform: android
app: app.apk
model: openrouter/example/semantic
grounder:
  model: openrouter/example/grounder
  protocol: source_pixels
"""
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    _replace_environment_checks(monkeypatch, [])

    report = run_doctor(tmp_path)

    credential_checks = tuple(
        check for check in report.sections[0].checks if check.name.startswith("model credentials:")
    )
    assert credential_checks
    assert all(check.status is DoctorStatus.FAIL for check in credential_checks)


def test_missing_application_artifact_is_blocking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "yewo.yaml").write_text(
        """
device: Pixel_8
platform: android
app: missing.apk
model: openrouter/example/semantic
grounder:
  model: openrouter/example/grounder
  protocol: source_pixels
"""
    )
    _replace_environment_checks(monkeypatch, [])

    report = run_doctor(tmp_path, credential_check=lambda profile: ())

    artifact = next(
        check for check in report.sections[0].checks if check.name == "application artifact"
    )
    assert artifact.status is DoctorStatus.FAIL
    assert report.blocking == 1


def test_avd_probe_does_not_contact_or_start_adb() -> None:
    commands: list[tuple[str, ...]] = []

    def runner(command, *, timeout_seconds):
        commands.append(tuple(command))
        return subprocess.CompletedProcess(command, 0, "Pixel_8\n", "")

    result = _avd_check(Path("/sdk/emulator/emulator"), "Pixel_8", runner)

    assert result.status is DoctorStatus.OK
    assert commands == [("/sdk/emulator/emulator", "-list-avds")]
