"""Tests for anonymous installation and usage telemetry."""

from collections.abc import Mapping
from pathlib import Path

import pytest

from yewo import telemetry
from yewo.telemetry import TelemetryEvent


@pytest.fixture(autouse=True)
def isolated_telemetry_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(telemetry, "_install_identity", None)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    monkeypatch.delenv("YEWO_TELEMETRY", raising=False)
    monkeypatch.delenv("CI", raising=False)


@pytest.fixture
def recorded_payloads(monkeypatch: pytest.MonkeyPatch) -> list[Mapping[str, object]]:
    """Collect payloads instead of sending them."""
    payloads: list[Mapping[str, object]] = []
    monkeypatch.setattr(telemetry, "_send_event", payloads.append)
    return payloads


def _enable_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enable telemetry while the suite replaces the transport with a recorder."""
    monkeypatch.setenv("YEWO_TELEMETRY", "1")


def test_install_identity_is_created_once_and_persists(tmp_path: Path) -> None:
    created = telemetry.resolve_install_identity()

    assert created.is_new_installation is True
    assert (tmp_path / "yewo" / "telemetry.json").exists()

    telemetry._install_identity = None
    reloaded = telemetry.resolve_install_identity()

    assert reloaded.install_id == created.install_id
    assert reloaded.is_new_installation is False


def test_unreadable_state_yields_a_fresh_install_identity(tmp_path: Path) -> None:
    state_path = tmp_path / "yewo" / "telemetry.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text("{ not json", encoding="utf-8")

    identity = telemetry.resolve_install_identity()

    assert identity.is_new_installation is True
    assert identity.install_id


def test_unwritable_state_directory_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "missing-root"))
    monkeypatch.setattr(
        telemetry.Path,
        "mkdir",
        lambda *_arguments, **_keywords: (_ for _ in ()).throw(OSError("read-only")),
    )

    identity = telemetry.resolve_install_identity()

    assert identity.install_id


def test_recorded_event_carries_the_install_id_and_base_properties(
    monkeypatch: pytest.MonkeyPatch,
    recorded_payloads: list[Mapping[str, object]],
) -> None:
    _enable_telemetry(monkeypatch)
    telemetry.record_event(TelemetryEvent.RUN_FINISHED, {"terminal_category": "completed"})

    assert len(recorded_payloads) == 1
    payload = recorded_payloads[0]
    assert payload["event"] == "yewo_run_finished"
    assert payload["distinct_id"] == telemetry.resolve_install_identity().install_id
    assert payload["api_key"] == telemetry.DEFAULT_PROJECT_TOKEN
    properties = payload["properties"]
    assert isinstance(properties, dict)
    assert properties["terminal_category"] == "completed"
    assert properties["is_continuous_integration"] is False
    assert properties["operating_system"]


def test_every_event_from_one_installation_shares_a_distinct_id(
    monkeypatch: pytest.MonkeyPatch,
    recorded_payloads: list[Mapping[str, object]],
) -> None:
    _enable_telemetry(monkeypatch)
    telemetry.record_event(TelemetryEvent.RUN_STARTED)
    telemetry.record_event(TelemetryEvent.RUN_FINISHED)

    distinct_ids = {payload["distinct_id"] for payload in recorded_payloads}
    assert len(distinct_ids) == 1


def test_installation_is_reported_only_on_the_first_run(
    monkeypatch: pytest.MonkeyPatch,
    recorded_payloads: list[Mapping[str, object]],
) -> None:
    _enable_telemetry(monkeypatch)
    telemetry.record_new_installation()
    telemetry._install_identity = None
    telemetry.record_new_installation()

    assert [payload["event"] for payload in recorded_payloads] == ["yewo_installed"]


def test_continuous_integration_runs_are_marked(
    monkeypatch: pytest.MonkeyPatch,
    recorded_payloads: list[Mapping[str, object]],
) -> None:
    _enable_telemetry(monkeypatch)
    monkeypatch.setenv("CI", "true")

    telemetry.record_event(TelemetryEvent.RUN_STARTED)

    properties = recorded_payloads[0]["properties"]
    assert isinstance(properties, dict)
    assert properties["is_continuous_integration"] is True


def test_disabled_preference_suppresses_every_event(
    monkeypatch: pytest.MonkeyPatch,
    recorded_payloads: list[Mapping[str, object]],
) -> None:
    _enable_telemetry(monkeypatch)
    monkeypatch.setenv("YEWO_TELEMETRY", "0")

    telemetry.record_new_installation()
    telemetry.record_event(TelemetryEvent.RUN_STARTED)

    assert recorded_payloads == []


def test_a_source_checkout_suppresses_every_event(
    monkeypatch: pytest.MonkeyPatch,
    recorded_payloads: list[Mapping[str, object]],
) -> None:
    _enable_telemetry(monkeypatch)
    monkeypatch.delenv("YEWO_TELEMETRY", raising=False)
    monkeypatch.setattr(telemetry, "_is_source_checkout", lambda: True)

    telemetry.record_event(TelemetryEvent.RUN_STARTED)

    assert recorded_payloads == []


def test_the_suite_runs_with_telemetry_off() -> None:
    """The boundary the suite relies on, asserted rather than assumed.

    Production no longer recognises a test runner, so nothing but this fixture keeps the suite
    off the network. A fixture that stopped applying would otherwise go unnoticed.
    """
    assert telemetry._is_telemetry_enabled() is False


def test_the_ingestion_target_is_the_configured_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("YEWO_TELEMETRY_HOST", "https://example.invalid/")
    monkeypatch.setenv("YEWO_TELEMETRY_TOKEN", "phc_override")

    assert telemetry._resolve_ingestion_host() == "https://example.invalid/"
    assert telemetry._resolve_project_token() == "phc_override"
