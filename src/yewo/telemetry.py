"""Anonymous installation and usage telemetry for the Yewo command line."""

from __future__ import annotations

import json
import os
import platform
import uuid
from collections.abc import Mapping
from enum import StrEnum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field


DEFAULT_PROJECT_TOKEN = "phc_sptcbP2oNYREzDfprVTwxKcSbnECXmjRCScmVHJ6Uj6U"
DEFAULT_INGESTION_HOST = "https://eu.i.posthog.com"

INGESTION_TIMEOUT_SECONDS = 2.0

_DISABLED_PREFERENCES = frozenset({"0", "false", "no", "off"})
_ENABLED_PREFERENCES = frozenset({"1", "true", "yes", "on"})


class TelemetryEvent(StrEnum):
    """An event name that Yewo reports to the product analytics project."""

    INSTALLED = "yewo_installed"
    RUN_STARTED = "yewo_run_started"
    RUN_FINISHED = "yewo_run_finished"


class InstallIdentity(BaseModel):
    """The random identifier that groups events from one Yewo installation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    install_id: str = Field(min_length=1)
    is_new_installation: bool


_install_identity: InstallIdentity | None = None


def record_new_installation() -> None:
    """Report the installation event on the first run of this installation."""
    if not _is_telemetry_enabled():
        return
    if not resolve_install_identity().is_new_installation:
        return
    record_event(TelemetryEvent.INSTALLED)


def record_event(event: TelemetryEvent, properties: Mapping[str, object] | None = None) -> None:
    """Report one anonymous usage event. A failed report never fails the run."""
    if not _is_telemetry_enabled():
        return
    payload = {
        "api_key": _resolve_project_token(),
        "event": event.value,
        "distinct_id": resolve_install_identity().install_id,
        "properties": {**_build_base_properties(), **(properties or {})},
    }
    _send_event(payload)


def resolve_install_identity() -> InstallIdentity:
    """Return the stored install identifier, or create one on first use."""
    global _install_identity

    if _install_identity is None:
        _install_identity = _build_install_identity()
    return _install_identity


def _build_install_identity() -> InstallIdentity:
    state_path = _resolve_state_path()
    stored_install_id = _read_stored_install_id(state_path)
    if stored_install_id is not None:
        return InstallIdentity(install_id=stored_install_id, is_new_installation=False)

    created_install_id = str(uuid.uuid4())
    _write_install_id(state_path, created_install_id)
    return InstallIdentity(install_id=created_install_id, is_new_installation=True)


def _read_stored_install_id(state_path: Path) -> str | None:
    try:
        stored_state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(stored_state, dict):
        return None
    stored_install_id = stored_state.get("install_id")
    if isinstance(stored_install_id, str) and stored_install_id:
        return stored_install_id
    return None


def _write_install_id(state_path: Path, install_id: str) -> None:
    # A read-only home directory must not fail a run. The identifier is then
    # regenerated on the next run, which overcounts installations.
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps({"install_id": install_id}), encoding="utf-8")
    except OSError:
        return


def _resolve_state_path() -> Path:
    configured_state_home = os.environ.get("XDG_STATE_HOME", "").strip()
    state_home = Path(configured_state_home) if configured_state_home else _default_state_home()
    return state_home / "yewo" / "telemetry.json"


def _default_state_home() -> Path:
    return Path.home() / ".local" / "state"


def _is_telemetry_enabled() -> bool:
    configured_preference = os.environ.get("YEWO_TELEMETRY", "").strip().lower()
    if configured_preference in _DISABLED_PREFERENCES:
        return False
    if configured_preference in _ENABLED_PREFERENCES:
        return True
    return not _is_source_checkout()


def _is_source_checkout() -> bool:
    repository_root = Path(__file__).resolve().parent.parent.parent
    return (repository_root / ".git").exists()


def _is_continuous_integration() -> bool:
    configured_indicator = os.environ.get("CI", "").strip().lower()
    return bool(configured_indicator) and configured_indicator not in _DISABLED_PREFERENCES


def _build_base_properties() -> dict[str, object]:
    return {
        "yewo_version": _resolve_yewo_version(),
        "python_version": platform.python_version(),
        "operating_system": platform.system(),
        "processor_architecture": platform.machine(),
        "is_continuous_integration": _is_continuous_integration(),
    }


def _resolve_yewo_version() -> str:
    try:
        return version("yewo")
    except PackageNotFoundError:
        return "unknown"


def _resolve_project_token() -> str:
    return os.environ.get("YEWO_TELEMETRY_TOKEN", "").strip() or DEFAULT_PROJECT_TOKEN


def _resolve_ingestion_host() -> str:
    return os.environ.get("YEWO_TELEMETRY_HOST", "").strip() or DEFAULT_INGESTION_HOST


def _send_event(payload: Mapping[str, object]) -> None:
    request = Request(
        f"{_resolve_ingestion_host().rstrip('/')}/i/v0/e/",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # Telemetry reports every transport and encoding failure as success. No
    # analytics failure may change the outcome of a run.
    try:
        with urlopen(request, timeout=INGESTION_TIMEOUT_SECONDS) as response:
            response.read()
    except Exception:  # noqa: BLE001
        return
