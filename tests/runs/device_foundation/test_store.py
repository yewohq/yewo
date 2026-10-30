"""Tests for device-foundation evidence persistence."""

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from yewo.runs.device_foundation import (
    DeviceFoundationActionKind,
    DeviceFoundationActionReceipt,
    DeviceFoundationActionStatus,
    DeviceFoundationStore,
)
from yewo.runs.errors import RunCoordinationError


CAPTURE_TIME = datetime(2026, 10, 30, tzinfo=timezone.utc)


def _receipt() -> DeviceFoundationActionReceipt:
    return DeviceFoundationActionReceipt(
        script_id="script_1",
        action_id="action_1",
        action_index=0,
        kind=DeviceFoundationActionKind.TAP,
        status=DeviceFoundationActionStatus.COMPLETED,
        before_observation_id="observation_1",
        after_observation_id="observation_2",
        started_at=CAPTURE_TIME,
        completed_at=CAPTURE_TIME,
    )


def test_store_requires_existing_observation_evidence(tmp_path: Path) -> None:
    with pytest.raises(RunCoordinationError, match="missing observation"):
        DeviceFoundationStore(tmp_path).persist_action(_receipt())


def test_store_persists_an_ordered_payload_free_receipt(tmp_path: Path) -> None:
    observations = tmp_path / "assets/observations"
    observations.mkdir(parents=True)
    for observation_id in ("observation_1", "observation_2"):
        (observations / f"{observation_id}.json").write_text("{}")

    path = DeviceFoundationStore(tmp_path).persist_action(_receipt())

    payload = json.loads((tmp_path / path).read_text())
    assert path == Path("assets/actions/000000-action_1.json")
    assert payload["before_observation_id"] == "observation_1"
    assert "text" not in payload
