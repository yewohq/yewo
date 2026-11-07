"""Tests for canonical run record persistence."""

from datetime import datetime, timezone
import json

import pytest

from yewo.reporting import (
    CANONICAL_RUN_RECORD_PATH,
    CanonicalRunRecord,
    CanonicalRunRecordStore,
    CanonicalRunRecordWriteError,
)
from yewo.runs import RunMode, RunStage, RunTerminalCategory


def _record() -> CanonicalRunRecord:
    completed_at = datetime(2026, 11, 6, 9, tzinfo=timezone.utc)
    return CanonicalRunRecord(
        run_id="run_1",
        mode=RunMode.NORMAL,
        started_at=completed_at,
        completed_at=completed_at,
        terminal_category=RunTerminalCategory.COMPLETED,
        terminal_stage=RunStage.FINISHED,
        terminal_reason="The goal workflow completed.",
        partial_evidence_available=True,
    )


def test_store_persists_one_versioned_canonical_record(tmp_path) -> None:
    path = CanonicalRunRecordStore(tmp_path).persist(_record())

    payload = json.loads((tmp_path / path).read_text())
    assert path == CANONICAL_RUN_RECORD_PATH
    assert payload["schema_version"] == 1
    assert payload["run_id"] == "run_1"
    assert payload["terminal_category"] == "completed"
    assert payload["qa_verdict"] == "unverified"
    assert payload["completeness"] == "complete"
    restored = CanonicalRunRecord.model_validate(payload)
    assert restored == _record()


def test_store_never_replaces_a_canonical_record(tmp_path) -> None:
    store = CanonicalRunRecordStore(tmp_path)
    store.persist(_record())

    with pytest.raises(CanonicalRunRecordWriteError, match="Could not persist"):
        store.persist(_record())
