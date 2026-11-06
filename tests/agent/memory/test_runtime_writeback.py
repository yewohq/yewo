"""Terminal AppMemory write-back isolation and artifact tests."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from yewo.agent import AgentResult
from yewo.agent.discovery.records import (
    DiscoveryCoverage,
    DiscoveryCoverageEntry,
    DiscoveryObjective,
    DiscoveryObjectiveState,
    DiscoveryRequest,
    DiscoveryResult,
)
from yewo.agent.memory.runtime_writeback import (
    MEMORY_WRITEBACK_REPORT_PATH,
    MemoryWritebackDisposition,
    MemoryWritebackReport,
    write_back_discovery_knowledge,
)
from yewo.agent.memory.store import AppMemoryStore, AppMemoryStoreError
from yewo.agent.replay.procedure.records import ProcedureReplayResult, ProcedureReplayTerminal
from yewo.artifacts import EvidenceApplicationIdentity
from yewo.observations import ObservationStore

from agent.discovery.support import agent_result, observation
from agent.memory.test_store import _memory, _png, _write_screenshot


_NOW = datetime(2026, 11, 3, 12, tzinfo=timezone.utc)


def _run_evidence(root: Path):
    png = _png()
    digest = sha256(png).hexdigest()
    before = _write_screenshot(
        root,
        observation("obs_1", "0000000000000000"),
        png,
        digest,
    )
    after = _write_screenshot(
        root,
        observation(
            "obs_2",
            "1111111111111111",
            predecessor_observation_id="obs_1",
            offset_seconds=1,
        ),
        png,
        digest,
    )
    ObservationStore(root).persist(before)
    ObservationStore(root).persist(after)
    evidence = agent_result(before, after)
    request = DiscoveryRequest(
        request_id="request_2",
        objectives=(
            DiscoveryObjective(
                objective_id="record_form",
                statement="The record form is reachable.",
            ),
        ),
    )
    return DiscoveryResult(
        request=request,
        terminal_reason=evidence.terminal_reason,
        coverage=DiscoveryCoverage(
            entries=(
                DiscoveryCoverageEntry(
                    objective_id="record_form",
                    state=DiscoveryObjectiveState.UNRESOLVED,
                    reason="The terminal test keeps coverage independent from write-back.",
                ),
            )
        ),
        agent_result=evidence,
    )


def _with_procedure_replay(
    result: DiscoveryResult,
    replay_evidence: AgentResult,
    application: EvidenceApplicationIdentity,
) -> DiscoveryResult:
    replay = ProcedureReplayResult(
        procedure_id="procedure_1",
        evidence_id="evidence_1",
        source_run_id="run_1",
        setup_id="attempt_1",
        application=application,
        terminal=ProcedureReplayTerminal.BUDGET_EXHAUSTED,
        procedure_step_count=1,
        completed_step_count=1,
        reason="The controlled replay stopped after its retained step.",
        agent_result=replay_evidence,
    )
    return result.model_copy(update={"procedure_replay": replay})


def test_terminal_writeback_applies_timing_and_persists_report(tmp_path: Path) -> None:
    compatibility, memory = _memory(tmp_path)
    store = AppMemoryStore(tmp_path / "shared")
    store.save(memory, tmp_path)
    result = _run_evidence(tmp_path)

    report = write_back_discovery_knowledge(
        tmp_path,
        "run_2",
        result,
        store,
        compatibility,
        clock=lambda: _NOW,
    )

    loaded = store.load(compatibility)
    assert loaded is not None
    assert report.disposition is MemoryWritebackDisposition.APPLIED
    assert report.timing_count == 1
    assert loaded.source_run_ids == ("run_1", "run_2")
    assert loaded.app_knowledge.timing[0].samples[0].duration_ms == 1_000
    persisted = MemoryWritebackReport.model_validate_json(
        (tmp_path / MEMORY_WRITEBACK_REPORT_PATH).read_bytes()
    )
    assert persisted == report


def test_duplicate_terminal_writeback_reports_skipped(tmp_path: Path) -> None:
    compatibility, memory = _memory(tmp_path)
    store = AppMemoryStore(tmp_path / "shared")
    store.save(memory, tmp_path)
    result = _run_evidence(tmp_path)

    first = write_back_discovery_knowledge(
        tmp_path,
        "run_2",
        result,
        store,
        compatibility,
        clock=lambda: _NOW,
    )
    repeated = write_back_discovery_knowledge(
        tmp_path,
        "run_2",
        result,
        store,
        compatibility,
        clock=lambda: _NOW,
    )

    assert first.disposition is MemoryWritebackDisposition.APPLIED
    assert repeated.disposition is MemoryWritebackDisposition.SKIPPED
    assert repeated.timing_count == 0
    assert repeated.reason == "Yewo already applied this write-back."


def test_duplicate_writeback_does_not_persist_new_replay_graph(tmp_path: Path) -> None:
    compatibility, memory = _memory(tmp_path)
    store = AppMemoryStore(tmp_path / "shared")
    store.save(memory, tmp_path)
    result = _run_evidence(tmp_path)
    first_result = _with_procedure_replay(
        result,
        result.agent_result,
        memory.compatibility.application,
    )

    first = write_back_discovery_knowledge(
        tmp_path,
        "run_2",
        first_result,
        store,
        compatibility,
        clock=lambda: _NOW,
    )
    retained = store.load(compatibility)
    assert retained is not None

    png = _png()
    digest = sha256(png).hexdigest()
    before = _write_screenshot(
        tmp_path,
        observation("fresh_obs_1", "2222222222222222"),
        png,
        digest,
    )
    after = _write_screenshot(
        tmp_path,
        observation(
            "fresh_obs_2",
            "3333333333333333",
            predecessor_observation_id=before.observation_id,
            offset_seconds=1,
        ),
        png,
        digest,
    )
    ObservationStore(tmp_path).persist(before)
    ObservationStore(tmp_path).persist(after)
    changed_result = _with_procedure_replay(
        result,
        agent_result(before, after),
        memory.compatibility.application,
    )

    repeated = write_back_discovery_knowledge(
        tmp_path,
        "run_2",
        changed_result,
        store,
        compatibility,
        clock=lambda: _NOW,
    )

    assert first.disposition is MemoryWritebackDisposition.APPLIED
    assert repeated.disposition is MemoryWritebackDisposition.SKIPPED
    assert store.load(compatibility) == retained


def test_writeback_failure_is_reported_without_raising(
    tmp_path: Path,
    monkeypatch,
) -> None:
    compatibility, memory = _memory(tmp_path)
    store = AppMemoryStore(tmp_path / "shared")
    store.save(memory, tmp_path)
    result = _run_evidence(tmp_path)

    def fail_update(*args, **kwargs):
        del args, kwargs
        raise AppMemoryStoreError("Injected write-back failure.")

    monkeypatch.setattr(store, "update", fail_update)

    report = write_back_discovery_knowledge(
        tmp_path,
        "run_2",
        result,
        store,
        compatibility,
        clock=lambda: _NOW,
    )

    assert report.disposition is MemoryWritebackDisposition.FAILED
    assert report.reason == "AppMemory write-back failed: AppMemoryStoreError."
    persisted = json.loads((tmp_path / MEMORY_WRITEBACK_REPORT_PATH).read_bytes())
    assert persisted["disposition"] == "failed"
