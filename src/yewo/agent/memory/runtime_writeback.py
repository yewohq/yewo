"""Isolated terminal write-back from current-run evidence into AppMemory."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from enum import StrEnum
from functools import partial
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from yewo.agent.discovery.records import DiscoveryResult
from yewo.agent.discovery.promotion import AppMemoryBuilder
from yewo.agent.discovery.replay_handoff import replay_observation_ids
from yewo.agent.memory.grounding_capture import build_grounding_assessments
from yewo.agent.memory.knowledge_capture import build_memory_delta
from yewo.agent.memory.records import AppMemory, MemoryCompatibility
from yewo.agent.memory.store import AppMemoryStore, load_discovery_memory
from yewo.agent.memory.writeback import (
    MemoryDelta,
    MemoryDeltaApplication,
    MemoryDeltaDisposition,
    MemoryWritebackSkip,
    apply_memory_delta_result,
)
from yewo.agent.recovery.advisor import (
    RECOVERY_ASSESSMENTS_PATH,
    RecoveryAssessment,
)
from yewo.agent.replay.procedure.store import CompiledProcedureStore, StoredProcedureBundle
from yewo.agent.replay.reconciliation.records import ReconcilerResponse
from yewo.agent.replay.reconciliation.store import RECONCILER_RESPONSE_PATH
from yewo.artifacts import (
    ARTIFACTS_DIRECTORY,
    ArtifactWriteError,
    AtomicArtifactWriter,
    RecordIdentifier,
)
from yewo.agent.runtime.records import AgentResult
from yewo.agent.screen_analysis import CurrentScreenAnalysis, ScreenAnalysisStore
from yewo.observations import Observation, ObservationStore
from yewo.run_narration import current_narrator


MEMORY_WRITEBACK_REPORT_PATH = ARTIFACTS_DIRECTORY / "agent" / "memory-writeback.json"


class MemoryWritebackDisposition(StrEnum):
    """Terminal outcome of one isolated knowledge write-back attempt."""

    APPLIED = "applied"
    SKIPPED = "skipped"
    FAILED = "failed"


class MemoryWritebackReport(BaseModel):
    """Safe run artifact describing one post-result memory update."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    source_run_id: RecordIdentifier
    disposition: MemoryWritebackDisposition
    writeback_id: RecordIdentifier | None = None
    memory_id: RecordIdentifier | None = None
    grounding_assessment_count: int = Field(default=0, ge=0)
    action_risk_count: int = Field(default=0, ge=0)
    rule_count: int = Field(default=0, ge=0)
    finding_count: int = Field(default=0, ge=0)
    timing_count: int = Field(default=0, ge=0)
    run_outcome_count: int = Field(default=0, ge=0)
    screen_summary_count: int = Field(default=0, ge=0)
    control_count: int = Field(default=0, ge=0)
    claim_review_count: int = Field(default=0, ge=0)
    skipped_records: tuple[MemoryWritebackSkip, ...] = Field(default=(), max_length=1_000)
    reason: str = Field(min_length=1, max_length=1_000)


def write_back_discovery_knowledge(
    root: Path,
    source_run_id: RecordIdentifier,
    result: DiscoveryResult,
    memory_store: AppMemoryStore,
    compatibility: MemoryCompatibility,
    procedure_store: CompiledProcedureStore | None = None,
    *,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> MemoryWritebackReport:
    """Update memory after the run result is durable. Preserve the run terminal."""
    try:
        report = _apply_writeback(
            root,
            source_run_id,
            result,
            memory_store,
            compatibility,
            procedure_store,
            clock(),
        )
    except Exception as error:  # noqa: BLE001 - terminal isolation boundary
        report = MemoryWritebackReport(
            source_run_id=source_run_id,
            disposition=MemoryWritebackDisposition.FAILED,
            reason=f"AppMemory write-back failed: {type(error).__name__}.",
        )
    _persist_report(root, report)
    _narrate_report(report)
    return report


def _apply_writeback(
    root: Path,
    source_run_id: RecordIdentifier,
    result: DiscoveryResult,
    memory_store: AppMemoryStore,
    compatibility: MemoryCompatibility,
    procedure_store: CompiledProcedureStore | None,
    applied_at: datetime,
) -> MemoryWritebackReport:
    memory = load_discovery_memory(memory_store, compatibility)
    if memory is None:
        return MemoryWritebackReport(
            source_run_id=source_run_id,
            disposition=MemoryWritebackDisposition.SKIPPED,
            reason="No compatible AppMemory snapshot was available.",
        )
    replay = result.procedure_replay
    observation_ids = replay_observation_ids(result.agent_result, replay)
    observations = ObservationStore(root).load_selected(observation_ids)
    observations_by_id = {item.observation_id: item for item in observations}
    bundle = _load_replayed_procedure(result, compatibility, procedure_store)
    combined_steps = tuple(
        {
            step.step_id: step
            for step in (
                *(replay.agent_result.steps if replay is not None else ()),
                *result.agent_result.steps,
            )
        }.values()
    )
    combined_analysis_ids = tuple(
        dict.fromkeys(
            (
                *(replay.agent_result.screen_analysis_ids if replay is not None else ()),
                *result.agent_result.screen_analysis_ids,
            )
        )
    )
    combined_result = result.agent_result.model_copy(
        update={
            "steps": combined_steps,
            "screen_analysis_ids": combined_analysis_ids,
        }
    )
    screen_analyses = ScreenAnalysisStore(root).load_selected(combined_analysis_ids)
    recovery_assessments = _load_recovery_assessments(root)
    reconciler_response = _load_reconciler_response(root)
    applied_delta: list[MemoryDelta] = []
    delta_applications: list[MemoryDeltaApplication] = []

    updated = memory_store.update(
        compatibility,
        partial(
            _update_memory,
            discovery_result=result,
            source_run_id=source_run_id,
            compatibility=compatibility,
            observations_by_id=observations_by_id,
            combined_result=combined_result,
            screen_analyses=screen_analyses,
            bundle=bundle,
            recovery_assessments=recovery_assessments,
            reconciler_response=reconciler_response,
            applied_at=applied_at,
            applied_deltas=applied_delta,
            delta_applications=delta_applications,
        ),
    )
    delta = applied_delta[0]
    application = delta_applications[0]
    if application.disposition is not MemoryDeltaDisposition.APPLIED:
        return MemoryWritebackReport(
            source_run_id=source_run_id,
            disposition=MemoryWritebackDisposition.SKIPPED,
            writeback_id=delta.receipt.writeback_id,
            memory_id=updated.memory_id,
            reason=(
                "Yewo already applied this write-back."
                if application.disposition is MemoryDeltaDisposition.DUPLICATE
                else "Yewo skipped evidence older than the retained write-back horizon."
            ),
        )
    return MemoryWritebackReport(
        source_run_id=source_run_id,
        disposition=MemoryWritebackDisposition.APPLIED,
        writeback_id=delta.receipt.writeback_id,
        memory_id=updated.memory_id,
        grounding_assessment_count=len(delta.grounding_assessments),
        action_risk_count=len(delta.action_risks),
        rule_count=len(delta.rules),
        finding_count=len(delta.findings),
        timing_count=len(delta.timing),
        run_outcome_count=len(delta.run_outcomes),
        screen_summary_count=len(delta.screen_summaries),
        control_count=len(delta.controls),
        claim_review_count=len(delta.claim_reviews),
        skipped_records=delta.skipped_records,
        reason="Yewo applied current-run evidence to AppMemory.",
    )


def _load_replayed_procedure(
    discovery_result: DiscoveryResult,
    compatibility: MemoryCompatibility,
    procedure_store: CompiledProcedureStore | None,
) -> StoredProcedureBundle | None:
    replay = discovery_result.procedure_replay
    if replay is None or not replay.grounding_decisions:
        return None
    if procedure_store is None:
        raise ValueError("Procedure replay write-back requires its procedure store.")
    objective = discovery_result.request.objectives[0]
    bundle = procedure_store.load(
        compatibility,
        objective.objective_id,
        objective.statement,
    )
    if bundle is None:
        raise ValueError("Procedure replay write-back requires its retained procedure.")
    return bundle


def _promote_replay_graph(
    current: AppMemory,
    discovery_result: DiscoveryResult,
    observations_by_id: dict[str, Observation],
    compatibility: MemoryCompatibility,
    source_run_id: RecordIdentifier,
    applied_at: datetime,
) -> tuple[AppMemory, MemoryWritebackSkip | None]:
    replay = discovery_result.procedure_replay
    if replay is None:
        return current, None
    try:
        promotion = AppMemoryBuilder().promote(
            discovery_result.request,
            replay.agent_result,
            observations_by_id,
            compatibility,
            verified_objective_ids=None,
            source_run_id=source_run_id,
            updated_at=applied_at,
            existing_memory=current,
        )
    except (ValidationError, ValueError):
        return current, MemoryWritebackSkip(
            record_kind="graph",
            reason="Current-run repair graph evidence could not be promoted safely.",
        )
    return promotion.memory or current, None


def _update_memory(
    current: AppMemory,
    *,
    discovery_result: DiscoveryResult,
    source_run_id: RecordIdentifier,
    compatibility: MemoryCompatibility,
    observations_by_id: dict[str, Observation],
    combined_result: AgentResult,
    screen_analyses: tuple[CurrentScreenAnalysis, ...],
    bundle: StoredProcedureBundle | None,
    recovery_assessments: tuple[RecoveryAssessment, ...],
    reconciler_response: ReconcilerResponse | None,
    applied_at: datetime,
    applied_deltas: list[MemoryDelta],
    delta_applications: list[MemoryDeltaApplication],
) -> AppMemory:
    promoted, graph_skip = _promote_replay_graph(
        current,
        discovery_result,
        observations_by_id,
        compatibility,
        source_run_id,
        applied_at,
    )
    replay = discovery_result.procedure_replay
    grounding_assessments = (
        build_grounding_assessments(
            promoted,
            source_run_id,
            replay,
            bundle.procedure,
            applied_at,
        )
        if replay is not None and bundle is not None
        else ()
    )
    delta = build_memory_delta(
        promoted,
        source_run_id,
        combined_result,
        observations_by_id,
        applied_at,
        screen_analyses=screen_analyses,
        procedure_attempt_id=replay.setup_id if replay is not None else None,
        grounding_assessments=grounding_assessments,
        recovery_assessments=recovery_assessments,
        reconciler_response=reconciler_response,
        verification=(
            discovery_result.verification or (replay.verification if replay is not None else None)
        ),
        procedure_replay=replay,
    )
    if graph_skip is not None:
        delta = delta.model_copy(update={"skipped_records": (*delta.skipped_records, graph_skip)})
    applied_deltas.append(delta)
    application = apply_memory_delta_result(promoted, delta)
    delta_applications.append(application)
    if application.disposition is MemoryDeltaDisposition.APPLIED:
        return application.memory
    return current


def _load_recovery_assessments(root: Path) -> tuple[RecoveryAssessment, ...]:
    path = root / RECOVERY_ASSESSMENTS_PATH
    if not path.exists():
        return ()
    if path.is_symlink() or not path.is_file():
        raise ValueError("The recovery-assessment ledger must be a regular file.")
    assessments: list[RecoveryAssessment] = []
    try:
        for line in path.read_bytes().splitlines():
            assessments.append(RecoveryAssessment.model_validate_json(line))
    except (OSError, ValidationError) as error:
        raise ValueError("The recovery-assessment ledger is invalid.") from error
    return tuple(assessments)


def _load_reconciler_response(root: Path) -> ReconcilerResponse | None:
    path = root / RECONCILER_RESPONSE_PATH
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError("The reconciler response must be a regular file.")
    try:
        return ReconcilerResponse.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        raise ValueError("The reconciler response is invalid.") from error


def _persist_report(root: Path, report: MemoryWritebackReport) -> None:
    content = report.model_dump_json(indent=2).encode("utf-8") + b"\n"
    try:
        AtomicArtifactWriter(root).write(MEMORY_WRITEBACK_REPORT_PATH, content)
    except ArtifactWriteError:
        narrator = current_narrator()
        if narrator is not None:
            narrator.record("app-memory", "Could not persist the memory write-back report.")


def _narrate_report(report: MemoryWritebackReport) -> None:
    narrator = current_narrator()
    if narrator is not None:
        narrator.record(
            "app-memory",
            f"Knowledge write-back finished with status {report.disposition.value}.",
        )
