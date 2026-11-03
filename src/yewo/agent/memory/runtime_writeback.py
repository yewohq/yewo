"""Isolated terminal write-back from current-run evidence into AppMemory."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from yewo.agent.discovery.records import DiscoveryResult
from yewo.agent.discovery.promotion import AppMemoryBuilder
from yewo.agent.discovery.replay_handoff import replay_observation_ids
from yewo.agent.memory.grounding_capture import build_grounding_assessments
from yewo.agent.memory.knowledge_capture import build_memory_delta
from yewo.agent.memory.records import MemoryCompatibility
from yewo.agent.memory.store import AppMemoryStore
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
from yewo.agent.replay.procedure_store import CompiledProcedureStore
from yewo.agent.replay.reconciliation_records import ReconcilerResponse
from yewo.agent.replay.reconciliation_store import RECONCILER_RESPONSE_PATH
from yewo.artifacts import (
    ARTIFACTS_DIRECTORY,
    ArtifactWriteError,
    AtomicArtifactWriter,
    RecordIdentifier,
)
from yewo.observations import ObservationStore
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

    schema_version: Literal[1] = 1
    source_run_id: RecordIdentifier
    disposition: MemoryWritebackDisposition
    writeback_id: RecordIdentifier | None = None
    memory_id: RecordIdentifier | None = None
    grounding_assessment_count: int = Field(default=0, ge=0)
    hazard_count: int = Field(default=0, ge=0)
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
    memory = memory_store.load(compatibility)
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
    bundle = None
    if replay is not None and replay.grounding_decisions:
        if procedure_store is None:
            raise ValueError("Procedure replay write-back requires its procedure store.")
        objective = result.request.objectives[0]
        bundle = procedure_store.load(
            compatibility,
            objective.objective_id,
            objective.statement,
        )
        if bundle is None:
            raise ValueError("Procedure replay write-back requires its retained procedure.")
    combined_steps = tuple(
        {
            step.step_id: step
            for step in (
                *(replay.agent_result.steps if replay is not None else ()),
                *result.agent_result.steps,
            )
        }.values()
    )
    combined_result = result.agent_result.model_copy(update={"steps": combined_steps})
    recovery_assessments = _load_recovery_assessments(root)
    reconciler_response = _load_reconciler_response(root)
    applied_delta: list[MemoryDelta] = []
    delta_applications: list[MemoryDeltaApplication] = []

    def update_memory(current):
        graph_skip: MemoryWritebackSkip | None = None
        try:
            graph_result = replay.agent_result if replay is not None else None
            if graph_result is None:
                promoted = current
            else:
                promotion = AppMemoryBuilder().promote(
                    result.request,
                    graph_result,
                    observations_by_id,
                    compatibility,
                    verified_objective_ids=None,
                    source_run_id=source_run_id,
                    updated_at=applied_at,
                    existing=current,
                )
                promoted = promotion.memory or current
        except (ValidationError, ValueError):
            promoted = current
            graph_skip = MemoryWritebackSkip(
                record_kind="graph",
                reason="Current-run repair graph evidence could not be promoted safely.",
            )
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
            procedure_attempt_id=replay.setup_id if replay is not None else None,
            grounding_assessments=grounding_assessments,
            recovery_assessments=recovery_assessments,
            reconciler_response=reconciler_response,
            verification=(
                result.verification or (replay.verification if replay is not None else None)
            ),
            procedure_replay=replay,
        )
        if graph_skip is not None:
            delta = delta.model_copy(
                update={"skipped_records": (*delta.skipped_records, graph_skip)}
            )
        applied_delta.append(delta)
        application = apply_memory_delta_result(promoted, delta)
        delta_applications.append(application)
        return (
            application.memory
            if application.disposition is MemoryDeltaDisposition.APPLIED
            else current
        )

    updated = memory_store.update(
        compatibility,
        root,
        update_memory,
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
        hazard_count=len(delta.hazards),
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
