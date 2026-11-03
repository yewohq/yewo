"""Idempotent application of one evidence-derived app-memory update."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from yewo.agent.memory.knowledge import (
    AppKnowledge,
    ClaimReview,
    CLAIM_REVIEW_RETENTION,
    ControlKnowledge,
    GroundingAssessment,
    GroundingAssessmentStatus,
    ActionRiskKnowledge,
    KnowledgeFinding,
    KnowledgeRule,
    MemoryWritebackReceipt,
    RunOutcomeKnowledge,
    RUN_OUTCOME_RETENTION,
    ScreenSummaryKnowledge,
    TransitionTimingKnowledge,
    app_knowledge_source_run_ids,
    KNOWLEDGE_EVIDENCE_RETENTION,
    MEMORY_WRITEBACK_RECEIPT_RETENTION,
)
from yewo.agent.memory.records import AppMemory
from yewo.artifacts import RecordIdentifier


class MemoryWritebackSkip(BaseModel):
    """One evidence record omitted without discarding the rest of its batch."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    record_kind: Literal["action_risk", "rule", "timing", "claim", "graph", "control", "outcome"]
    source_step_id: RecordIdentifier | None = None
    reason: str = Field(min_length=1, max_length=500)


class MemoryDelta(BaseModel):
    """One replay or discovery batch ready for deterministic memory merge."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    receipt: MemoryWritebackReceipt
    grounding_assessments: tuple[GroundingAssessment, ...] = Field(default=(), max_length=1_000)
    action_risks: tuple[ActionRiskKnowledge, ...] = Field(default=(), max_length=1_000)
    rules: tuple[KnowledgeRule, ...] = Field(default=(), max_length=1_000)
    findings: tuple[KnowledgeFinding, ...] = Field(default=(), max_length=1_000)
    timing: tuple[TransitionTimingKnowledge, ...] = Field(default=(), max_length=1_000)
    run_outcomes: tuple[RunOutcomeKnowledge, ...] = Field(default=(), max_length=1_000)
    screen_summaries: tuple[ScreenSummaryKnowledge, ...] = Field(default=(), max_length=1_000)
    controls: tuple[ControlKnowledge, ...] = Field(default=(), max_length=5_000)
    claim_reviews: tuple[ClaimReview, ...] = Field(default=(), max_length=5_000)
    skipped_records: tuple[MemoryWritebackSkip, ...] = Field(default=(), max_length=1_000)


class MemoryDeltaDisposition(StrEnum):
    """Whether one memory delta changed the retained snapshot."""

    APPLIED = "applied"
    DUPLICATE = "duplicate"
    STALE = "stale"


class MemoryDeltaApplication(BaseModel):
    """One deterministic delta application and its resulting memory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    memory: AppMemory
    disposition: MemoryDeltaDisposition


def apply_memory_delta(memory: AppMemory, delta: MemoryDelta) -> AppMemory:
    """Apply one batch once and validate the resulting complete memory graph."""
    return apply_memory_delta_result(memory, delta).memory


def apply_memory_delta_result(
    memory: AppMemory,
    delta: MemoryDelta,
) -> MemoryDeltaApplication:
    """Apply one batch and report whether it changed memory."""
    knowledge = memory.app_knowledge
    if delta.receipt.writeback_id in {item.writeback_id for item in knowledge.writebacks}:
        return MemoryDeltaApplication(
            memory=memory,
            disposition=MemoryDeltaDisposition.DUPLICATE,
        )
    if knowledge.writeback_floor is not None and (
        delta.receipt.applied_at <= knowledge.writeback_floor
    ):
        return MemoryDeltaApplication(
            memory=memory,
            disposition=MemoryDeltaDisposition.STALE,
        )
    receipts = (*knowledge.writebacks, delta.receipt)
    evicted_receipts = receipts[:-MEMORY_WRITEBACK_RECEIPT_RETENTION]
    retained_receipts = receipts[-MEMORY_WRITEBACK_RECEIPT_RETENTION:]
    floor_candidates = (
        *((knowledge.writeback_floor,) if knowledge.writeback_floor is not None else ()),
        *(receipt.applied_at for receipt in evicted_receipts),
    )
    merged = AppKnowledge(
        grounding_assessments=_merge_by_id(
            knowledge.grounding_assessments,
            delta.grounding_assessments,
            lambda item: item.assessment_id,
            _merge_grounding,
        ),
        action_risks=_merge_by_id(
            knowledge.action_risks,
            delta.action_risks,
            lambda item: item.action_risk_id,
            _merge_action_risk,
        ),
        rules=_merge_by_id(knowledge.rules, delta.rules, lambda item: item.rule_id, _merge_rule),
        findings=_merge_by_id(
            knowledge.findings, delta.findings, lambda item: item.finding_id, _merge_finding
        ),
        timing=_merge_by_id(
            knowledge.timing, delta.timing, lambda item: item.transition_id, _merge_timing
        ),
        run_outcomes=_merge_by_id(
            knowledge.run_outcomes,
            delta.run_outcomes,
            lambda item: item.run_id,
            _merge_run_outcome,
        )[-RUN_OUTCOME_RETENTION:],
        screen_summaries=_merge_by_id(
            knowledge.screen_summaries,
            delta.screen_summaries,
            lambda item: item.screen_id,
            _merge_screen_summary,
        ),
        controls=_merge_by_id(
            knowledge.controls,
            delta.controls,
            lambda item: item.control_id,
            _merge_control,
        ),
        claim_reviews=_merge_by_id(
            knowledge.claim_reviews,
            delta.claim_reviews,
            lambda item: item.review_id,
            _merge_claim_review,
        )[-CLAIM_REVIEW_RETENTION:],
        writeback_floor=max(floor_candidates, default=None),
        writebacks=retained_receipts,
    )
    payload = memory.model_dump()
    payload.update(
        {
            "source_run_ids": tuple(
                dict.fromkeys((*memory.source_run_ids, delta.receipt.source_run_id))
            ),
            "app_knowledge": merged,
            "updated_at": delta.receipt.applied_at,
        }
    )
    updated = _retain_referenced_source_runs(AppMemory.model_validate(payload))
    return MemoryDeltaApplication(
        memory=updated,
        disposition=MemoryDeltaDisposition.APPLIED,
    )


def _retain_referenced_source_runs(memory: AppMemory) -> AppMemory:
    referenced_run_ids = {
        *(variant.source_run_id for screen in memory.screens for variant in screen.variants),
        *(attempt.source_run_id for attempt in memory.unsuccessful_attempts),
        *app_knowledge_source_run_ids(memory.app_knowledge),
    }
    retained_run_ids = tuple(
        run_id for run_id in memory.source_run_ids if run_id in referenced_run_ids
    )
    if retained_run_ids == memory.source_run_ids:
        return memory
    return AppMemory.model_validate(
        {**memory.model_dump(mode="python"), "source_run_ids": retained_run_ids}
    )


_Record = TypeVar("_Record")


def _merge_by_id(
    current: tuple[_Record, ...],
    updates: tuple[_Record, ...],
    identifier: Callable[[_Record], RecordIdentifier],
    merge: Callable[[_Record, _Record], _Record],
) -> tuple[_Record, ...]:
    merged = list(current)
    indexes = {identifier(item): index for index, item in enumerate(merged)}
    for update in updates:
        record_id = identifier(update)
        index = indexes.get(record_id)
        if index is None:
            indexes[record_id] = len(merged)
            merged.append(update)
        else:
            merged[index] = merge(merged[index], update)
    return tuple(merged)


def _merge_grounding(
    current: GroundingAssessment,
    update: GroundingAssessment,
) -> GroundingAssessment:
    _require_same(
        current,
        update,
        {
            "status",
            "successful_use_count",
            "last_success_at",
            "invalidation_reason",
            "replacement_assessment_id",
            "sources",
        },
        "grounding assessment",
    )
    status = (
        GroundingAssessmentStatus.INVALIDATED
        if GroundingAssessmentStatus.INVALIDATED in {current.status, update.status}
        else GroundingAssessmentStatus.ACTIVE
    )
    invalidated = update if update.status is GroundingAssessmentStatus.INVALIDATED else current
    return current.model_copy(
        update={
            "status": status,
            "successful_use_count": current.successful_use_count + update.successful_use_count,
            "last_success_at": _latest_timestamp(current.last_success_at, update.last_success_at),
            "invalidation_reason": (
                invalidated.invalidation_reason
                if status is GroundingAssessmentStatus.INVALIDATED
                else None
            ),
            "replacement_assessment_id": (
                invalidated.replacement_assessment_id
                if status is GroundingAssessmentStatus.INVALIDATED
                else None
            ),
            "sources": _retained_evidence((*current.sources, *update.sources)),
        }
    )


def _merge_action_risk(
    current: ActionRiskKnowledge, update: ActionRiskKnowledge
) -> ActionRiskKnowledge:
    _require_same(current, update, {"seen_count", "recovered_count", "sources"}, "action risk")
    return current.model_copy(
        update={
            "seen_count": current.seen_count + update.seen_count,
            "recovered_count": current.recovered_count + update.recovered_count,
            "sources": _retained_evidence((*current.sources, *update.sources)),
        }
    )


def _merge_rule(current: KnowledgeRule, update: KnowledgeRule) -> KnowledgeRule:
    _require_same(current, update, {"sources"}, "rule")
    return current.model_copy(
        update={"sources": _retained_evidence((*current.sources, *update.sources))}
    )


def _merge_finding(current: KnowledgeFinding, update: KnowledgeFinding) -> KnowledgeFinding:
    _require_same(current, update, {"sources"}, "finding")
    return current.model_copy(
        update={"sources": _retained_evidence((*current.sources, *update.sources))}
    )


def _merge_timing(
    current: TransitionTimingKnowledge,
    update: TransitionTimingKnowledge,
) -> TransitionTimingKnowledge:
    return current.model_copy(
        update={"samples": _unique_models((*current.samples, *update.samples))[-200:]}
    )


def _merge_run_outcome(
    current: RunOutcomeKnowledge,
    update: RunOutcomeKnowledge,
) -> RunOutcomeKnowledge:
    _require_same(current, update, set(), "run outcome")
    return current


def _merge_screen_summary(
    current: ScreenSummaryKnowledge,
    update: ScreenSummaryKnowledge,
) -> ScreenSummaryKnowledge:
    return update.model_copy(
        update={"sources": _retained_evidence((*current.sources, *update.sources))}
    )


def _merge_control(current: ControlKnowledge, update: ControlKnowledge) -> ControlKnowledge:
    _require_same(current, update, {"state", "options", "exercised", "sources"}, "control")
    return update.model_copy(
        update={
            "options": tuple(dict.fromkeys((*current.options, *update.options))),
            "exercised": current.exercised or update.exercised,
            "sources": _retained_evidence((*current.sources, *update.sources)),
        }
    )


def _merge_claim_review(current: ClaimReview, update: ClaimReview) -> ClaimReview:
    _require_same(current, update, {"sources"}, "claim review")
    return current.model_copy(
        update={"sources": _retained_evidence((*current.sources, *update.sources))}
    )


def _require_same(
    current: BaseModel,
    update: BaseModel,
    excluded_fields: set[str],
    label: str,
) -> None:
    if current.model_dump(exclude=excluded_fields) != update.model_dump(exclude=excluded_fields):
        raise ValueError(f"Memory delta conflicts with the retained {label}.")


def _unique_models(records: tuple[_Record, ...]) -> tuple[_Record, ...]:
    unique: list[_Record] = []
    encoded: set[str] = set()
    for record in records:
        key = record.model_dump_json() if isinstance(record, BaseModel) else repr(record)
        if key not in encoded:
            encoded.add(key)
            unique.append(record)
    return tuple(unique)


def _retained_evidence(records: tuple[_Record, ...]) -> tuple[_Record, ...]:
    return _unique_models(records)[-KNOWLEDGE_EVIDENCE_RETENTION:]


def _latest_timestamp(first: datetime | None, second: datetime | None) -> datetime | None:
    values = tuple(value for value in (first, second) if value is not None)
    return max(values, default=None)
