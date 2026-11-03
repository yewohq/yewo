"""Versioned cross-run application knowledge and write-back provenance."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yewo.agent.actuation import ActionKind, VisibleControlKind
from yewo.artifacts import RecordIdentifier


KNOWLEDGE_EVIDENCE_RETENTION = 200
MEMORY_WRITEBACK_RECEIPT_RETENTION = 1_000
RUN_OUTCOME_RETENTION = 1_000
CLAIM_REVIEW_RETENTION = 5_000


class ActionRiskKind(StrEnum):
    """Observed application failure that an exact action may trigger."""

    CRASH = "crash"
    ANR = "anr"
    FREEZE = "freeze"
    APPLICATION_EXIT = "application_exit"


class GroundingAssessmentStatus(StrEnum):
    """Whether one exact stored grounding remains reusable."""

    ACTIVE = "active"
    INVALIDATED = "invalidated"


class GroundingRole(StrEnum):
    """Which grounding location of a transition is assessed."""

    TARGET = "target"
    DESTINATION = "destination"


class KnowledgeEvidenceReference(BaseModel):
    """One current-run artifact reference supporting a knowledge record."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    run_id: RecordIdentifier
    step_id: RecordIdentifier | None = None
    observation_id: RecordIdentifier | None = None
    procedure_attempt_id: RecordIdentifier | None = None


class GroundingAssessment(BaseModel):
    """Replay evidence for one grounding on one exact source observation."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    assessment_id: RecordIdentifier
    transition_id: RecordIdentifier
    role: GroundingRole
    source_observation_id: RecordIdentifier
    status: GroundingAssessmentStatus = GroundingAssessmentStatus.ACTIVE
    successful_use_count: int = Field(default=0, ge=0)
    last_success_at: datetime | None = None
    invalidation_reason: str | None = Field(default=None, min_length=1, max_length=1_000)
    replacement_assessment_id: RecordIdentifier | None = None
    sources: tuple[KnowledgeEvidenceReference, ...] = Field(
        default=(), max_length=KNOWLEDGE_EVIDENCE_RETENTION
    )

    @model_validator(mode="after")
    def _validate_status_evidence(self) -> Self:
        _validate_optional_timestamp(self.last_success_at, "Grounding success time")
        is_invalidated = self.status is GroundingAssessmentStatus.INVALIDATED
        if is_invalidated != (self.invalidation_reason is not None):
            raise ValueError("Invalidated grounding requires one invalidation reason.")
        if self.successful_use_count > 0 and self.last_success_at is None:
            raise ValueError("Reinforced grounding requires its last success time.")
        return self


class ActionRiskKnowledge(BaseModel):
    """Observed application failure tied to an exact screen and action."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    action_risk_id: RecordIdentifier
    screen_id: RecordIdentifier
    action: ActionKind
    target_description: str | None = Field(default=None, min_length=1, max_length=2_000)
    kind: ActionRiskKind
    exception_type: str | None = Field(default=None, min_length=1, max_length=500)
    top_frame: str | None = Field(default=None, min_length=1, max_length=1_000)
    seen_count: int = Field(ge=1)
    recovered_count: int = Field(default=0, ge=0)
    sources: tuple[KnowledgeEvidenceReference, ...] = Field(
        min_length=1, max_length=KNOWLEDGE_EVIDENCE_RETENTION
    )

    @model_validator(mode="after")
    def _validate_counts(self) -> Self:
        if self.recovered_count > self.seen_count:
            raise ValueError("Action-risk recovery count cannot exceed its seen count.")
        return self


class KnowledgeScope(BaseModel):
    """Typed application context to which one claim applies."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    screen_id: RecordIdentifier | None = None
    action: ActionKind | None = None
    target_description: str | None = Field(default=None, min_length=1, max_length=2_000)


class KnowledgeRule(BaseModel):
    """Model-authored guidance that cannot itself block an action."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    rule_id: RecordIdentifier
    text: str = Field(min_length=1, max_length=4_000)
    scope: KnowledgeScope = KnowledgeScope()
    linked_action_risk_id: RecordIdentifier | None = None
    sources: tuple[KnowledgeEvidenceReference, ...] = Field(
        min_length=1, max_length=KNOWLEDGE_EVIDENCE_RETENTION
    )


class KnowledgeFinding(BaseModel):
    """Reconciler-authored application claim retained as a labelled hint."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    finding_id: RecordIdentifier
    text: str = Field(min_length=1, max_length=4_000)
    scope: KnowledgeScope = KnowledgeScope()
    sources: tuple[KnowledgeEvidenceReference, ...] = Field(
        min_length=1, max_length=KNOWLEDGE_EVIDENCE_RETENTION
    )


class TransitionTimingSample(BaseModel):
    """One successful settled-action duration with idempotent provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: RecordIdentifier
    measurement: Literal["source_to_settled_v1", "dispatch_to_settled_v1"] = "source_to_settled_v1"
    duration_ms: float = Field(ge=0, le=300_000)
    source: KnowledgeEvidenceReference


class TransitionTimingKnowledge(BaseModel):
    """Bounded successful timing samples for one semantic transition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    transition_id: RecordIdentifier
    samples: tuple[TransitionTimingSample, ...] = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def _validate_unique_samples(self) -> Self:
        sample_ids = [sample.sample_id for sample in self.samples]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("Transition timing cannot repeat a sample.")
        return self


class MemoryWritebackReceipt(BaseModel):
    """One idempotently applied memory write-back batch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    writeback_id: RecordIdentifier
    source_run_id: RecordIdentifier
    procedure_attempt_id: RecordIdentifier | None = None
    applied_at: datetime

    @model_validator(mode="after")
    def _validate_timestamp(self) -> Self:
        _validate_optional_timestamp(self.applied_at, "Memory write-back time")
        return self


class RunOutcomeKnowledge(BaseModel):
    """Auditor-labelled final application state for one completed run."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    run_id: RecordIdentifier
    satisfied: bool
    observed_app_state: str = Field(min_length=1, max_length=4_000)
    source: KnowledgeEvidenceReference


class ScreenSummaryKnowledge(BaseModel):
    """Latest planner-authored label for one retained logical screen."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    screen_id: RecordIdentifier
    text: str = Field(min_length=1, max_length=120)
    sources: tuple[KnowledgeEvidenceReference, ...] = Field(
        min_length=1, max_length=KNOWLEDGE_EVIDENCE_RETENTION
    )


class ControlKnowledge(BaseModel):
    """Verifier-observed control on one retained logical screen."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    control_id: RecordIdentifier
    screen_id: RecordIdentifier
    name: str = Field(min_length=1, max_length=200)
    kind: VisibleControlKind
    state: str | None = Field(default=None, min_length=1, max_length=200)
    options: tuple[str, ...] = Field(default=(), max_length=100)
    exercised: bool = False
    sources: tuple[KnowledgeEvidenceReference, ...] = Field(
        min_length=1, max_length=KNOWLEDGE_EVIDENCE_RETENTION
    )


class ClaimReview(BaseModel):
    """Reconciler-proposed verdict retained without changing claim trust."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    review_id: RecordIdentifier
    claim_id: RecordIdentifier
    proposed_status: Literal["confirmed", "contradicted"]
    reason: str = Field(min_length=1, max_length=2_000)
    sources: tuple[KnowledgeEvidenceReference, ...] = Field(
        min_length=1, max_length=KNOWLEDGE_EVIDENCE_RETENTION
    )


class AppKnowledge(BaseModel):
    """Cross-run knowledge that may guide but never prove current completion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    grounding_assessments: tuple[GroundingAssessment, ...] = Field(default=(), max_length=5_000)
    action_risks: tuple[ActionRiskKnowledge, ...] = Field(default=(), max_length=1_000)
    rules: tuple[KnowledgeRule, ...] = Field(default=(), max_length=1_000)
    findings: tuple[KnowledgeFinding, ...] = Field(default=(), max_length=1_000)
    timing: tuple[TransitionTimingKnowledge, ...] = Field(default=(), max_length=5_000)
    run_outcomes: tuple[RunOutcomeKnowledge, ...] = Field(
        default=(), max_length=RUN_OUTCOME_RETENTION
    )
    screen_summaries: tuple[ScreenSummaryKnowledge, ...] = Field(default=(), max_length=1_000)
    controls: tuple[ControlKnowledge, ...] = Field(default=(), max_length=5_000)
    claim_reviews: tuple[ClaimReview, ...] = Field(default=(), max_length=CLAIM_REVIEW_RETENTION)
    writeback_floor: datetime | None = None
    writebacks: tuple[MemoryWritebackReceipt, ...] = Field(
        default=(), max_length=MEMORY_WRITEBACK_RECEIPT_RETENTION
    )

    @model_validator(mode="after")
    def _validate_writeback_floor(self) -> Self:
        _validate_optional_timestamp(self.writeback_floor, "Write-back retention floor")
        return self


def validate_app_knowledge(
    knowledge: AppKnowledge,
    screen_ids: set[RecordIdentifier],
    transition_ids: set[RecordIdentifier],
    observation_ids: set[RecordIdentifier],
    source_run_ids: set[RecordIdentifier],
) -> None:
    """Reject duplicate identifiers and references outside the memory graph."""
    _validate_knowledge_identifiers(knowledge)
    _validate_knowledge_graph_references(
        knowledge,
        screen_ids,
        transition_ids,
        observation_ids,
    )
    _validate_knowledge_sources(knowledge, source_run_ids)


def _validate_knowledge_identifiers(knowledge: AppKnowledge) -> None:
    _validate_unique_ids(
        "grounding assessment",
        [item.assessment_id for item in knowledge.grounding_assessments],
    )
    grounding_keys = [
        (item.transition_id, item.role, item.source_observation_id)
        for item in knowledge.grounding_assessments
    ]
    if len(grounding_keys) != len(set(grounding_keys)):
        raise ValueError("App knowledge grounding assessment keys must be unique.")
    _validate_unique_ids("action risk", [item.action_risk_id for item in knowledge.action_risks])
    _validate_unique_ids("rule", [item.rule_id for item in knowledge.rules])
    _validate_unique_ids("finding", [item.finding_id for item in knowledge.findings])
    _validate_unique_ids("timing transition", [item.transition_id for item in knowledge.timing])
    _validate_unique_ids("run outcome", [item.run_id for item in knowledge.run_outcomes])
    _validate_unique_ids("screen summary", [item.screen_id for item in knowledge.screen_summaries])
    _validate_unique_ids("control", [item.control_id for item in knowledge.controls])
    _validate_unique_ids("claim review", [item.review_id for item in knowledge.claim_reviews])
    _validate_unique_ids("write-back", [item.writeback_id for item in knowledge.writebacks])


def _validate_knowledge_graph_references(
    knowledge: AppKnowledge,
    screen_ids: set[RecordIdentifier],
    transition_ids: set[RecordIdentifier],
    observation_ids: set[RecordIdentifier],
) -> None:
    _validate_grounding_and_timing_references(knowledge, transition_ids, observation_ids)
    _validate_screen_references(knowledge, screen_ids)
    _validate_claim_references(knowledge)


def _validate_grounding_and_timing_references(
    knowledge: AppKnowledge,
    transition_ids: set[RecordIdentifier],
    observation_ids: set[RecordIdentifier],
) -> None:
    if any(
        item.transition_id not in transition_ids
        or item.source_observation_id not in observation_ids
        for item in knowledge.grounding_assessments
    ):
        raise ValueError("Every grounding assessment must reference retained graph evidence.")
    if any(item.transition_id not in transition_ids for item in knowledge.timing):
        raise ValueError("Every timing record must reference a retained transition.")
    assessment_ids = {item.assessment_id for item in knowledge.grounding_assessments}
    if any(
        item.replacement_assessment_id is not None
        and item.replacement_assessment_id not in assessment_ids
        for item in knowledge.grounding_assessments
    ):
        raise ValueError("Every replacement grounding must reference a retained assessment.")


def _validate_screen_references(
    knowledge: AppKnowledge,
    screen_ids: set[RecordIdentifier],
) -> None:
    if any(item.screen_id not in screen_ids for item in knowledge.action_risks):
        raise ValueError("Every action risk must reference a retained screen.")
    if any(
        item.scope.screen_id is not None and item.scope.screen_id not in screen_ids
        for item in (*knowledge.rules, *knowledge.findings)
    ):
        raise ValueError("Every scoped claim must reference a retained screen.")
    if any(item.screen_id not in screen_ids for item in knowledge.screen_summaries):
        raise ValueError("Every screen summary must reference a retained screen.")
    if any(item.screen_id not in screen_ids for item in knowledge.controls):
        raise ValueError("Every control must reference a retained screen.")


def _validate_claim_references(knowledge: AppKnowledge) -> None:
    claim_ids = {
        *(item.rule_id for item in knowledge.rules),
        *(item.finding_id for item in knowledge.findings),
    }
    if any(item.claim_id not in claim_ids for item in knowledge.claim_reviews):
        raise ValueError("Every claim review must reference a retained claim.")
    action_risk_ids = {item.action_risk_id for item in knowledge.action_risks}
    if any(
        item.linked_action_risk_id is not None and item.linked_action_risk_id not in action_risk_ids
        for item in knowledge.rules
    ):
        raise ValueError("Every linked rule must reference a retained action risk.")


def _validate_knowledge_sources(
    knowledge: AppKnowledge,
    source_run_ids: set[RecordIdentifier],
) -> None:
    references = _knowledge_evidence_references(knowledge)
    referenced_run_ids = {
        *(reference.run_id for reference in references),
        *(item.run_id for item in knowledge.run_outcomes),
        *(receipt.source_run_id for receipt in knowledge.writebacks),
    }
    if not referenced_run_ids <= source_run_ids:
        raise ValueError("Every app-knowledge source must reference a retained memory run.")


def app_knowledge_source_run_ids(knowledge: AppKnowledge) -> set[RecordIdentifier]:
    """Return every run identifier required by retained knowledge."""
    return {
        *(reference.run_id for reference in _knowledge_evidence_references(knowledge)),
        *(item.run_id for item in knowledge.run_outcomes),
        *(receipt.source_run_id for receipt in knowledge.writebacks),
    }


def _knowledge_evidence_references(
    knowledge: AppKnowledge,
) -> list[KnowledgeEvidenceReference]:
    references = [
        reference for item in knowledge.grounding_assessments for reference in item.sources
    ]
    references.extend(reference for item in knowledge.action_risks for reference in item.sources)
    references.extend(
        reference for item in (*knowledge.rules, *knowledge.findings) for reference in item.sources
    )
    references.extend(sample.source for item in knowledge.timing for sample in item.samples)
    references.extend(item.source for item in knowledge.run_outcomes)
    references.extend(source for item in knowledge.screen_summaries for source in item.sources)
    references.extend(source for item in knowledge.controls for source in item.sources)
    references.extend(source for item in knowledge.claim_reviews for source in item.sources)
    return references


def _validate_unique_ids(label: str, identifiers: list[RecordIdentifier]) -> None:
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"App knowledge {label} identifiers must be unique.")


def _validate_optional_timestamp(value: datetime | None, label: str) -> None:
    if value is not None and value.utcoffset() is None:
        raise ValueError(f"{label} must include a timezone.")
