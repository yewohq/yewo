"""Contracts for reconciler output and compiled replay procedures."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yewo.agent.actuation import ActionOutcomeDisposition, ActionOutcomeSignal, ActionProposal
from yewo.agent.memory.knowledge import KnowledgeEvidenceReference
from yewo.agent.replay.reconciliation_records import TargetVisualDescriptor
from yewo.artifacts import EvidenceApplicationIdentity, RecordIdentifier
from yewo.grounding import GroundingEvidence


class ReconcilerDecision(BaseModel):
    """The model's explanation of one keep-or-drop choice."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    keep: tuple[int, ...] = Field(default=(), max_length=500)
    drop: tuple[int, ...] = Field(default=(), max_length=500)
    why: str = Field(min_length=1, max_length=2_000)


class ReconcilerClaimVerdict(BaseModel):
    """Evidence-backed review suggestion for one retained model-authored claim."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    claim_id: RecordIdentifier
    status: Literal["confirmed", "contradicted"]
    source_step_ids: tuple[RecordIdentifier, ...] = Field(min_length=1, max_length=100)
    reason: str = Field(min_length=1, max_length=2_000)


class ReconcilerResponse(BaseModel):
    """Untrusted model output that selects only recorded source steps."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    request_id: RecordIdentifier
    procedure: tuple[int, ...] = Field(max_length=500)
    milestones: dict[int, int]
    decisions: tuple[ReconcilerDecision, ...] = Field(default=(), max_length=100)
    findings: tuple[str, ...] = Field(default=(), max_length=100)
    claim_verdicts: tuple[ReconcilerClaimVerdict, ...] = Field(default=(), max_length=100)

    @model_validator(mode="after")
    def _validate_claim_verdicts(self) -> Self:
        claim_ids = [item.claim_id for item in self.claim_verdicts]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("Reconciler claim verdict identifiers must be unique.")
        if any(
            len(item.source_step_ids) != len(set(item.source_step_ids))
            for item in self.claim_verdicts
        ):
            raise ValueError("A claim verdict cannot repeat a source step.")
        return self


class CompilerCheck(StrEnum):
    """Stable compiler gates for untrusted reconciler output."""

    SELECTION_ONLY = "selection_only"
    STARTS_AT_LAUNCH = "starts_at_launch"
    NO_LOST_STATE = "no_lost_state_setting_work"
    RESULTS_REUSABLE = "results_are_reusable"
    COVERS_TASK = "covers_the_task"
    AVOIDS_KNOWN_HAZARDS = "avoids_known_hazards"


class CompilerReason(BaseModel):
    """One fail-closed compiler reason."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    check: CompilerCheck
    source_step: int | None = Field(default=None, gt=0)
    reason: str = Field(min_length=1, max_length=2_000)
    hazard_id: RecordIdentifier | None = None
    hazard_sources: tuple[KnowledgeEvidenceReference, ...] = ()

    @model_validator(mode="after")
    def _validate_hazard_evidence(self) -> Self:
        if self.hazard_sources and self.hazard_id is None:
            raise ValueError("Hazard evidence requires a hazard identifier.")
        if self.hazard_id is not None and self.check is not CompilerCheck.AVOIDS_KNOWN_HAZARDS:
            raise ValueError("Only the known-hazard check can reference a hazard.")
        return self


class CompilerRefusal(BaseModel):
    """Structured reasons why the compiler did not produce a procedure."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    request_id: RecordIdentifier
    evidence_id: RecordIdentifier
    reasons: tuple[CompilerReason, ...] = Field(min_length=1)


class ProcedureEffect(StrEnum):
    """Replay behavior assigned independently from authorization policy."""

    NAVIGATION = "navigation"
    STATE_SETTING = "state_setting"
    COMMIT = "commit"
    ENVIRONMENT = "environment"
    UNKNOWN = "unknown"


class ProcedureExpectation(BaseModel):
    """Immediate current-run evidence required after replaying one step."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    destination_state_id: RecordIdentifier | None = None
    required_signals: frozenset[ActionOutcomeSignal] = frozenset()
    source_outcome_disposition: ActionOutcomeDisposition | None = None


class ReplayProcedureStep(BaseModel):
    """One compiled action copied from source evidence without modification."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    procedure_index: int = Field(ge=0)
    source_step: int = Field(gt=0)
    source_step_id: RecordIdentifier
    source_state_id: RecordIdentifier
    proposal: ActionProposal
    target_grounding: GroundingEvidence | None = None
    target_visual_descriptor: TargetVisualDescriptor | None = None
    destination_grounding: GroundingEvidence | None = None
    procedure_effect: ProcedureEffect
    expects: ProcedureExpectation
    captures_completion_conditions: tuple[int, ...] = Field(default=(), max_length=100)


class CompilationWarning(BaseModel):
    """A non-fatal uncertainty that replay expectations must resolve."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    source_step: int = Field(gt=0)
    reason: str = Field(min_length=1, max_length=1_000)


class ReplayProcedure(BaseModel):
    """A compiler-approved procedure independent from application memory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    visual_policy_version: Literal[1, 2] = 1
    procedure_id: RecordIdentifier
    request_id: RecordIdentifier
    evidence_id: RecordIdentifier
    source_run_id: RecordIdentifier
    application: EvidenceApplicationIdentity
    source_step_ids: tuple[RecordIdentifier, ...]
    steps: tuple[ReplayProcedureStep, ...]
    warnings: tuple[CompilationWarning, ...] = ()

    @model_validator(mode="after")
    def _validate_compiled_links(self) -> Self:
        if self.source_step_ids != tuple(step.source_step_id for step in self.steps):
            raise ValueError("Procedure source identifiers must match compiled steps.")
        if tuple(step.procedure_index for step in self.steps) != tuple(range(len(self.steps))):
            raise ValueError(
                "The first compiled procedure index must be zero. All indexes must be contiguous."
            )
        return self


CompilationOutcome = ReplayProcedure | CompilerRefusal


from yewo.agent.replay.reconciliation_report import (  # noqa: E402, F401
    ProcedureCompilationReport,
    ProcedureCompilationStatus,
)
