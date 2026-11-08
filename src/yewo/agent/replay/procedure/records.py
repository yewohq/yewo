"""Execution records for compiler-approved replay procedures."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yewo.agent.actuation import ActionKind
from yewo.agent.planning.goal_verifier import GoalVerification
from yewo.agent.runtime.records import AgentResult
from yewo.artifacts import EvidenceApplicationIdentity, RecordIdentifier, Sha256Digest
from yewo.devices.application_lifecycle import TrustedSystemSurface
from yewo.observations import PresentationKey, ScreenFingerprint, ScreenshotReference
from yewo.grounding import GroundedLocation, GroundingResolutionSource


class ProcedureResetMethod(StrEnum):
    """Trusted setup method that establishes an empty replay start."""

    FRESH_INSTALL = "fresh_install"
    DETERMINISTIC_RESET = "deterministic_reset"


class ProcedureReplaySetup(BaseModel):
    """Evidence that replay starts from the procedure's empty application state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    setup_id: RecordIdentifier
    method: ProcedureResetMethod
    application: EvidenceApplicationIdentity
    setup_contract_sha256: Sha256Digest


class ProcedureReplayTerminal(StrEnum):
    """Why compiled procedure execution and its optional audit stopped."""

    COMPLETED = "completed"
    REPLAY_COMPLETE = "replay_complete"
    DIVERGED = "diverged"
    BUDGET_EXHAUSTED = "budget_exhausted"
    APPLICATION_FAILURE = "application_failure"
    DEVICE_REFUSED = "device_refused"


class ProcedureGroundingDecisionCode(StrEnum):
    """One compiled grounding reuse or delegation decision."""

    AUTHORIZED_COORDINATE_REUSED = "authorized_coordinate_reused"
    AUTHORIZED_COORDINATE_RELOCATED = "authorized_coordinate_relocated"
    NO_COMPILED_GROUNDING = "no_compiled_grounding"
    CORRECTION_REQUESTED = "correction_requested"
    STORED_GROUNDING_INVALID = "stored_grounding_invalid"
    TARGET_CONTRACT_CHANGED = "target_contract_changed"
    AUTHORIZED_OBSERVATION_CHANGED = "authorized_observation_changed"
    GEOMETRY_CHANGED = "geometry_changed"
    LOCATION_EXCLUDED = "location_excluded"
    TARGET_VISUAL_CHANGED = "target_visual_changed"


class ProcedureGroundingDecision(BaseModel):
    """Explain one procedure-grounding reuse or fallback decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    grounding_id: RecordIdentifier
    source_step_id: RecordIdentifier | None = None
    procedure_index: int | None = Field(default=None, ge=0)
    expected_state_id: RecordIdentifier | None = None
    observation_id: RecordIdentifier | None = None
    screenshot_sha256: str
    decision: ProcedureGroundingDecisionCode
    expected_phash_distance: int | None = Field(default=None, ge=0, le=64)
    competing_phash_distance: int | None = Field(default=None, ge=0, le=64)
    source_visual_match_ratio: float | None = Field(default=None, ge=0, le=1)
    target_hash_distance: int | None = Field(default=None, ge=0, le=64)
    target_runner_up_distance: int | None = Field(default=None, ge=0, le=64)
    stored_location: GroundedLocation | None = None
    resolved_location: GroundedLocation | None = None
    resolution_source: GroundingResolutionSource


class MilestoneEvidence(BaseModel):
    """Current-run observation captured at one compiled milestone."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    completion_condition_index: int = Field(ge=0)
    procedure_index: int = Field(ge=0)
    source_step: int = Field(gt=0)
    observation_id: RecordIdentifier
    screenshot_artifact_id: RecordIdentifier
    screen_fingerprint: ScreenFingerprint


class ProcedureDivergence(BaseModel):
    """Expected and actual evidence at one failed procedure guard."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    procedure_id: RecordIdentifier
    procedure_index: int = Field(ge=0)
    source_step: int = Field(gt=0)
    guard: Literal["source", "expectation", "runtime"]
    expected_state_id: RecordIdentifier | None = None
    actual_observation_id: RecordIdentifier
    actual_fingerprint: ScreenFingerprint
    reason: str = Field(min_length=1, max_length=2_000)


class ProcedureRepairStepSummary(BaseModel):
    """Compiler-approved work that remains after a successful repair."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    procedure_index: int = Field(ge=0)
    action: ActionKind
    target_expression: str | None = Field(default=None, min_length=1, max_length=1_000)
    expected_outcome: str | None = Field(default=None, min_length=1, max_length=2_000)
    completion_condition_count: int = Field(default=0, ge=0)
    completion_conditions: tuple[str, ...] = Field(default=(), max_length=5)

    @model_validator(mode="after")
    def _validate_completion_conditions(self) -> Self:
        if len(self.completion_conditions) > self.completion_condition_count:
            raise ValueError("Repair milestone summaries exceed the captured condition count.")
        return self


class ProcedureRepairTarget(BaseModel):
    """One compiled state that bounded repair may use to resume replay."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    expected_state_id: RecordIdentifier
    resume_procedure_index: int = Field(ge=0)
    presentation: PresentationKey
    reference_screenshot: ScreenshotReference | None = None
    established_by_action: ActionKind | None = None
    established_by_target_expression: str | None = Field(
        default=None, min_length=1, max_length=1_000
    )
    keyboard_reset_target_expression: str | None = Field(
        default=None, min_length=1, max_length=1_000
    )
    remaining_procedure_step_count: int = Field(default=0, ge=0)
    remaining_step_summaries: tuple[ProcedureRepairStepSummary, ...] = Field(
        default=(), max_length=5
    )
    reason: str = Field(min_length=1, max_length=2_000)

    @model_validator(mode="after")
    def _validate_remaining_steps(self) -> Self:
        if len(self.remaining_step_summaries) > self.remaining_procedure_step_count:
            raise ValueError("Repair step summaries exceed the remaining procedure length.")
        indexes = [item.procedure_index for item in self.remaining_step_summaries]
        if indexes != sorted(set(indexes)):
            raise ValueError("Repair step summaries must have unique ascending indexes.")
        if indexes and indexes[0] != self.resume_procedure_index:
            raise ValueError("Repair step summaries must start at the resume index.")
        return self


class ProcedureRepairEvidence(BaseModel):
    """Bounded actions and deterministic rejoin evidence for one divergence."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    target: ProcedureRepairTarget
    maximum_actions: int = Field(gt=0, le=100)
    action_count: int = Field(ge=0, le=100)
    model_call_count: int = Field(ge=0, le=100)
    succeeded: bool = False
    rejoined_observation_id: RecordIdentifier | None = None
    reason: str = Field(min_length=1, max_length=2_000)

    @model_validator(mode="after")
    def _validate_rejoin_evidence(self) -> Self:
        if self.succeeded != (self.rejoined_observation_id is not None):
            raise ValueError("Successful procedure repair requires a rejoin observation.")
        if self.action_count > self.maximum_actions:
            raise ValueError("Procedure repair actions exceed the configured limit.")
        return self


class ProcedureEnvironmentEvent(BaseModel):
    """One deterministic action that dismissed a trusted system surface."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    surface: TrustedSystemSurface
    action: Literal[ActionKind.BACK] = ActionKind.BACK
    before_observation_id: RecordIdentifier


class ProcedureReplayResult(BaseModel):
    """One compiled procedure execution and its final audit when available."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    procedure_id: RecordIdentifier
    evidence_id: RecordIdentifier
    source_run_id: RecordIdentifier
    execution_mode: Literal["replay"] = "replay"
    visual_policy_version: Literal[1] = 1
    setup_id: RecordIdentifier
    application: EvidenceApplicationIdentity
    terminal: ProcedureReplayTerminal
    procedure_step_count: int = Field(gt=0)
    completed_step_count: int = Field(ge=0)
    failed_procedure_index: int | None = Field(default=None, ge=0)
    milestones: tuple[MilestoneEvidence, ...] = Field(default=(), max_length=100)
    grounding_decisions: tuple[ProcedureGroundingDecision, ...] = Field(
        default=(), max_length=1_000
    )
    divergences: tuple[ProcedureDivergence, ...] = Field(default=(), max_length=100)
    repairs: tuple[ProcedureRepairEvidence, ...] = Field(default=(), max_length=100)
    environment_events: tuple[ProcedureEnvironmentEvent, ...] = Field(default=(), max_length=100)
    verification: GoalVerification | None = None
    reason: str = Field(min_length=1, max_length=4_000)
    procedure_execution_ms: float = Field(default=0, ge=0)
    final_audit_ms: float = Field(default=0, ge=0)
    total_elapsed_ms: float = Field(default=0, ge=0)
    agent_result: AgentResult

    @model_validator(mode="after")
    def _validate_terminal_evidence(self) -> Self:
        if len(self.repairs) > len(self.divergences):
            raise ValueError("Procedure repair history exceeds divergence history.")
        self._validate_completion_count()
        self._validate_terminal_contract()
        self._validate_repair_contract()
        self._validate_milestone_conditions()
        return self

    def _validate_completion_count(self) -> None:
        if self.completed_step_count > self.procedure_step_count:
            raise ValueError("Procedure completion exceeds the compiled step count.")

    def _validate_terminal_contract(self) -> None:
        is_completed_replay = self.terminal in {
            ProcedureReplayTerminal.COMPLETED,
            ProcedureReplayTerminal.REPLAY_COMPLETE,
        }
        if is_completed_replay:
            self._validate_completed_replay()
            return
        if self.terminal is ProcedureReplayTerminal.DIVERGED:
            self._validate_diverged_replay()
            return
        if self.verification is not None:
            raise ValueError("A stopped replay cannot contain final audit evidence.")

    def _validate_completed_replay(self) -> None:
        if self.completed_step_count != self.procedure_step_count:
            raise ValueError("Replay completion requires every procedure expectation to pass.")
        if not self.divergences and self.failed_procedure_index is not None:
            raise ValueError("Clean replay completion cannot contain a failed procedure index.")
        has_complete_repair_history = (
            self.failed_procedure_index is not None
            and bool(self.repairs)
            and all(repair.succeeded for repair in self.repairs)
        )
        if self.divergences and not has_complete_repair_history:
            raise ValueError("Repaired replay completion requires successful repair history.")
        if self.verification is None:
            raise ValueError("Replay completion requires final goal verification.")
        expected_satisfied = self.terminal is ProcedureReplayTerminal.COMPLETED
        if self.verification.satisfied is not expected_satisfied:
            raise ValueError("The replay terminal must match final goal verification.")

    def _validate_diverged_replay(self) -> None:
        if self.failed_procedure_index is None or not self.divergences:
            raise ValueError("Divergence requires the failed procedure guard evidence.")
        if self.verification is not None:
            raise ValueError("Divergence cannot contain final audit evidence.")

    def _validate_repair_contract(self) -> None:
        if self.repairs and not self.divergences:
            raise ValueError("Procedure repair requires preserved divergence evidence.")

    def _validate_milestone_conditions(self) -> None:
        milestone_conditions = [item.completion_condition_index for item in self.milestones]
        if len(milestone_conditions) != len(set(milestone_conditions)):
            raise ValueError("Milestone evidence must contain each completion condition once.")
