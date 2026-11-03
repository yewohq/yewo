"""Execution records for compiler-approved replay procedures."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yewo.agent.actuation import ActionKind
from yewo.agent.planning.goal_verifier import GoalVerification
from yewo.agent.runtime.records import AgentResult
from yewo.artifacts import EvidenceApplicationIdentity, RecordIdentifier
from yewo.devices.application_lifecycle import TrustedSystemSurface
from yewo.observations import DeviceOrientation, ScreenFingerprint
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


class ProcedureReplayTerminal(StrEnum):
    """Why compiled procedure execution and its optional audit stopped."""

    COMPLETED = "completed"
    REPLAY_COMPLETE = "replay_complete"
    DIVERGED = "diverged"
    BUDGET_EXHAUSTED = "budget_exhausted"
    APPLICATION_FAILURE = "application_failure"
    DEVICE_REFUSED = "device_refused"


class ProcedureGroundingDecisionCode(StrEnum):
    """Why one compiled grounding was reused or delegated."""

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


class ProcedureRepairTarget(BaseModel):
    """One compiled state that bounded repair may use to resume replay."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    expected_state_id: RecordIdentifier
    resume_procedure_index: int = Field(ge=0)
    foreground_application_id: str | None = Field(default=None, min_length=1)
    platform_context: str | None = Field(default=None, min_length=1)
    keyboard_visible: bool | None = None
    orientation: DeviceOrientation | None = None
    established_by_action: ActionKind | None = None
    established_by_target_expression: str | None = Field(
        default=None, min_length=1, max_length=1_000
    )
    keyboard_reset_target_expression: str | None = Field(
        default=None, min_length=1, max_length=1_000
    )
    next_action: ActionKind | None = None
    next_target_expression: str | None = Field(default=None, min_length=1, max_length=1_000)
    reason: str = Field(min_length=1, max_length=2_000)


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
        if self.model_call_count > self.action_count:
            raise ValueError("Procedure repair model calls exceed repair actions.")
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

    schema_version: Literal[1] = 1
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
    divergence: ProcedureDivergence | None = None
    repair: ProcedureRepairEvidence | None = None
    divergences: tuple[ProcedureDivergence, ...] = Field(default=(), max_length=100)
    repairs: tuple[ProcedureRepairEvidence, ...] = Field(default=(), max_length=100)
    environment_events: tuple[ProcedureEnvironmentEvent, ...] = Field(default=(), max_length=100)
    verification: GoalVerification | None = None
    reason: str = Field(min_length=1, max_length=4_000)
    procedure_execution_ms: float = Field(default=0, ge=0)
    final_audit_ms: float = Field(default=0, ge=0)
    total_elapsed_ms: float = Field(default=0, ge=0)
    agent_result: AgentResult

    @model_validator(mode="before")
    @classmethod
    def _normalize_evidence_history(cls, value: object) -> object:
        """Lift a single current record into its complete one-item history."""
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        divergence = normalized.get("divergence")
        repair = normalized.get("repair")
        if divergence is not None and "divergences" not in normalized:
            normalized["divergences"] = (divergence,)
        if repair is not None and "repairs" not in normalized:
            normalized["repairs"] = (repair,)
        return normalized

    @model_validator(mode="after")
    def _validate_terminal_evidence(self) -> Self:
        self._validate_current_history_records()
        self._validate_completion_count()
        self._validate_terminal_contract()
        self._validate_repair_contract()
        self._validate_milestone_conditions()
        return self

    def _validate_current_history_records(self) -> None:
        if self.divergence != (self.divergences[-1] if self.divergences else None):
            raise ValueError("The current divergence must equal the final divergence record.")
        if self.repair != (self.repairs[-1] if self.repairs else None):
            raise ValueError("The current repair must equal the final repair record.")
        if len(self.repairs) > len(self.divergences):
            raise ValueError("Procedure repair history exceeds divergence history.")

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
        if self.divergence is None and self.failed_procedure_index is not None:
            raise ValueError("Clean replay completion cannot contain a failed procedure index.")
        has_complete_repair_history = (
            self.failed_procedure_index is not None
            and bool(self.repairs)
            and all(repair.succeeded for repair in self.repairs)
        )
        if self.divergence is not None and not has_complete_repair_history:
            raise ValueError("Repaired replay completion requires successful repair history.")
        if self.verification is None:
            raise ValueError("Replay completion requires final goal verification.")
        expected_satisfied = self.terminal is ProcedureReplayTerminal.COMPLETED
        if self.verification.satisfied is not expected_satisfied:
            raise ValueError("The replay terminal must match final goal verification.")

    def _validate_diverged_replay(self) -> None:
        if self.failed_procedure_index is None or self.divergence is None:
            raise ValueError("Divergence requires the failed procedure guard evidence.")
        if self.verification is not None:
            raise ValueError("Divergence cannot contain final audit evidence.")

    def _validate_repair_contract(self) -> None:
        if self.repair is not None and self.divergence is None:
            raise ValueError("Procedure repair requires preserved divergence evidence.")

    def _validate_milestone_conditions(self) -> None:
        milestone_conditions = [item.completion_condition_index for item in self.milestones]
        if len(milestone_conditions) != len(set(milestone_conditions)):
            raise ValueError("Milestone evidence must contain each completion condition once.")
