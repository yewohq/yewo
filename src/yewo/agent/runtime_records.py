"""Records and append-only step persistence for the Device Agent."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yewo.agent.actuation import (
    ActionKind,
    ActionOutcome,
    ActionPolicyLimits,
    ActionProposal,
    ActionRefusal,
    AuthorizedAction,
    DispatchReceipt,
)
from yewo.artifacts import (
    ARTIFACTS_DIRECTORY,
    AppendOnlyJsonlWriter,
    ArtifactWriteError,
    RecordIdentifier,
)
from yewo.grounding import GroundingEvidence


TaskText = Annotated[str, Field(min_length=1, max_length=4_000)]


class AgentTerminalReason(StrEnum):
    """Why the reusable device loop stopped."""

    DONE_REQUESTED = "done_requested"
    CANNOT_PROGRESS = "cannot_progress"
    BUDGET_EXHAUSTED = "budget_exhausted"
    APPLICATION_EXITED = "application_exited"
    APPLICATION_INACTIVE = "application_inactive"
    DEVICE_REFUSED = "device_refused"
    INTERRUPTED = "interrupted"
    RUNTIME_FAILURE = "runtime_failure"


class AgentTask(BaseModel):
    """Workflow-neutral task information supplied to the shared runtime."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    objective: TaskText
    known_facts: tuple[TaskText, ...] = Field(default=(), max_length=100)
    open_work: tuple[TaskText, ...] = Field(default=(), max_length=100)
    completion_conditions: tuple[TaskText, ...] = Field(min_length=1, max_length=100)
    declared_value_names: tuple[RecordIdentifier, ...] = Field(default=(), max_length=100)

    @model_validator(mode="after")
    def _validate_unique_declared_values(self) -> Self:
        if len(set(self.declared_value_names)) != len(self.declared_value_names):
            raise ValueError("Declared value names must be unique.")
        return self


class AgentModePolicy(BaseModel):
    """Workflow limits applied to one Device Agent invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action_limits: ActionPolicyLimits
    maximum_steps: int = Field(gt=0, le=50)
    maximum_grounding_requests: int = Field(ge=0, le=100)
    maximum_settlement_observations: int = Field(default=6, ge=2, le=20)
    settlement_interval_seconds: float = Field(default=0.35, ge=0, le=5)


class AgentUsage(BaseModel):
    """Logical runtime requests and accepted device dispatches."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    planning_requests: int = Field(ge=0)
    grounding_requests: int = Field(ge=0)
    dispatched_actions: int = Field(ge=0)


class AgentStep(BaseModel):
    """Canonical evidence for one planner turn."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    step_id: RecordIdentifier
    step_index: int = Field(gt=0)
    source_observation_id: RecordIdentifier
    proposal: ActionProposal
    target_grounding: GroundingEvidence | None = None
    destination_grounding: GroundingEvidence | None = None
    authorization: AuthorizedAction | None = None
    refusal: ActionRefusal | None = None
    receipt: DispatchReceipt | None = None
    settlement_observation_ids: tuple[RecordIdentifier, ...] = Field(
        default=(),
        max_length=20,
    )
    settled_observation_id: RecordIdentifier | None = None
    outcome: ActionOutcome | None = None
    terminal_reason: AgentTerminalReason | None = None

    @model_validator(mode="after")
    def _validate_step_evidence(self) -> Self:
        if self.source_observation_id != self.proposal.source_observation_id:
            raise ValueError("The step source does not match the proposal observation.")
        if self.refusal is not None and any(
            item is not None for item in (self.authorization, self.receipt, self.outcome)
        ):
            raise ValueError("A refused step cannot contain dispatch evidence.")
        if self.refusal is not None and self.refusal.proposal_id != self.proposal.proposal_id:
            raise ValueError("The step refusal does not match the proposal.")
        if self.authorization is not None and (
            self.authorization.proposal_id != self.proposal.proposal_id
            or self.authorization.source_observation_id != self.source_observation_id
        ):
            raise ValueError("The step authorization does not match its proposal evidence.")
        if self.receipt is not None and self.authorization is None:
            raise ValueError("A dispatch receipt requires an authorization.")
        if (
            self.receipt is not None
            and self.authorization is not None
            and (
                self.receipt.authorization_id != self.authorization.authorization_id
                or self.receipt.proposal_id != self.proposal.proposal_id
            )
        ):
            raise ValueError("The step receipt does not match its authorization.")
        if self.outcome is not None and self.receipt is None:
            raise ValueError("An action outcome requires a dispatch receipt.")
        if (
            self.outcome is not None
            and self.settled_observation_id != self.outcome.settled_observation_id
        ):
            raise ValueError("The step and outcome must name the same settled observation.")
        if self.settled_observation_id is not None and (
            not self.settlement_observation_ids
            or self.settlement_observation_ids[-1] != self.settled_observation_id
        ):
            raise ValueError("The settled observation must end the settlement observation chain.")
        if (
            self.outcome is not None
            and self.authorization is not None
            and self.receipt is not None
            and (
                self.outcome.authorization_id != self.authorization.authorization_id
                or self.outcome.receipt_id != self.receipt.receipt_id
                or self.outcome.source_observation_id != self.source_observation_id
            )
        ):
            raise ValueError("The step outcome does not match its dispatch evidence.")
        if self.proposal.action in {ActionKind.DONE, ActionKind.CANNOT_PROGRESS} and any(
            item is not None
            for item in (
                self.target_grounding,
                self.destination_grounding,
                self.authorization,
                self.refusal,
                self.receipt,
                self.outcome,
            )
        ):
            raise ValueError("A planner terminal cannot contain device-action evidence.")
        expected_terminal = {
            ActionKind.DONE: AgentTerminalReason.DONE_REQUESTED,
            ActionKind.CANNOT_PROGRESS: AgentTerminalReason.CANNOT_PROGRESS,
        }.get(self.proposal.action)
        if expected_terminal is not None and self.terminal_reason is not expected_terminal:
            raise ValueError("The planner terminal does not match the step terminal reason.")
        return self


class AgentResult(BaseModel):
    """Typed conclusion and evidence returned to the calling workflow."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    terminal_reason: AgentTerminalReason
    reason: str = Field(min_length=1)
    final_observation_id: RecordIdentifier
    usage: AgentUsage
    steps: tuple[AgentStep, ...]

    @model_validator(mode="after")
    def _validate_returnable_terminal(self) -> Self:
        if self.terminal_reason in {
            AgentTerminalReason.INTERRUPTED,
            AgentTerminalReason.RUNTIME_FAILURE,
        }:
            raise ValueError(
                "Interruptions and runtime failures must propagate to the coordinator."
            )
        return self


class AgentStepPersistenceError(RuntimeError):
    """The runtime could not append canonical step evidence."""


class AgentStepStore:
    """Append canonical Device Agent turns to one JSONL ledger."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._writer = AppendOnlyJsonlWriter(root)

    def persist(self, step: AgentStep) -> Path:
        """Append one immutable step without replacing prior evidence."""
        relative_path = ARTIFACTS_DIRECTORY / "agent" / "steps.jsonl"
        try:
            self._writer.append(
                relative_path,
                step.model_dump(mode="json"),
                identity_field="step_id",
            )
        except ArtifactWriteError as error:
            raise AgentStepPersistenceError(
                "Could not persist Device Agent step evidence."
            ) from error
        return relative_path
