"""Records and append-only step persistence for the Device Agent."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from collections.abc import Callable
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yewo.agent.actuation import (
    ACTION_DEFINITIONS,
    ActionHandling,
    ActionKind,
    ActionOutcome,
    ActionOutcomeSignal,
    ActionPolicyLimits,
    ActionProposal,
    ActionRefusal,
    AuthorizedAction,
    DispatchReceipt,
)
from yewo.agent.action_diagnostics import ActionApplicationFailure
from yewo.agent.planning_evidence import (
    PlannerDecisionMetadata,
    PlannerDecisionSource,
)
from yewo.agent.recovery_records import RecoveryBudgetCredit
from yewo.artifacts import (
    ARTIFACTS_DIRECTORY,
    AppendOnlyJsonlWriter,
    ArtifactWriteError,
    RecordIdentifier,
    new_record_identifier,
)
from yewo.devices.inventory import DevicePlatform
from yewo.grounding import (
    GroundingCorrection,
    GroundingEvidence,
    GroundingOutputShape,
    GroundingRequest,
    GroundingTargetOperation,
)
from yewo.observations import Observation
from yewo.runs import RunTerminalCategory
from yewo.runs.errors import RunWorkflowEvidenceError, RunWorkflowInterrupted


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
    targeted_type_available: bool = False
    discovery_mode: bool = False

    @model_validator(mode="after")
    def _validate_unique_declared_values(self) -> Self:
        if len(set(self.declared_value_names)) != len(self.declared_value_names):
            raise ValueError("Declared value names must be unique.")
        return self


class AgentModePolicy(BaseModel):
    """Workflow limits applied to one Device Agent invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action_limits: ActionPolicyLimits
    maximum_steps: int = Field(gt=0, le=100)
    maximum_grounding_requests: int = Field(ge=0, le=100)
    maximum_model_requests: int | None = Field(default=None, gt=0, le=201)
    maximum_duration_seconds: float = Field(default=300, gt=0, le=3_600)
    maximum_settlement_observations: int = Field(default=6, ge=2, le=20)
    settlement_interval_seconds: float = Field(default=0.35, ge=0, le=5)


class AgentUsage(BaseModel):
    """Runtime steps, provider requests, and accepted device dispatches."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    steps: int = Field(default=0, ge=0)
    planning_requests: int = Field(ge=0)
    grounding_requests: int = Field(ge=0)
    dispatched_actions: int = Field(ge=0)


class AgentStep(BaseModel):
    """Canonical evidence for one planner turn."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[2] = 2
    step_id: RecordIdentifier
    step_index: int = Field(gt=0)
    source_observation_id: RecordIdentifier
    proposal: ActionProposal
    decision: PlannerDecisionMetadata = PlannerDecisionMetadata(source=PlannerDecisionSource.MODEL)
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
    recovery_observation_id: RecordIdentifier | None = None
    outcome: ActionOutcome | None = None
    application_failure: ActionApplicationFailure | None = None
    terminal_reason: AgentTerminalReason | None = None

    @model_validator(mode="after")
    def _validate_step_evidence(self) -> Self:
        if self.source_observation_id != self.proposal.source_observation_id:
            raise ValueError("The step source does not match the proposal observation.")
        if (
            self.decision.source is not PlannerDecisionSource.MODEL
            and self.proposal.action not in {ActionKind.DONE, ActionKind.CANNOT_PROGRESS}
            and not self.decision.reused_transition_ids
        ):
            raise ValueError("A reused device action must identify retained transitions.")
        if self.refusal is not None and any(
            item is not None
            for item in (self.authorization, self.receipt, self.outcome, self.application_failure)
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
        if self.application_failure is not None and (self.outcome is None or ActionOutcomeSignal.APP_EXITED not in self.outcome.signals):
            raise ValueError("An application failure requires an exited outcome signal.")
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
            self.recovery_observation_id is not None
            and self.recovery_observation_id in self.settlement_observation_ids
        ):
            raise ValueError("A recovery observation is separate from settlement evidence.")
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
                self.application_failure,
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
    recovery_budget_credit: RecoveryBudgetCredit = RecoveryBudgetCredit()
    steps: tuple[AgentStep, ...]

    @model_validator(mode="after")
    def _validate_recovery_credit(self) -> Self:
        credit = self.recovery_budget_credit
        usage = self.usage
        credited = (credit.steps, credit.planning_requests, credit.grounding_requests, credit.dispatched_actions)
        gross = (usage.steps, usage.planning_requests, usage.grounding_requests, usage.dispatched_actions)
        if any(value > used for value, used in zip(credited, gross, strict=True)):
            raise ValueError("Recovery budget credit cannot exceed gross usage.")
        return self

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

    def __init__(
        self,
        root: Path,
        *,
        identifier_factory: Callable[[], str] = new_record_identifier,
    ) -> None:
        self.root = root
        self._writer = AppendOnlyJsonlWriter(root)
        self._identifier_factory = identifier_factory

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

    def record(
        self,
        step_index: int,
        observation: Observation,
        proposal: ActionProposal,
        decision: PlannerDecisionMetadata,
        terminal_reason: AgentTerminalReason | None = None,
        *,
        target_grounding: GroundingEvidence | None = None,
        destination_grounding: GroundingEvidence | None = None,
        authorization: AuthorizedAction | None = None,
        refusal: ActionRefusal | None = None,
        receipt: DispatchReceipt | None = None,
        settlement_observation_ids: tuple[RecordIdentifier, ...] = (),
        settled_observation_id: RecordIdentifier | None = None,
        recovery_observation_id: RecordIdentifier | None = None,
        outcome: ActionOutcome | None = None,
        application_failure: ActionApplicationFailure | None = None,
    ) -> AgentStep:
        """Build and persist one canonical Device Agent step."""
        step = AgentStep(
            step_id=self._identifier_factory(),
            step_index=step_index,
            source_observation_id=observation.observation_id,
            proposal=proposal,
            decision=decision,
            target_grounding=target_grounding,
            destination_grounding=destination_grounding,
            authorization=authorization,
            refusal=refusal,
            receipt=receipt,
            settlement_observation_ids=settlement_observation_ids,
            settled_observation_id=settled_observation_id,
            recovery_observation_id=recovery_observation_id,
            outcome=outcome,
            application_failure=application_failure,
            terminal_reason=terminal_reason,
        )
        self.persist(step)
        return step

    def record_partial_or_raise(
        self,
        step_index: int,
        observation: Observation,
        proposal: ActionProposal,
        decision: PlannerDecisionMetadata,
        target_grounding: GroundingEvidence | None,
        destination_grounding: GroundingEvidence | None,
        authorization: AuthorizedAction | None,
        receipt: DispatchReceipt | None,
        terminal: AgentTerminalReason,
        steps: list[AgentStep],
        original_error: BaseException,
        *,
        settlement_observation_ids: tuple[RecordIdentifier, ...] = (),
    ) -> None:
        """Persist partial evidence without hiding the original workflow failure."""
        try:
            step = self.record(
                step_index,
                observation,
                proposal,
                decision,
                terminal,
                target_grounding=target_grounding,
                destination_grounding=destination_grounding,
                authorization=authorization,
                receipt=receipt,
                settlement_observation_ids=settlement_observation_ids,
            )
        except AgentStepPersistenceError as persistence_error:
            raise RunWorkflowEvidenceError(
                interrupted=isinstance(original_error, RunWorkflowInterrupted),
                original_error=original_error,
            ) from persistence_error
        steps.append(step)


def agent_terminal_category(
    terminal: AgentTerminalReason,
    *,
    done_category: RunTerminalCategory = RunTerminalCategory.COMPLETED,
) -> RunTerminalCategory:
    """Translate stable Device Agent terminals at a workflow boundary."""
    if terminal is AgentTerminalReason.DONE_REQUESTED:
        return done_category
    return {
        AgentTerminalReason.CANNOT_PROGRESS: RunTerminalCategory.INCOMPLETE,
        AgentTerminalReason.BUDGET_EXHAUSTED: RunTerminalCategory.BUDGET_EXHAUSTED,
        AgentTerminalReason.APPLICATION_EXITED: RunTerminalCategory.APPLICATION_FAILURE,
        AgentTerminalReason.APPLICATION_INACTIVE: RunTerminalCategory.INCOMPLETE,
        AgentTerminalReason.DEVICE_REFUSED: RunTerminalCategory.INCOMPLETE,
    }[terminal]


def invocation_budget_reason(
    policy: AgentModePolicy,
    elapsed_seconds: float,
    model_requests: int,
    requested_model_requests: int,
) -> str | None:
    """Return the first invocation limit blocking the next planning operation."""
    if elapsed_seconds >= policy.maximum_duration_seconds:
        return "The Device Agent duration budget is exhausted."
    if (
        policy.maximum_model_requests is not None
        and model_requests + requested_model_requests > policy.maximum_model_requests
    ):
        return "The Device Agent model-request budget is exhausted."
    return None


def grounding_budget_reason(
    policy: AgentModePolicy,
    planning_requests: int,
    grounding_requests: int,
    requested_groundings: int,
) -> str | None:
    """Return the first model or grounding limit blocked by a proposal."""
    if (
        policy.maximum_model_requests is not None
        and planning_requests + grounding_requests + requested_groundings
        > policy.maximum_model_requests
    ):
        return "The Device Agent model-request budget is exhausted."
    if grounding_requests + requested_groundings > policy.maximum_grounding_requests:
        return "The visual-grounding request budget is exhausted."
    return None


def available_runtime_actions(
    limits: ActionPolicyLimits,
    observation: Observation,
    *,
    targeted_type_available: bool = False,
) -> frozenset[ActionKind]:
    """Return executable actions available for the current observation."""
    return frozenset(
        action
        for action in limits.allowed_actions & limits.supported_actions
        if ACTION_DEFINITIONS[action].handling is ActionHandling.EXECUTE
        and ACTION_DEFINITIONS[action].side_effect in limits.allowed_side_effects
        and (action is not ActionKind.BACK or limits.allow_exploratory_back)
        and (action is not ActionKind.BACK or observation.device.platform is DevicePlatform.ANDROID)
        and (action is not ActionKind.DISMISS_KEYBOARD or observation.keyboard_visible is True)
        and (
            not ACTION_DEFINITIONS[action].requires_keyboard_visible
            or observation.keyboard_visible is True
            or (action is ActionKind.TYPE and targeted_type_available)
        )
        and not (
            observation.keyboard_visible is True
            and ACTION_DEFINITIONS[action].blocked_when_keyboard_visible
        )
    )


def proposal_terminal(proposal: ActionProposal) -> AgentTerminalReason | None:
    """Translate a planner terminal proposal into a runtime terminal."""
    if proposal.action is ActionKind.DONE:
        return AgentTerminalReason.DONE_REQUESTED
    if proposal.action is ActionKind.CANNOT_PROGRESS:
        return AgentTerminalReason.CANNOT_PROGRESS
    return None


def required_grounding_requests(proposal: ActionProposal) -> int:
    """Count visual grounding calls required by one semantic proposal."""
    return int(proposal.target_expression is not None) + int(
        proposal.destination_expression is not None
    )


def build_runtime_grounding_request(
    grounding_id: RecordIdentifier,
    expression: str | None,
    operation: GroundingTargetOperation | None,
    observation: Observation,
    correction: GroundingCorrection | None,
) -> GroundingRequest | None:
    """Build one visual-grounding request from the current observation."""
    if expression is None or operation is None:
        return None
    output_shape = (
        GroundingOutputShape.REGION
        if operation is GroundingTargetOperation.SCROLL_REGION
        else GroundingOutputShape.POINT
    )
    return GroundingRequest(
        grounding_id=grounding_id,
        screenshot=observation.redacted_screenshot,
        source_fingerprint=observation.screen_fingerprint,
        source_orientation=observation.orientation,
        source_keyboard_visible=observation.keyboard_visible,
        expression=expression,
        operation=operation,
        output_shape=output_shape,
        excluded_regions=observation.system_regions,
        correction=correction,
    )


def remaining_mode_policy(
    policy: AgentModePolicy,
    usage: AgentUsage,
    elapsed_seconds: float,
) -> AgentModePolicy | None:
    """Subtract one invocation from a shared mode policy."""
    remaining_steps = policy.maximum_steps - usage.steps
    remaining_grounding = policy.maximum_grounding_requests - usage.grounding_requests
    remaining_actions = policy.action_limits.maximum_actions - usage.dispatched_actions
    remaining_duration = policy.maximum_duration_seconds - elapsed_seconds
    if min(remaining_steps, remaining_actions) <= 0 or remaining_grounding < 0:
        return None
    if remaining_duration <= 0:
        return None
    remaining_model_requests = None
    if policy.maximum_model_requests is not None:
        remaining_model_requests = policy.maximum_model_requests - (
            usage.planning_requests + usage.grounding_requests
        )
        if remaining_model_requests <= 0:
            return None
    return policy.model_copy(
        update={
            "maximum_steps": remaining_steps,
            "maximum_grounding_requests": remaining_grounding,
            "maximum_model_requests": remaining_model_requests,
            "maximum_duration_seconds": remaining_duration,
            "action_limits": policy.action_limits.model_copy(
                update={"maximum_actions": remaining_actions}
            ),
        }
    )


def goal_verification_budget_reason(
    policy: AgentModePolicy,
    model_requests: int,
    elapsed_seconds: float,
) -> str | None:
    """Return why a final auditor request cannot start."""
    if elapsed_seconds >= policy.maximum_duration_seconds:
        return "The duration budget ended before final verification."
    if (
        policy.maximum_model_requests is not None
        and model_requests >= policy.maximum_model_requests
    ):
        return "The model-request budget ended before final verification."
    return None
