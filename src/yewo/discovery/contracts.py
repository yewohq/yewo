"""Typed observations, capabilities, frontiers, budgets, and terminals for discovery."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from yewo.agent.memory.replay_store import RelativeBounds
from yewo.discovery.artifacts import (
    ArtifactReference,
    DiscoveryArtifact,
    DiscoveryArtifactHeader,
    DiscoveryContract,
)
from yewo.identity import new_uuid


class CapabilityState(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    ROUTE_VERIFIED = "route_verified"
    CAPABILITY_OBSERVED = "capability_observed"
    UNAVAILABLE = "unavailable"
    BLOCKED_BY_POLICY = "blocked_by_policy"
    BLOCKED_BY_SETUP = "blocked_by_setup"
    AMBIGUOUS = "ambiguous"
    NOT_EVALUATED = "not_evaluated"


SUCCESSFUL_CAPABILITY_STATES = frozenset(
    {CapabilityState.ROUTE_VERIFIED, CapabilityState.CAPABILITY_OBSERVED}
)
TERMINAL_CAPABILITY_STATES = frozenset(
    state
    for state in CapabilityState
    if state not in (CapabilityState.PENDING, CapabilityState.ACTIVE)
)


class CapabilityQuestion(DiscoveryContract):
    """One semantic capability discovery must settle exactly once."""

    question_id: str = Field(default_factory=new_uuid)
    intent: str = Field(min_length=1)
    required: bool = True
    state: CapabilityState = CapabilityState.PENDING
    evidence_observation_ids: tuple[str, ...] = ()

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_CAPABILITY_STATES

    def activate(self) -> "CapabilityQuestion":
        if self.state is not CapabilityState.PENDING:
            raise ValueError("Only a pending capability question can become active.")
        return self.model_copy(update={"state": CapabilityState.ACTIVE})

    def close(
        self,
        state: CapabilityState,
        evidence_observation_ids: tuple[str, ...] = (),
    ) -> "CapabilityQuestion":
        if self.is_terminal:
            raise ValueError("A terminal capability question is immutable.")
        if state not in TERMINAL_CAPABILITY_STATES:
            raise ValueError("A capability question must close with a terminal state.")
        if self.state is CapabilityState.PENDING and state not in (
            CapabilityState.BLOCKED_BY_POLICY,
            CapabilityState.BLOCKED_BY_SETUP,
            CapabilityState.NOT_EVALUATED,
        ):
            raise ValueError("This capability terminal requires an active question.")
        if state in SUCCESSFUL_CAPABILITY_STATES and not evidence_observation_ids:
            raise ValueError("A successful capability state requires accepted evidence.")
        return self.model_copy(
            update={
                "state": state,
                "evidence_observation_ids": evidence_observation_ids,
            }
        )


class ObservedControl(DiscoveryContract):
    """Secret-safe control evidence from one exact observation."""

    control_identity: str
    role: str
    bounds: RelativeBounds
    text_digest: str = ""
    actionable: bool


class DiscoveryObservation(DiscoveryContract):
    """One exact settled app observation accepted by discovery policy."""

    observation_id: str
    screen_state_id: str
    screen_family_candidate_id: str | None = None
    phash: str
    controls: tuple[ObservedControl, ...] = ()
    keyboard_visible: bool | None = None
    overlay_visible: bool | None = None
    foreground_app_id: str | None = None
    evidence: tuple[ArtifactReference, ...] = ()


class CapabilityCandidate(DiscoveryContract):
    """An observed capability proposal that has no requirement authority."""

    candidate_id: str = Field(default_factory=new_uuid)
    capability: str = Field(min_length=1)
    question_ids: tuple[str, ...] = ()
    observation_ids: tuple[str, ...] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class FrontierState(str, Enum):
    AVAILABLE = "available"
    ATTEMPTED = "attempted"
    VERIFIED = "verified"
    POLICY_REFUSED = "policy_refused"


class SideEffectClass(str, Enum):
    READ_ONLY = "read_only"
    REVERSIBLE = "reversible"
    STATE_MUTATING = "state_mutating"
    EXTERNAL_COMMUNICATION = "external_communication"
    FINANCIAL = "financial"
    DESTRUCTIVE = "destructive"
    SECURITY_SENSITIVE = "security_sensitive"
    UNKNOWN = "unknown"


class DiscoveryTransitionEvidence(DiscoveryContract):
    """Accepted local edge evidence; it is not scenario or product evidence."""

    transition_identity: str
    source_observation_id: str
    destination_observation_id: str
    control_identity: str
    action_kind: str
    authorization_receipt_id: str
    dispatch_receipt_id: str
    settlement_receipt_id: str
    side_effect: SideEffectClass
    reversible: bool


class RelevantFrontierEntry(DiscoveryContract):
    """One control owned by its source observation and relevant questions."""

    frontier_id: str = Field(default_factory=new_uuid)
    source_observation_id: str
    control_identity: str
    relevant_question_ids: tuple[str, ...] = Field(min_length=1)
    state: FrontierState = FrontierState.AVAILABLE
    transition_identity: str | None = None

    def close(
        self,
        state: Literal[
            FrontierState.ATTEMPTED,
            FrontierState.VERIFIED,
            FrontierState.POLICY_REFUSED,
        ],
        transition_identity: str | None = None,
    ) -> "RelevantFrontierEntry":
        if self.state is not FrontierState.AVAILABLE:
            raise ValueError("A closed discovery frontier entry is immutable.")
        if state is FrontierState.VERIFIED and transition_identity is None:
            raise ValueError("A verified frontier entry requires a transition identity.")
        return self.model_copy(
            update={"state": state, "transition_identity": transition_identity}
        )


class DiscoveryBudgetDimension(str, Enum):
    OBSERVATIONS = "observations"
    ACTIONS = "actions"
    MODEL_PROPOSALS = "model_proposals"
    RESETS = "resets"
    DURATION_MS = "duration_ms"
    STATE_DEPTH = "state_depth"
    COST_MICRO_USD = "cost_micro_usd"
    SIDE_EFFECT_ACTIONS = "side_effect_actions"


class DiscoveryBudgets(DiscoveryContract):
    """Independent ceilings; spending one dimension cannot borrow from another."""

    observations: int = Field(gt=0)
    actions: int = Field(ge=0)
    model_proposals: int = Field(ge=0)
    resets: int = Field(ge=0)
    duration_ms: int = Field(gt=0)
    state_depth: int = Field(ge=0)
    cost_micro_usd: int = Field(ge=0)
    side_effect_actions: int = Field(ge=0)
    attempts_per_transition: int = Field(gt=0)


class TransitionAttemptUsage(DiscoveryContract):
    transition_identity: str
    attempts: int = Field(ge=0)


class DiscoveryBudgetUsage(DiscoveryContract):
    observations: int = Field(default=0, ge=0)
    actions: int = Field(default=0, ge=0)
    model_proposals: int = Field(default=0, ge=0)
    resets: int = Field(default=0, ge=0)
    duration_ms: int = Field(default=0, ge=0)
    state_depth: int = Field(default=0, ge=0)
    cost_micro_usd: int = Field(default=0, ge=0)
    side_effect_actions: int = Field(default=0, ge=0)
    transition_attempts: tuple[TransitionAttemptUsage, ...] = ()

    @model_validator(mode="after")
    def _validate_transition_attempts(self) -> "DiscoveryBudgetUsage":
        identities = [usage.transition_identity for usage in self.transition_attempts]
        if len(identities) != len(set(identities)):
            raise ValueError("Transition attempt identities must be unique.")
        return self

    def exhausted_dimension(
        self, limits: DiscoveryBudgets
    ) -> DiscoveryBudgetDimension | None:
        for dimension in DiscoveryBudgetDimension:
            if getattr(self, dimension.value) >= getattr(limits, dimension.value):
                return dimension
        return None

    def can_attempt_transition(self, transition_identity: str, limits: DiscoveryBudgets) -> bool:
        attempts = next(
            (
                usage.attempts
                for usage in self.transition_attempts
                if usage.transition_identity == transition_identity
            ),
            0,
        )
        return attempts < limits.attempts_per_transition
class DiscoveryTerminal(str, Enum):
    DISCOVERY_COMPLETE = "discovery_complete"
    DISCOVERY_INCOMPLETE = "discovery_incomplete"
    DISCOVERY_BLOCKED = "discovery_blocked"
    DISCOVERY_BUDGET_EXHAUSTED = "discovery_budget_exhausted"
    DISCOVERY_INTERRUPTED = "discovery_interrupted"


class DiscoveryBlocker(str, Enum):
    POLICY = "policy"
    SETUP = "setup"
    RESET = "reset"
    INFRASTRUCTURE = "infrastructure"


class DiscoveryTerminalRecord(DiscoveryContract):
    terminal: DiscoveryTerminal
    reason: str
    exhausted_budget: DiscoveryBudgetDimension | None = None
    blocker: DiscoveryBlocker | None = None

    @model_validator(mode="after")
    def _validate_terminal_reason(self) -> "DiscoveryTerminalRecord":
        if self.terminal is DiscoveryTerminal.DISCOVERY_BUDGET_EXHAUSTED:
            if self.exhausted_budget is None or self.blocker is not None:
                raise ValueError("A budget terminal requires exactly one exhausted budget.")
            expected = f"The {self.exhausted_budget.value} discovery budget was exhausted."
        elif self.terminal is DiscoveryTerminal.DISCOVERY_BLOCKED:
            if self.blocker is None or self.exhausted_budget is not None:
                raise ValueError("A blocked terminal requires exactly one blocker.")
            expected = f"Discovery was blocked by {self.blocker.value}."
        else:
            if self.exhausted_budget is not None or self.blocker is not None:
                raise ValueError("This discovery terminal cannot carry a budget or blocker.")
            expected = {
                DiscoveryTerminal.DISCOVERY_COMPLETE: (
                    "Every required discovery capability was verified."
                ),
                DiscoveryTerminal.DISCOVERY_INCOMPLETE: (
                    "The relevant safe frontier was exhausted before every capability "
                    "was verified."
                ),
                DiscoveryTerminal.DISCOVERY_INTERRUPTED: (
                    "Discovery was interrupted externally."
                ),
            }[self.terminal]
        if self.reason != expected:
            raise ValueError("A discovery terminal reason must use its canonical wording.")
        return self


class DiscoveryTerminationSignals(DiscoveryContract):
    externally_interrupted: bool = False
    exhausted_budget: DiscoveryBudgetDimension | None = None
    blocker: DiscoveryBlocker | None = None
    relevant_frontier_remaining: bool = True


class DiscoveryRun(DiscoveryArtifact):
    """Non-verdict-bearing evidence and state for one bounded discovery run."""

    header: DiscoveryArtifactHeader
    questions: tuple[CapabilityQuestion, ...] = Field(min_length=1)
    observations: tuple[DiscoveryObservation, ...] = ()
    transitions: tuple[DiscoveryTransitionEvidence, ...] = ()
    capability_candidates: tuple[CapabilityCandidate, ...] = ()
    frontier: tuple[RelevantFrontierEntry, ...] = ()
    budgets: DiscoveryBudgets
    usage: DiscoveryBudgetUsage = Field(default_factory=DiscoveryBudgetUsage)
    terminal: DiscoveryTerminalRecord | None = None

    @model_validator(mode="after")
    def _validate_contract(self) -> "DiscoveryRun":
        if self.header.schema_name != "discovery-run":
            raise ValueError("A discovery run requires the discovery-run schema.")
        question_ids = [question.question_id for question in self.questions]
        observation_ids = [observation.observation_id for observation in self.observations]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("Capability question identifiers must be unique.")
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("Discovery observation identifiers must be unique.")
        known_questions = set(question_ids)
        questions_by_id = {question.question_id: question for question in self.questions}
        known_observations = set(observation_ids)
        self._validate_evidence_graph(known_questions, known_observations, questions_by_id)
        for question in self.questions:
            if not set(question.evidence_observation_ids).issubset(known_observations):
                raise ValueError("A capability question requires accepted observations.")
        for dimension in DiscoveryBudgetDimension:
            if getattr(self.usage, dimension.value) > getattr(
                self.budgets, dimension.value
            ):
                raise ValueError("Discovery budget usage cannot exceed its limit.")
        if any(
            usage.attempts > self.budgets.attempts_per_transition
            for usage in self.usage.transition_attempts
        ):
            raise ValueError("Transition attempts cannot exceed their independent limit.")
        if self.terminal is not None and any(
            not question.is_terminal for question in self.questions
        ):
            raise ValueError("A terminal discovery run cannot retain an open question.")
        required = tuple(question for question in self.questions if question.required)
        if (
            self.terminal is not None
            and self.terminal.terminal is DiscoveryTerminal.DISCOVERY_COMPLETE
            and any(
                question.state not in SUCCESSFUL_CAPABILITY_STATES for question in required
            )
        ):
            raise ValueError("A complete discovery requires every required capability.")
        return self

    def _validate_evidence_graph(
        self,
        known_questions: set[str],
        known_observations: set[str],
        questions_by_id: dict[str, CapabilityQuestion],
    ) -> None:
        transition_ids = [transition.transition_identity for transition in self.transitions]
        if len(transition_ids) != len(set(transition_ids)):
            raise ValueError("Discovery transition identities must be unique.")
        for transition in self.transitions:
            referenced = {
                transition.source_observation_id,
                transition.destination_observation_id,
            }
            if not referenced.issubset(known_observations):
                raise ValueError("A discovery transition requires both exact observations.")
        self._validate_frontier(known_questions, known_observations, questions_by_id)
        candidate_ids = [candidate.candidate_id for candidate in self.capability_candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("Capability candidate identifiers must be unique.")
        for candidate in self.capability_candidates:
            if not set(candidate.question_ids).issubset(known_questions):
                raise ValueError("A capability candidate names an unknown question.")
            if not set(candidate.observation_ids).issubset(known_observations):
                raise ValueError("A capability candidate requires accepted observations.")

    def _validate_frontier(
        self,
        known_questions: set[str],
        known_observations: set[str],
        questions_by_id: dict[str, CapabilityQuestion],
    ) -> None:
        frontier_ids = [entry.frontier_id for entry in self.frontier]
        if len(frontier_ids) != len(set(frontier_ids)):
            raise ValueError("Discovery frontier identifiers must be unique.")
        for entry in self.frontier:
            if entry.source_observation_id not in known_observations:
                raise ValueError("A frontier entry requires its source observation.")
            if not set(entry.relevant_question_ids).issubset(known_questions):
                raise ValueError("A frontier entry names an unknown capability question.")
            if entry.state is FrontierState.AVAILABLE and any(
                questions_by_id[question_id].is_terminal
                for question_id in entry.relevant_question_ids
            ):
                raise ValueError("Available frontier work requires an unresolved question.")


def _close_open_questions(
    questions: tuple[CapabilityQuestion, ...], state: CapabilityState
) -> tuple[CapabilityQuestion, ...]:
    closed = []
    for question in questions:
        if question.is_terminal:
            closed.append(question)
            continue
        if question.state is CapabilityState.PENDING and state not in (
            CapabilityState.BLOCKED_BY_POLICY,
            CapabilityState.BLOCKED_BY_SETUP,
            CapabilityState.NOT_EVALUATED,
        ):
            question = question.activate()
        closed.append(question.close(state))
    return tuple(closed)


def finalize_discovery(
    run: DiscoveryRun, signals: DiscoveryTerminationSignals
) -> DiscoveryRun:
    """Select one terminal by fixed precedence and close every open question."""
    if run.terminal is not None:
        raise ValueError("A terminal discovery run is immutable.")

    questions = run.questions
    exhausted = signals.exhausted_budget
    required = tuple(question for question in questions if question.required)
    all_required_succeeded = all(
        question.state in SUCCESSFUL_CAPABILITY_STATES for question in required
    )

    if signals.externally_interrupted:
        terminal = DiscoveryTerminalRecord(
            terminal=DiscoveryTerminal.DISCOVERY_INTERRUPTED,
            reason="Discovery was interrupted externally.",
        )
        questions = _close_open_questions(questions, CapabilityState.NOT_EVALUATED)
    elif exhausted is not None:
        terminal = DiscoveryTerminalRecord(
            terminal=DiscoveryTerminal.DISCOVERY_BUDGET_EXHAUSTED,
            reason=f"The {exhausted.value} discovery budget was exhausted.",
            exhausted_budget=exhausted,
        )
        questions = _close_open_questions(questions, CapabilityState.NOT_EVALUATED)
    elif signals.blocker is not None:
        terminal = DiscoveryTerminalRecord(
            terminal=DiscoveryTerminal.DISCOVERY_BLOCKED,
            reason=f"Discovery was blocked by {signals.blocker.value}.",
            blocker=signals.blocker,
        )
        blocked_state = {
            DiscoveryBlocker.POLICY: CapabilityState.BLOCKED_BY_POLICY,
            DiscoveryBlocker.SETUP: CapabilityState.BLOCKED_BY_SETUP,
        }.get(signals.blocker, CapabilityState.NOT_EVALUATED)
        questions = _close_open_questions(questions, blocked_state)
    elif all_required_succeeded:
        terminal = DiscoveryTerminalRecord(
            terminal=DiscoveryTerminal.DISCOVERY_COMPLETE,
            reason="Every required discovery capability was verified.",
        )
        questions = _close_open_questions(questions, CapabilityState.NOT_EVALUATED)
    elif not signals.relevant_frontier_remaining:
        terminal = DiscoveryTerminalRecord(
            terminal=DiscoveryTerminal.DISCOVERY_INCOMPLETE,
            reason="The relevant safe frontier was exhausted before every capability was verified.",
        )
        questions = _close_open_questions(questions, CapabilityState.UNAVAILABLE)
    else:
        raise ValueError("Discovery cannot terminate while relevant safe work remains.")

    return run.model_copy(update={"questions": questions, "terminal": terminal})
