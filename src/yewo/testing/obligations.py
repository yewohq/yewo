"""Deterministic routing and settlement of scenario obligations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Sequence

from pydantic import BaseModel, Field, model_validator

from yewo.identity import derived_uuid


class ObligationKind(str, Enum):
    SCREEN_ASSERTION = "screen_assertion"
    TRANSITION_ASSERTION = "transition_assertion"
    ACTION_POSTCONDITION = "action_postcondition"
    MEASUREMENT = "measurement"
    RUN_INVARIANT = "run_invariant"
    RUN_FINAL = "run_final"


class EvidenceOwner(str, Enum):
    SCREEN_VERIFIER = "screen_verifier"
    TRANSITION_VERIFIER = "transition_verifier"
    LOCAL_VERIFIER = "local_verifier"
    METRIC_COLLECTOR = "metric_collector"
    UNIVERSAL_SURVEILLANCE = "universal_surveillance"
    RUN_AGGREGATOR = "run_aggregator"


class ObligationActivation(str, Enum):
    DURING_INSTRUCTION = "during_instruction"
    AFTER_INSTRUCTION = "after_instruction"
    AFTER_ACTION = "after_action"
    ON_TRANSITION = "on_transition"
    WHEN_SUBJECT_OBSERVED = "when_subject_observed"
    AT_RUN_END = "at_run_end"


class ObligationSettlement(str, Enum):
    EXISTS = "exists"
    FOR_ALL = "for_all"
    PROHIBITED = "prohibited"
    CONDITIONAL = "conditional"
    EXACTLY_ONCE = "exactly_once"


class ObligationEvent(str, Enum):
    OBSERVATION = "observation"
    TRANSITION = "transition"
    MEASUREMENT = "measurement"
    RUN_END = "run_end"


class RouteDecision(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    CONDITION_ABSENT = "condition_absent"
    UNKNOWN = "unknown"
    CLOSED = "closed"


class ObligationState(str, Enum):
    INACTIVE = "inactive"
    ACTIVE = "active"
    SATISFIED = "satisfied"
    FAILED = "failed"
    UNVERIFIED = "unverified"


class ObligationSpec(BaseModel):
    """Activation and settlement policy attached to one expected outcome."""

    activation: ObligationActivation = ObligationActivation.AT_RUN_END
    settlement: ObligationSettlement = ObligationSettlement.EXISTS
    required: bool = True
    instruction_index: int | None = Field(default=None, ge=1)
    subject: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _validate_activation_subject(self) -> "ObligationSpec":
        if self.activation in (
            ObligationActivation.DURING_INSTRUCTION,
            ObligationActivation.AFTER_INSTRUCTION,
        ) and self.instruction_index is None:
            raise ValueError("An instruction activation must name instruction_index.")
        if (
            self.activation is ObligationActivation.WHEN_SUBJECT_OBSERVED
            and self.subject is None
        ):
            raise ValueError("A subject activation must name its subject.")
        return self


@dataclass(frozen=True)
class ObligationContract:
    obligation_id: str
    outcome_index: int
    kind: ObligationKind
    evidence_owner: EvidenceOwner
    activation: ObligationActivation
    settlement: ObligationSettlement
    required: bool
    instruction_index: int | None = None
    subject: str | None = None


@dataclass(frozen=True)
class ObligationRouteReceipt:
    obligation_id: str
    event: ObligationEvent
    decision: RouteDecision
    evidence_reference: str | None = None
    reason: str = ""


@dataclass(frozen=True)
class ObligationSettlementReceipt:
    obligation_id: str
    state: ObligationState
    evidence_references: tuple[str, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class ObligationReadiness:
    total: int
    active: int
    inactive: int
    unknown: int
    closed: int

    @property
    def strict_ready(self) -> bool:
        return self.total > 0 and self.unknown == 0


class ObligationRunEvidence(BaseModel):
    """All routing and settlement evidence produced for one scenario run."""

    routes: list[ObligationRouteReceipt] = Field(default_factory=list)
    settlements: list[ObligationSettlementReceipt] = Field(default_factory=list)
    readiness: ObligationReadiness | None = None


class OutcomeContract(Protocol):
    @property
    def scope(self) -> "ScopeValue": ...

    @property
    def obligation(self) -> ObligationSpec: ...


class ScenarioContract(Protocol):
    @property
    def scenario_uuid(self) -> str: ...

    @property
    def expected_outcomes(self) -> Sequence[OutcomeContract]: ...

    def outcome_uuid(self, index: int) -> str: ...


class ScopeValue(Protocol):
    value: str


class ObligationController:
    """Own activation and settlement without reading planner narration."""

    def __init__(self, contracts: Sequence[ObligationContract]):
        self._contracts = tuple(contracts)
        self._states = {
            contract.obligation_id: ObligationState.INACTIVE
            for contract in self._contracts
        }
        self._route_receipts: list[ObligationRouteReceipt] = []
        self._evidence_by_id: dict[str, list[str]] = {
            contract.obligation_id: [] for contract in self._contracts
        }
        self._pass_count_by_id = {
            contract.obligation_id: 0 for contract in self._contracts
        }
        self._failed_by_id = {
            contract.obligation_id: False for contract in self._contracts
        }

    @classmethod
    def from_scenario(cls, scenario: ScenarioContract) -> "ObligationController":
        contracts = [
            _build_contract(scenario.outcome_uuid(index), index, outcome)
            for index, outcome in enumerate(scenario.expected_outcomes)
        ]
        return cls(contracts)

    @property
    def contracts(self) -> tuple[ObligationContract, ...]:
        return self._contracts

    @property
    def route_receipts(self) -> tuple[ObligationRouteReceipt, ...]:
        return tuple(self._route_receipts)

    def route(
        self,
        event: ObligationEvent,
        *,
        evidence_reference: str | None = None,
        active_instruction_index: int | None = None,
        completed_instruction_indices: frozenset[int] = frozenset(),
        subject_observed: bool | None = None,
    ) -> tuple[int, ...]:
        """Return outcome indexes that this event may evaluate."""
        routed: list[int] = []
        for contract in self._contracts:
            decision, reason = self._decision(
                contract,
                event,
                active_instruction_index,
                completed_instruction_indices,
                subject_observed,
            )
            self._route_receipts.append(
                ObligationRouteReceipt(
                    obligation_id=contract.obligation_id,
                    event=event,
                    decision=decision,
                    evidence_reference=evidence_reference,
                    reason=reason,
                )
            )
            if decision in (RouteDecision.ACTIVE, RouteDecision.UNKNOWN):
                if self._states[contract.obligation_id] is ObligationState.INACTIVE:
                    self._states[contract.obligation_id] = ObligationState.ACTIVE
                routed.append(contract.outcome_index)
        return tuple(routed)

    def record_verification(
        self,
        outcome_index: int,
        status: str,
        applicable: bool,
        evidence_reference: str,
    ) -> None:
        contract = self._contract_by_index(outcome_index)
        if (
            contract is None
            or not applicable
            or self._states[contract.obligation_id] is ObligationState.INACTIVE
        ):
            return
        evidence = self._evidence_by_id[contract.obligation_id]
        evidence.append(evidence_reference)
        if status == "passed":
            self._pass_count_by_id[contract.obligation_id] += 1
            if contract.settlement is not ObligationSettlement.EXACTLY_ONCE:
                self._states[contract.obligation_id] = ObligationState.SATISFIED
        elif status == "failed":
            self._failed_by_id[contract.obligation_id] = True
            if contract.settlement in (
                ObligationSettlement.PROHIBITED,
                ObligationSettlement.FOR_ALL,
            ):
                self._states[contract.obligation_id] = ObligationState.FAILED

    def close_run(self) -> tuple[ObligationSettlementReceipt, ...]:
        receipts = []
        for contract in self._contracts:
            state = self._states[contract.obligation_id]
            if state is ObligationState.ACTIVE:
                pass_count = self._pass_count_by_id[contract.obligation_id]
                if contract.settlement is ObligationSettlement.EXACTLY_ONCE:
                    state = (
                        ObligationState.SATISFIED
                        if pass_count == 1
                        else ObligationState.FAILED
                    )
                elif self._failed_by_id[contract.obligation_id]:
                    state = ObligationState.FAILED
                else:
                    state = ObligationState.UNVERIFIED
                self._states[contract.obligation_id] = state
            receipts.append(
                ObligationSettlementReceipt(
                    obligation_id=contract.obligation_id,
                    state=state,
                    evidence_references=tuple(
                        self._evidence_by_id[contract.obligation_id]
                    ),
                    reason=_settlement_reason(state),
                )
            )
        return tuple(receipts)

    def readiness(self) -> ObligationReadiness:
        decisions = [receipt.decision for receipt in self._route_receipts]
        return ObligationReadiness(
            total=len(decisions),
            active=decisions.count(RouteDecision.ACTIVE),
            inactive=decisions.count(RouteDecision.INACTIVE),
            unknown=decisions.count(RouteDecision.UNKNOWN),
            closed=decisions.count(RouteDecision.CLOSED),
        )

    def _decision(
        self,
        contract: ObligationContract,
        event: ObligationEvent,
        active_instruction_index: int | None,
        completed_instruction_indices: frozenset[int],
        subject_observed: bool | None,
    ) -> tuple[RouteDecision, str]:
        if self._states[contract.obligation_id] in (
            ObligationState.SATISFIED,
            ObligationState.FAILED,
        ):
            return RouteDecision.CLOSED, "The obligation already settled."
        activation = contract.activation
        if activation is ObligationActivation.AT_RUN_END:
            active = event is ObligationEvent.RUN_END
        elif activation is ObligationActivation.ON_TRANSITION:
            active = event is ObligationEvent.TRANSITION
        elif activation is ObligationActivation.DURING_INSTRUCTION:
            if active_instruction_index is None:
                return RouteDecision.UNKNOWN, "Instruction state was unavailable."
            active = active_instruction_index == contract.instruction_index
        elif activation is ObligationActivation.AFTER_INSTRUCTION:
            if contract.instruction_index is None:
                return RouteDecision.UNKNOWN, "The obligation names no instruction."
            active = contract.instruction_index in completed_instruction_indices
        elif activation is ObligationActivation.WHEN_SUBJECT_OBSERVED:
            if subject_observed is None:
                return RouteDecision.UNKNOWN, "Subject presence was not established."
            if not subject_observed:
                return RouteDecision.CONDITION_ABSENT, "The subject was absent."
            active = True
        else:
            return RouteDecision.UNKNOWN, "No typed action event was supplied."
        return (
            (RouteDecision.ACTIVE, "The typed activation condition matched.")
            if active
            else (RouteDecision.INACTIVE, "The activation window is not open.")
        )

    def _contract_by_index(self, index: int) -> ObligationContract | None:
        return next(
            (contract for contract in self._contracts if contract.outcome_index == index),
            None,
        )


def _build_contract(
    obligation_id: str, index: int, outcome: OutcomeContract
) -> ObligationContract:
    scope = outcome.scope.value
    if scope == "transition":
        kind = ObligationKind.TRANSITION_ASSERTION
        owner = EvidenceOwner.TRANSITION_VERIFIER
    elif scope == "measured":
        kind = ObligationKind.MEASUREMENT
        owner = EvidenceOwner.METRIC_COLLECTOR
    else:
        kind = ObligationKind.SCREEN_ASSERTION
        owner = EvidenceOwner.SCREEN_VERIFIER
    stable_id = obligation_id or derived_uuid("scenario-obligation-v1", index)
    return ObligationContract(
        obligation_id=stable_id,
        outcome_index=index,
        kind=kind,
        evidence_owner=owner,
        activation=outcome.obligation.activation,
        settlement=outcome.obligation.settlement,
        required=outcome.obligation.required,
        instruction_index=outcome.obligation.instruction_index,
        subject=outcome.obligation.subject,
    )


def select_outcomes(outcomes: Sequence, indexes: Sequence[int]) -> list:
    """Keep original positions so rule identifiers cannot be renumbered."""
    selected = set(indexes)
    return [outcome if index in selected else None for index, outcome in enumerate(outcomes)]


def aggregate_readiness(
    measurements: Sequence[ObligationReadiness],
) -> ObligationReadiness:
    """Combine run-local readiness without converting missing cases to success."""
    return ObligationReadiness(
        total=sum(item.total for item in measurements),
        active=sum(item.active for item in measurements),
        inactive=sum(item.inactive for item in measurements),
        unknown=sum(item.unknown for item in measurements),
        closed=sum(item.closed for item in measurements),
    )


def _settlement_reason(state: ObligationState) -> str:
    if state is ObligationState.SATISFIED:
        return "Accepted evidence satisfied the obligation."
    if state is ObligationState.FAILED:
        return "Accepted evidence violated the obligation."
    if state is ObligationState.UNVERIFIED:
        return "The activation window closed without sufficient evidence."
    return "The obligation never became active."
