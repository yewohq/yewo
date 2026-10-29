"""Typed boundaries around model choice and live discovery mechanics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import Field

from yewo.agent.actuation.actions import Action, PlannerTapAction
from yewo.agent.actuation.safety import ActionAuthorization
from yewo.agent.run.settlement import SettlementReceipt
from yewo.discovery.artifacts import DiscoveryContract
from yewo.discovery.contracts import (
    DiscoveryObservation,
    DiscoveryRun,
    SideEffectClass,
)
from yewo.identity import derived_uuid, new_uuid


class DiscoveryFinding(DiscoveryContract):
    """Universal-policy observation emitted outside scenario verdicts."""

    finding_id: str = Field(default_factory=new_uuid)
    kind: str
    severity: str
    observation_id: str
    description: str


class CapabilityEvidenceReceipt(DiscoveryContract):
    """Local predicate receipt; model proposals cannot construct this contract."""

    receipt_id: str = Field(default_factory=new_uuid)
    evidence_owner: Literal["local_predicate"] = "local_predicate"
    question_id: str
    observation_id: str
    kind: Literal["observed", "route_verified"]
    predicate_id: str
    transition_identity: str | None = None


@dataclass(frozen=True)
class DiscoveryActionCandidate:
    """One concrete action already grounded in an exact observation."""

    source_observation_id: str
    source_screen_state_id: str
    control_identity: str
    relevant_question_ids: tuple[str, ...]
    action: Action | PlannerTapAction
    side_effect: SideEffectClass
    reversible: bool
    source_phash: str
    screen_size: tuple[int, int]
    expected_destination_phash: str | None = None
    reset_after: bool = False

    @property
    def frontier_id(self) -> str:
        return derived_uuid(
            "discovery-frontier-v1",
            self.source_screen_state_id,
            self.control_identity,
            *sorted(self.relevant_question_ids),
        )

    @property
    def attempt_identity(self) -> str:
        return derived_uuid(
            "discovery-attempt-v1",
            self.source_screen_state_id,
            self.control_identity,
        )


@dataclass(frozen=True)
class DiscoveryObservationBatch:
    observation: DiscoveryObservation
    candidates: tuple[DiscoveryActionCandidate, ...] = ()
    capability_evidence: tuple[CapabilityEvidenceReceipt, ...] = ()
    findings: tuple[DiscoveryFinding, ...] = ()
    state_depth: int = 0


@dataclass(frozen=True)
class DiscoveryProposalReceipt:
    selected_frontier_id: str | None
    model_calls: int
    cost_micro_usd: int


@dataclass(frozen=True)
class DiscoverySetupReceipt:
    success: bool
    setup_contract_sha256: str
    reason: str = ""


@dataclass(frozen=True)
class DiscoveryDispatchReceipt:
    dispatch_id: str
    authorization_id: str
    success: bool
    reason: str = ""


@dataclass(frozen=True)
class DiscoveryExecutionReceipt:
    dispatch: DiscoveryDispatchReceipt
    settlement: SettlementReceipt
    destination: DiscoveryObservationBatch
    capability_evidence: tuple[CapabilityEvidenceReceipt, ...] = ()
    findings: tuple[DiscoveryFinding, ...] = ()


@dataclass(frozen=True)
class DiscoveryResetReceipt:
    success: bool
    observation: DiscoveryObservationBatch | None = None
    reason: str = ""


@dataclass(frozen=True)
class DiscoveryResult:
    run: DiscoveryRun
    findings: tuple[DiscoveryFinding, ...]


class DiscoveryProposer(Protocol):
    """A model or deterministic chooser that can select only offered IDs."""

    model_calls_per_proposal: int
    maximum_cost_micro_usd: int

    def select(
        self,
        run: DiscoveryRun,
        candidates: tuple[DiscoveryActionCandidate, ...],
    ) -> DiscoveryProposalReceipt: ...


class DiscoveryRuntime(Protocol):
    """Device mechanics beneath the deterministic discovery controller."""

    def prepare(self, setup_contract_sha256: str) -> DiscoverySetupReceipt: ...

    def observe(self) -> DiscoveryObservationBatch: ...

    def execute(
        self,
        authorization: ActionAuthorization,
        candidate: DiscoveryActionCandidate,
    ) -> DiscoveryExecutionReceipt: ...

    def reset(self) -> DiscoveryResetReceipt: ...
