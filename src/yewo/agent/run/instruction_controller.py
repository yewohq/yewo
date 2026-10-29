"""Deterministic ownership of instruction identity and progress."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from yewo.identity import derived_uuid, new_uuid


class InstructionState(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    SATISFIED = "satisfied"
    FAILED = "failed"
    BLOCKED = "blocked"
    INTERRUPTED = "interrupted"
    SKIPPED = "skipped"


TERMINAL_INSTRUCTION_STATES = frozenset(
    {
        InstructionState.SATISFIED,
        InstructionState.FAILED,
        InstructionState.BLOCKED,
        InstructionState.INTERRUPTED,
        InstructionState.SKIPPED,
    }
)


@dataclass(frozen=True)
class InstructionContract:
    """One instruction whose identity does not depend on its list position."""

    instruction_id: str
    intent: str


@dataclass(frozen=True)
class InstructionAttempt:
    """One bounded attempt under an instruction."""

    attempt_id: str
    instruction_id: str
    observation_id: str


@dataclass(frozen=True)
class PlannerProgressClaim:
    """Untrusted planner narration retained only for diagnostics."""

    claimed_index: int | None
    claimed_goal_satisfied: bool
    authoritative_index: int | None
    accepted: bool = False


@dataclass(frozen=True)
class InstructionCompletionReceipt:
    """Evidence accepted by the controller to settle one instruction."""

    instruction_id: str
    observation_id: str
    evidence_owner: str
    receipt_id: str = field(default_factory=new_uuid)


class InstructionController:
    """Pure state machine for instruction progress.

    Shadow mode records attempts and rejected planner claims. Only an explicit
    completion receipt can advance state, which later obligation routing will
    produce.
    """

    def __init__(
        self,
        instructions: Iterable[InstructionContract],
        *,
        shadow_mode: bool = True,
    ) -> None:
        self._instructions = tuple(instructions)
        self._states = {
            instruction.instruction_id: InstructionState.PENDING
            for instruction in self._instructions
        }
        self._attempts: list[InstructionAttempt] = []
        self._planner_claims: list[PlannerProgressClaim] = []
        self.shadow_mode = shadow_mode
        if self._instructions:
            self._states[self._instructions[0].instruction_id] = InstructionState.ACTIVE

    @classmethod
    def from_goal(
        cls, goal: str | list[str], *, shadow_mode: bool = True
    ) -> "InstructionController":
        intents = [goal] if isinstance(goal, str) else list(goal)
        occurrences: dict[str, int] = {}
        contracts = []
        for intent in intents:
            occurrence = occurrences.get(intent, 0) + 1
            occurrences[intent] = occurrence
            contracts.append(
                InstructionContract(
                    instruction_id=derived_uuid(
                        "instruction-contract-v1", intent, occurrence
                    ),
                    intent=intent,
                )
            )
        return cls(contracts, shadow_mode=shadow_mode)

    @property
    def instructions(self) -> tuple[InstructionContract, ...]:
        return self._instructions

    @property
    def attempts(self) -> tuple[InstructionAttempt, ...]:
        return tuple(self._attempts)

    @property
    def planner_claims(self) -> tuple[PlannerProgressClaim, ...]:
        return tuple(self._planner_claims)

    @property
    def active_index(self) -> int | None:
        for index, instruction in enumerate(self._instructions, start=1):
            if self._states[instruction.instruction_id] is InstructionState.ACTIVE:
                return index
        return None

    @property
    def active_instruction(self) -> InstructionContract | None:
        index = self.active_index
        return self._instructions[index - 1] if index is not None else None

    def state_of(self, instruction_id: str) -> InstructionState:
        return self._states[instruction_id]

    def begin_attempt(self, observation_id: str) -> InstructionAttempt | None:
        instruction = self.active_instruction
        if instruction is None:
            return None
        attempt = InstructionAttempt(
            attempt_id=new_uuid(),
            instruction_id=instruction.instruction_id,
            observation_id=observation_id,
        )
        self._attempts.append(attempt)
        return attempt

    def record_planner_claim(
        self, claimed_index: int | None, claimed_goal_satisfied: bool
    ) -> PlannerProgressClaim:
        claim = PlannerProgressClaim(
            claimed_index=claimed_index,
            claimed_goal_satisfied=claimed_goal_satisfied,
            authoritative_index=self.active_index,
        )
        self._planner_claims.append(claim)
        return claim

    def accept_completion(self, receipt: InstructionCompletionReceipt) -> None:
        """Advance only from a receipt for the currently active instruction."""
        instruction = self.active_instruction
        if instruction is None or receipt.instruction_id != instruction.instruction_id:
            raise ValueError("Completion evidence does not belong to the active instruction.")
        if not receipt.observation_id or not receipt.evidence_owner:
            raise ValueError("Completion evidence must name its observation and owner.")
        self._transition(instruction.instruction_id, InstructionState.SATISFIED)
        for candidate in self._instructions:
            if self._states[candidate.instruction_id] is InstructionState.PENDING:
                self._states[candidate.instruction_id] = InstructionState.ACTIVE
                return

    def transition_terminal(
        self, instruction_id: str, state: InstructionState
    ) -> None:
        if state not in TERMINAL_INSTRUCTION_STATES:
            raise ValueError("The requested instruction state is not terminal.")
        self._transition(instruction_id, state)

    def _transition(self, instruction_id: str, state: InstructionState) -> None:
        current = self._states[instruction_id]
        if current in TERMINAL_INSTRUCTION_STATES:
            raise ValueError("A terminal instruction state is immutable.")
        if current is not InstructionState.ACTIVE:
            raise ValueError("Only the active instruction can transition.")
        self._states[instruction_id] = state
