"""Instruction progress belongs to deterministic evidence, not planner claims."""

import pytest

from yewo.agent.run.instruction_controller import (
    InstructionCompletionReceipt,
    InstructionController,
    InstructionState,
)
from yewo.agent.run.evidence_identity import (
    ExecutionContextIdentity,
    new_observation_id,
)
from yewo.agent.run.record import StepLog


def _receipt(controller: InstructionController, observation_id: str):
    active = controller.active_instruction
    assert active is not None
    return InstructionCompletionReceipt(
        instruction_id=active.instruction_id,
        observation_id=observation_id,
        evidence_owner="test-obligation-verifier",
    )


def test_instruction_identity_survives_an_unrelated_insertion() -> None:
    original = InstructionController.from_goal(["Open editor", "Save schedule"])
    expanded = InstructionController.from_goal(
        ["Launch application", "Open editor", "Save schedule"]
    )

    original_ids = {
        instruction.intent: instruction.instruction_id
        for instruction in original.instructions
    }
    expanded_ids = {
        instruction.intent: instruction.instruction_id
        for instruction in expanded.instructions
    }

    assert original_ids["Open editor"] == expanded_ids["Open editor"]
    assert original_ids["Save schedule"] == expanded_ids["Save schedule"]


def test_duplicate_intents_receive_distinct_stable_identities() -> None:
    controller = InstructionController.from_goal(["Save", "Save"])

    assert controller.instructions[0].instruction_id != controller.instructions[1].instruction_id


def test_planner_claim_is_diagnostic_and_cannot_advance_progress() -> None:
    controller = InstructionController.from_goal(["Open editor", "Save schedule"])

    claim = controller.record_planner_claim(2, True)

    assert not claim.accepted
    assert claim.authoritative_index == 1
    assert controller.planner_claims == (claim,)
    assert controller.active_index == 1
    assert controller.state_of(controller.instructions[0].instruction_id) is InstructionState.ACTIVE

    steps = StepLog()
    steps.open(1)
    steps.record_instruction_claim(controller, claim)
    recorded = steps.records[0].instruction_authority
    assert recorded is not None
    assert recorded.authoritative_index == 1
    assert recorded.planner_claimed_index == 2
    assert not recorded.planner_claim_accepted


def test_completion_receipt_advances_and_terminal_state_is_immutable() -> None:
    controller = InstructionController.from_goal(["Open editor", "Save schedule"])
    first_id = controller.instructions[0].instruction_id

    controller.accept_completion(_receipt(controller, "observation-1"))

    assert controller.state_of(first_id) is InstructionState.SATISFIED
    assert controller.active_index == 2
    with pytest.raises(ValueError, match="does not belong"):
        controller.accept_completion(
            InstructionCompletionReceipt(
                instruction_id=first_id,
                observation_id="observation-2",
                evidence_owner="test-obligation-verifier",
            )
        )
    with pytest.raises(ValueError, match="immutable"):
        controller.transition_terminal(first_id, InstructionState.FAILED)

    controller.accept_completion(_receipt(controller, "observation-3"))
    assert controller.active_instruction is None


def test_each_attempt_has_identity_under_the_active_instruction() -> None:
    controller = InstructionController.from_goal("Open editor")

    first = controller.begin_attempt("observation-1")
    second = controller.begin_attempt("observation-2")

    assert first is not None
    assert second is not None
    assert first.attempt_id != second.attempt_id
    assert first.instruction_id == second.instruction_id
    assert first.observation_id == "observation-1"
    assert controller.attempts == (first, second)


def test_controller_rejects_invalid_transitions_and_incomplete_receipts() -> None:
    controller = InstructionController.from_goal(["Open editor", "Save schedule"])
    active = controller.active_instruction
    assert active is not None

    with pytest.raises(ValueError, match="not terminal"):
        controller.transition_terminal(active.instruction_id, InstructionState.ACTIVE)
    with pytest.raises(ValueError, match="must name"):
        controller.accept_completion(
            InstructionCompletionReceipt(
                instruction_id=active.instruction_id,
                observation_id="",
                evidence_owner="",
            )
        )
    with pytest.raises(ValueError, match="Only the active"):
        controller.transition_terminal(
            controller.instructions[1].instruction_id,
            InstructionState.SKIPPED,
        )


def test_empty_contract_has_no_active_instruction_or_attempt() -> None:
    controller = InstructionController.from_goal([])

    assert controller.active_instruction is None
    assert controller.active_index is None
    assert controller.begin_attempt("observation") is None


def test_observations_are_unique_and_context_identity_is_reproducible() -> None:
    instruction_ids = ("instruction-a", "instruction-b")

    first = ExecutionContextIdentity.build(
        app_id="app-build-1",
        platform="android",
        actor="member",
        setup="clear_data",
        instruction_ids=instruction_ids,
    )
    second = ExecutionContextIdentity.build(
        app_id="app-build-1",
        platform="android",
        actor="member",
        setup="clear_data",
        instruction_ids=instruction_ids,
    )

    assert new_observation_id() != new_observation_id()
    assert first.context_id == second.context_id
    assert first.contract_version == second.contract_version
