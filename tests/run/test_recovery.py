"""Recovery is bounded by policy and settled only by original evidence."""

from types import SimpleNamespace

import pytest

from yewo.agent.actuation.actions import Coordinates, TypeAction
from yewo.agent.device.focus import FocusedFieldReading
from yewo.agent.memory.replay_store import ValueNames
from yewo.agent.run.postconditions import pending_for, verify_context
from yewo.agent.run.record import StepLog
from yewo.agent.run.outcomes import goal_reached
from yewo.agent.plan.schema import DeviceStepResponse, DoneAction
from yewo.agent.run.recovery import (
    RecoveryCause,
    RecoveryMode,
    RecoveryStatus,
    RecoveryTransaction,
    recovery_directive,
    recovery_mode_for,
    open_semantic_cycle_recovery,
    fail_active_recovery,
)
from yewo.agent.run.instruction_controller import (
    InstructionCompletionReceipt,
    InstructionController,
)


def transaction() -> RecoveryTransaction:
    return RecoveryTransaction(
        failed_step=1,
        opened_step=2,
        original_intent="Card number field",
        value_name="card_number",
        failure_reason="The entry action did not deliver the declared value.",
        cause=RecoveryCause.GROUNDING_DRIFT,
        mode=RecoveryMode.MODEL_PROPOSAL,
    )


def test_a_model_gets_only_one_proposal_for_an_equivalent_state():
    recovery = transaction()

    assert recovery.may_propose("screen-a:keyboard=open:focus=none")
    recovery.record_proposal("screen-a:keyboard=open:focus=none")

    assert not recovery.may_propose("screen-a:keyboard=open:focus=none")
    assert recovery.may_propose("screen-a:keyboard=closed:focus=none")


def test_a_recovered_grounding_only_proposes_a_human_reviewed_change():
    recovery = transaction()

    recovery.recover("The original field postcondition passed.")

    assert recovery.status is RecoveryStatus.RECOVERED
    assert recovery.suggestion is not None
    assert recovery.suggestion.human_review_required
    assert not recovery.suggestion.applied


def test_a_transaction_record_never_contains_the_declared_value():
    recovery = transaction()

    assert "4111111111111111" not in recovery.model_dump_json()


def test_policy_selects_authority_from_the_failure_class():
    assert recovery_mode_for(RecoveryCause.MECHANICAL) is RecoveryMode.DETERMINISTIC
    assert recovery_mode_for(RecoveryCause.GROUNDING_DRIFT) is RecoveryMode.MODEL_PROPOSAL
    assert recovery_mode_for(RecoveryCause.SEMANTIC_OBSTRUCTION) is RecoveryMode.MODEL_PROPOSAL
    assert recovery_mode_for(RecoveryCause.NON_RECOVERABLE) is RecoveryMode.STOP


@pytest.mark.parametrize("cause", list(RecoveryCause))
@pytest.mark.parametrize("terminal", ["recover", "resume", "fail"])
def test_every_recovery_terminal_is_immutable(cause, terminal):
    recovery = RecoveryTransaction.open(
        failed_step=1,
        opened_step=2,
        original_intent="Complete the original intent.",
        failure_reason="It was not complete.",
        cause=cause,
    )
    getattr(recovery, terminal)("terminal evidence")

    assert recovery.is_terminal
    with pytest.raises(ValueError, match="immutable"):
        recovery.fail("replacement evidence")
    with pytest.raises(ValueError, match="closed"):
        recovery.record_action(
            {
                "step": 2,
                "state_key": "screen",
                "action_type": "tap",
                "description": "bridge",
                "success": True,
            }
        )


def test_interruption_closes_the_active_transaction_with_its_evidence():
    recovery = transaction()
    ctx = SimpleNamespace(active_recovery=recovery)

    fail_active_recovery(ctx, "Execution was interrupted.")

    assert ctx.active_recovery is None
    assert recovery.status is RecoveryStatus.FAILED
    assert recovery.final_evidence == "Execution was interrupted."


def test_recovery_intent_is_language_agnostic():
    recovery = RecoveryTransaction.open(
        failed_step=1,
        opened_step=2,
        original_intent="أدخل الاسم ثم 選択してください",
        failure_reason="The original action did not satisfy its postcondition.",
        cause=RecoveryCause.SEMANTIC_OBSTRUCTION,
    )

    directive = recovery_directive(recovery)

    assert "أدخل الاسم" in directive
    assert "選択してください" in directive


def test_a_recovery_model_cannot_declare_its_own_success():
    recovery = transaction()
    ctx = SimpleNamespace(
        active_recovery=recovery,
        postcondition_blocked=False,
        postcondition_verifications=[
            SimpleNamespace(reason="The original postcondition did not pass.")
        ],
        steps_executed=2,
        intervention_steps=0,
        history=SimpleNamespace(),
        capability_receipts=[],
        steps=SimpleNamespace(records=[]),
        recovery_transactions=[recovery],
    )
    response = DeviceStepResponse(
        current_screen="Form",
        evaluation_previous_goal="Not complete",
        observation="The form is visible.",
        reasoning="Stop.",
        goal_satisfied=True,
        goal_evidence="Looks complete.",
        action=DoneAction(reason="Looks complete."),
    )

    result = goal_reached(True, response, None, "screen", ctx)

    assert result is not None
    assert result.status == "postcondition_stop"
    assert recovery.status is RecoveryStatus.FAILED


def test_a_semantic_cycle_opens_a_bounded_model_recovery():
    steps = StepLog()
    steps.open(8)
    ctx = SimpleNamespace(
        active_recovery=None,
        active_instruction="Add one exercise using controls the app presents.",
        recovery_transactions=[],
        steps=steps,
    )

    open_semantic_cycle_recovery(ctx, "The route returned to its earlier state.")

    assert ctx.active_recovery is not None
    assert ctx.active_recovery.cause is RecoveryCause.SEMANTIC_OBSTRUCTION
    assert ctx.active_recovery.mode is RecoveryMode.MODEL_PROPOSAL


def test_a_planner_instruction_claim_cannot_resume_cycle_recovery():
    recovery = RecoveryTransaction.open(
        failed_step=7,
        opened_step=8,
        original_intent="Add one exercise.",
        failure_reason="The route cycled.",
        cause=RecoveryCause.SEMANTIC_OBSTRUCTION,
    )
    controller = InstructionController.from_goal(
        ["Launch", "Create", "Name", "Add exercise", "Enter settings"]
    )
    for index in range(1, 4):
        active = controller.active_instruction
        assert active is not None
        controller.accept_completion(
            InstructionCompletionReceipt(
                instruction_id=active.instruction_id,
                observation_id=f"observation-{index}",
                evidence_owner="test-obligation-verifier",
            )
        )

    claim = controller.record_planner_claim(5, False)

    assert not claim.accepted
    assert controller.active_index == 4
    assert controller.active_instruction is not None
    assert controller.active_instruction.intent == "Add exercise"
    assert recovery.status is RecoveryStatus.OPEN


def test_failed_delivery_opens_a_transaction_and_passed_retry_closes_it():
    values = {"card_number": "4111111111111111"}
    action = TypeAction(
        text=values["card_number"],
        coordinates=Coordinates(x=100, y=200),
        target="Card number field",
        reason="Enter the declared card number.",
    )
    pending = pending_for(action, ValueNames(values), values, "4111")
    assert pending is not None
    steps = StepLog()
    steps.open(2)
    ctx = SimpleNamespace(
        pending_entered_value=pending,
        postcondition_verifications=[],
        input_transactions=[],
        postcondition_blocked=False,
        postcondition_recovery_required=False,
        steps=steps,
        active_recovery=None,
        recovery_transactions=[],
    )
    snapshot = SimpleNamespace(gui_elements=[])

    verify_context(
        snapshot,
        ctx,
        read_focused_field=lambda: FocusedFieldReading("4111", (10, 10, 200, 240)),
    )

    assert ctx.active_recovery is not None
    assert len(ctx.recovery_transactions) == 1
    assert values["card_number"] not in ctx.active_recovery.model_dump_json()

    retry = pending_for(action, ValueNames(values), values, values["card_number"])
    assert retry is not None
    ctx.pending_entered_value = retry
    verify_context(
        snapshot,
        ctx,
        read_focused_field=lambda: FocusedFieldReading(
            values["card_number"], (10, 10, 200, 240)
        ),
    )

    assert ctx.active_recovery is None
    assert ctx.recovery_transactions[0].status is RecoveryStatus.RECOVERED
