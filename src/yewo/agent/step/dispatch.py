"""Sending one action to the device, and saying what it did.

Separated from the loop. It is the only place that touches the device, and
the only place that names what the attempt did afterwards. One attempt fills
four readers: the step record, the history the planner reads next, the
trail, and the replay ledger. They have to agree, so they are filled here
rather than at four call sites.

An action that raises is still an action that happened. The error is kept
and the screen is settled anyway. The stall policy is told as well: a step
that reached nothing stalls a screen as surely as one refused before
dispatch.
"""

import logging
from typing import TYPE_CHECKING, Optional

from yewo.agent.actuation.actions import TypeAction
from yewo.agent.actuation.safety import (
    ActionSafetyPolicy,
    ActionSource,
    execute_authorized,
)
from yewo.agent.device.receipts import Receipt
from yewo.agent.grounding import marks
from yewo.agent.memory.history import ActionEntry
from yewo.agent.memory.replay import ReplayLedger, target_label
from yewo.agent.memory.trail import SessionTrail
from yewo.agent.perception import settle
from yewo.agent.plan.schema import DeviceStepResponse
from yewo.agent.step import narration, observation
from yewo.agent.step.observation import PerceptionSnapshot
from yewo.agent.run.postconditions import TextEntryReceipt, pending_for
from yewo.errors import FocusVerificationFailed, TargetNotInteractive
from yewo.utils.secrets import SecretScrubber

if TYPE_CHECKING:
    from yewo.agent.run.context import RunContext

logger = logging.getLogger(__name__)
ACTION_SAFETY_POLICY = ActionSafetyPolicy()


def execute(
    step: int,
    action,
    step_response: DeviceStepResponse,
    device_control,
    snapshot: PerceptionSnapshot,
    session_trail: Optional[SessionTrail],
    ctx: "RunContext",
    scrubber: SecretScrubber,
    replay_ledger: Optional[ReplayLedger],
    settle_time: float,
    requested_action=None,
) -> bool:
    """Touches the device once, and writes down what happened.

    Returns:
        Whether the app navigated somewhere outside itself, which the next
        step reads to decide whether drift is expected.
    """
    error: Optional[str] = None
    last_external_nav = ctx.last_external_nav
    try:
        source = (
            ActionSource.RECOVERY
            if ctx.active_recovery is not None
            else ActionSource.PLANNER
        )
        authorization = ACTION_SAFETY_POLICY.authorize(
            action,
            source=source,
            observation_id=snapshot.observation_id,
            screen_phash=snapshot.current_phash,
            screen_size=observation.screen_size(snapshot),
            context_id=(
                ctx.execution_context.context_id
                if ctx.execution_context is not None
                else None
            ),
            instruction_controller=ctx.instruction_controller,
            keyboard_visible=ctx.keyboard_open,
            expected_destination_phash=ctx.loop_detector.expected_back_destination(
                snapshot.current_phash
            ),
        )
        ctx.steps.record_authorization(authorization)
        receipt = execute_authorized(authorization, device_control, scrubber)
        if isinstance(receipt, Receipt):
            ctx.capability_receipts.append(receipt)
        if isinstance(action, TypeAction) and isinstance(receipt, TextEntryReceipt):
            ctx.pending_entered_value = pending_for(
                action,
                ctx.value_names,
                ctx.declared_values,
                receipt.observed,
                verification_policies=ctx.value_verifications,
                focus=receipt.focus,
                source_observation_id=authorization.observation_id,
                source_context_id=authorization.context_id,
                source_phash=snapshot.current_phash,
            )
        if isinstance(action, TypeAction):
            ctx.loop_detector.record_verified_text_entry()
        last_external_nav = step_response.external_navigation
        ctx.stall.action_dispatched()
        ctx.loop_detector.record_success(
            action,
            snapshot.current_phash,
            observation.screen_size(snapshot),
        )
    except FocusVerificationFailed as exc:
        error = str(exc)
        ctx.loop_detector.record_focus_verification_failure()
        logger.error("Action execution failed: %s", error)
    except TargetNotInteractive as exc:
        error = str(exc)
        marks.refuse_mark(
            ctx.text_strikes,
            ctx.refused_marks,
            snapshot.current_phash,
            step_response.action.model_dump().get("mark"),
            snapshot.marks,
        )
        logger.error("Action execution failed: %s", error)
    except Exception as exc:  # noqa: BLE001 - the device boundary
        error = str(exc)
        logger.error("Action execution failed: %s", error)

    if error is not None:
        ctx.stall.record_blocked(snapshot.current_phash)

    post_phash = settle.settle(device_control, settle_time)
    ctx.steps.record_settlement_id(ctx.loop_detector.active_settlement_id)
    outcome_advisory = ctx.loop_detector.record_delivery_outcome(
        action,
        snapshot.current_phash,
        post_phash,
        observation.screen_size(snapshot),
        failed=error is not None,
    )
    if outcome_advisory is not None:
        ctx.pending_loop_nudge = "\n".join(
            part for part in (ctx.pending_loop_nudge, outcome_advisory) if part
        )
    action_result = narration.action_result(
        action, snapshot.current_phash, post_phash, error
    )
    action_desc = narration.describe(action)
    planned_action = requested_action or action
    planned_desc = narration.describe(planned_action)
    label = target_label(action, snapshot.gui_elements)

    ctx.steps.record_action(action, action_desc, action_result, label)

    if replay_ledger is not None and error is None:
        replay_ledger.record(
            snapshot.current_phash, action, post_phash, action_desc, label
        )
    ctx.history.append(
        ActionEntry(step_num=step + 1, action_desc=planned_desc, result=action_result)
    )
    if error is None:
        advisory = ctx.loop_detector.record_unchanged_tap(
            action,
            snapshot.current_phash,
            post_phash,
            observation.screen_size(snapshot),
        )
        if advisory is not None:
            ctx.pending_loop_nudge = "\n".join(
                part for part in (ctx.pending_loop_nudge, advisory) if part
            )

    ctx.last_action_type = action.type
    return last_external_nav
