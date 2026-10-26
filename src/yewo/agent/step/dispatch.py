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

from yewo.agent.actuation.orchestrator import ActionOrchestrator
from yewo.agent.actuation.actions import ActionType, TypeAction
from yewo.agent.device.receipts import Receipt
from yewo.agent.grounding import marks
from yewo.agent.memory.history import ActionEntry
from yewo.agent.memory.replay import ReplayLedger, target_label
from yewo.agent.memory.trail import SessionTrail
from yewo.agent.perception import settle
from yewo.agent.plan.schema import DeviceStepResponse
from yewo.agent.step import narration
from yewo.agent.step.observation import PerceptionSnapshot
from yewo.agent.run.postconditions import TextEntryReceipt, pending_for
from yewo.errors import TargetNotInteractive
from yewo.utils.secrets import SecretScrubber

if TYPE_CHECKING:
    from yewo.agent.run.context import RunContext

logger = logging.getLogger(__name__)


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
        receipt = ActionOrchestrator(device_control, scrubber).execute(action)
        if isinstance(receipt, Receipt):
            ctx.capability_receipts.append(receipt)
        if isinstance(action, TypeAction) and isinstance(receipt, TextEntryReceipt):
            ctx.pending_entered_value = pending_for(
                action,
                ctx.value_names,
                ctx.declared_values,
                receipt.observed,
                verification_policies=ctx.value_verifications,
            )
        last_external_nav = step_response.external_navigation
        ctx.stall.action_dispatched()
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
    if session_trail:
        session_trail.record_action(planned_desc)

    _record_unchanged_tap(ctx, action, snapshot.current_phash, post_phash, error)

    ctx.last_action_type = action.type
    return last_external_nav


UNCHANGED_TAP_NUDGE = (
    "This visual target received {attempts} delivered taps on the same unchanged "
    "screen. No visible effect was observed. Hidden app state may have changed. "
    "Do not claim that the taps failed. Continue to the next logical action. "
    "Repeat this target only when the goal explicitly requires repeated taps."
)


def _record_unchanged_tap(ctx, action, pre_phash, post_phash, error) -> None:
    """Add evidence after two taps without forbidding a valid repeated action."""
    if (
        error is not None
        or action.type not in (ActionType.TAP, ActionType.DOUBLE_TAP)
        or not pre_phash
        or pre_phash != post_phash
    ):
        return

    point = action.coordinates
    key = (pre_phash, action.type.value, point.x // 40, point.y // 40)
    attempts = ctx.unchanged_tap_attempts.get(key, 0) + 1
    ctx.unchanged_tap_attempts[key] = attempts
    if attempts < 2:
        return

    advisory = UNCHANGED_TAP_NUDGE.format(attempts=attempts)
    ctx.pending_loop_nudge = "\n".join(
        part for part in (ctx.pending_loop_nudge, advisory) if part
    )
