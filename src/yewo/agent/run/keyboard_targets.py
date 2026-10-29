"""The non-device planner action that reveals visual keyboard key targets."""

import logging

from yewo.agent.memory.history import ActionEntry, ActionResult
from yewo.agent.run import outcomes
from yewo.agent.step import narration, targeting

logger = logging.getLogger(__name__)


def expand(step, action, snapshot, session_trail, ctx, grounding):
    """Expand model-declared keys without reporting a miss or touching the device."""
    keyboard_observation = getattr(snapshot, "keyboard_observation", None)
    available = (
        keyboard_observation is not None
        and keyboard_observation.visual_visible
        and bool(keyboard_observation.key_elements)
    )
    decision = targeting.Decision(
        ok=available,
        observation=(
            None if available else "No model-declared visual keyboard keys are available to expand."
        ),
    )
    ctx.steps.record_grounding(grounding, decision, action)
    description = narration.describe(action)
    if not available:
        targeting.record_unexecuted(ctx.history, step, description, decision.observation or "")
        ctx.steps.record_blocked_action(action, description, decision.observation)
        ctx.stall.record_blocked(snapshot.current_phash)
        if ctx.stall.is_exhausted:
            return outcomes.loop_detected(
                ctx,
                session_trail,
                snapshot.current_phash,
                targeting.stalled_reason(decision),
            )
        return None

    ctx.keyboard_targets_expanded = True
    ctx.pending_loop_nudge = (
        "Visual keyboard targets are now expanded. Choose the numbered "
        "[key] target required by the keyboard UI test."
    )
    result = ActionResult(
        success=True,
        observation="Visual keyboard targets expanded without touching the device.",
    )
    ctx.history.append(ActionEntry(step_num=step + 1, action_desc=description, result=result))
    ctx.steps.record_action(action, description, result, None)
    ctx.steps.record_action_duration(0.0)
    ctx.stall.action_dispatched()
    ctx.last_action_type = action.type
    logger.info("Expanded visual keyboard targets for the next look.")
    return None
