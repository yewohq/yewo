"""State updates made from one device observation or planner description."""

import logging
import time

from yewo.agent.actuation.actions import ActionType
from yewo.agent.device import state
from yewo.agent.step import narration, observation

logger = logging.getLogger(__name__)


def elapsed_ms(started: float) -> float:
    """Return elapsed milliseconds for a step-stage record."""
    return round((time.perf_counter() - started) * 1000, 1)


def capture(agent, step, device_control, perception, session_dir, session_trail, ctx):
    """Read the screen and update the cross-step state derived from it."""
    ctx.drift_warning = state.context_drift(
        device_control, ctx.last_external_nav, agent.settle_time
    )
    snapshot = observation.capture(perception, session_trail, ctx.prev_phash)
    if snapshot.scan_info is not None:
        ctx.screen_height = snapshot.scan_info.screen_height
    agent.images.clear_after_navigation(snapshot.scan_info)
    ctx.current_phash = snapshot.current_phash
    ctx.steps.files(agent.artifacts.save_step(session_dir, step, snapshot))

    moved = snapshot.scan_info and snapshot.scan_info.reason in (
        "navigation",
        "layout_shift",
    )
    if moved or ctx.last_action_type == ActionType.TYPE.value:
        detected = device_control.is_keyboard_open()
        if detected is not None:
            ctx.keyboard_open = detected

    ctx.history.update_last_reaction(
        narration.reaction(snapshot.scan_info, ctx.prev_element_count, ctx.keyboard_open)
    )
    ctx.prev_element_count = snapshot.scan_info.element_count if snapshot.scan_info else None
    ctx.pending_loop_nudge = (
        "\n".join(
            filter(
                None,
                [
                    ctx.pending_loop_nudge,
                    ctx.loop_detector.check(None, snapshot.current_phash),
                ],
            )
        )
        or None
    )
    return snapshot


def apply_plan_result(agent, step_response, snapshot, session_trail, ctx) -> None:
    """Record the planner's safe screen description and previous-step view."""
    logger.info("Screen: %s", step_response.current_screen)
    logger.info("Reasoning: %s", agent.scrubber.scrub(step_response.reasoning))
    if step_response.evaluation_previous_goal:
        logger.info(
            "Previous step eval: %s",
            agent.scrubber.scrub(step_response.evaluation_previous_goal),
        )
    ctx.history.update_last_evaluation(step_response.evaluation_previous_goal)
    if session_trail and snapshot.current_phash:
        session_trail.record_screen(
            snapshot.current_phash,
            step_response.current_screen,
            snapshot.gui_elements,
        )
