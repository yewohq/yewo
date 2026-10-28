"""State updates made from one device observation or planner description."""

import logging
import time

from yewo.agent.device import state
from yewo.agent.perception import keyboard
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
    native_keyboard = device_control.is_keyboard_open()
    keyboard_observation = keyboard.observe(snapshot.gui_elements, native_keyboard)
    snapshot.keyboard_observation = keyboard_observation
    ctx.keyboard_open = keyboard_observation.effective_visible
    if not keyboard_observation.visual_visible:
        ctx.keyboard_targets_expanded = False
    if snapshot.scan_info is not None:
        snapshot.scan_info.keyboard_native_visible = native_keyboard
        snapshot.scan_info.keyboard_visual_visible = keyboard_observation.visual_visible
        snapshot.scan_info.keyboard_effective_visible = keyboard_observation.effective_visible
        snapshot.scan_info.keyboard_owned_count = len(keyboard_observation.owned_elements)
        snapshot.scan_info.keyboard_state_disagrees = keyboard_observation.state_disagrees
        snapshot.scan_info.keyboard_targets_expanded = ctx.keyboard_targets_expanded
    agent.images.clear_after_navigation(snapshot.scan_info)
    ctx.current_phash = snapshot.current_phash
    ctx.steps.files(agent.artifacts.save_step(session_dir, step, snapshot))

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
