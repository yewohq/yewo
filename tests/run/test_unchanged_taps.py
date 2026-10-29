"""Memory for taps whose visible screen does not change."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

from yewo.agent.actuation.actions import Coordinates, TapMarkAction
from yewo.agent.actuation.coordinate_actions import TapCoordinateAction
from yewo.agent.memory.history import AgentHistory
from yewo.agent.memory.loop_detector import RollingLoopDetector
from yewo.agent.plan.schema import MarksStepResponse
from yewo.agent.run.context import RunContext
from yewo.agent.run.record import StepLog
from yewo.agent.step import dispatch
from yewo.utils.secrets import SecretScrubber


def _context() -> tuple[RunContext, SecretScrubber]:
    scrubber = SecretScrubber()
    return (
        RunContext(
            history=AgentHistory(),
            loop_detector=RollingLoopDetector(),
            steps=StepLog(scrubber.scrub),
        ),
        scrubber,
    )


def _tap(
    ctx: RunContext,
    scrubber: SecretScrubber,
    *,
    step: int,
    post_phash: str = "same-screen",
    failure: Exception | None = None,
    point: tuple[int, int] = (298, 1568),
    screen_size: tuple[int, int] = (1080, 2400),
) -> Mock:
    requested = TapMarkAction(
        mark=25,
        target="Just once button",
        reason="choose Chrome for this action",
    )
    resolved = TapCoordinateAction(
        coordinates=Coordinates(x=point[0], y=point[1]),
        target="Just once button",
        reason="choose Chrome for this action",
    )
    response = MarksStepResponse(
        current_screen="Open with Chrome",
        observation="An Android app chooser is visible.",
        reasoning="Confirm the selected browser.",
        action=requested,
    )
    orchestrator = Mock()
    orchestrator.return_value.execute.side_effect = failure
    ctx.steps.open(step + 1)

    with (
        patch("yewo.agent.actuation.safety.ActionOrchestrator", orchestrator),
        patch("yewo.agent.step.dispatch.settle.settle", return_value=post_phash),
    ):
        dispatch.execute(
            step=step,
            action=resolved,
            requested_action=requested,
            step_response=response,
            device_control=Mock(),
            snapshot=SimpleNamespace(
                current_phash="same-screen",
                observation_id=f"observation-{step + 1}",
                gui_elements=[],
                marks=[],
                scan_info=SimpleNamespace(
                    screen_width=screen_size[0], screen_height=screen_size[1]
                ),
            ),
            session_trail=None,
            ctx=ctx,
            scrubber=scrubber,
            replay_ledger=None,
            settle_time=0.0,
        )
    return orchestrator


def test_history_keeps_the_visual_target_and_reports_the_delivery():
    ctx, scrubber = _context()

    _tap(ctx, scrubber, step=0)

    history = ctx.history.format_for_prompt()
    assert "element 25" in history
    assert "Just once button" in history
    assert "delivered to the device" in history
    assert "(298, 1568)" not in history
    assert ctx.pending_loop_nudge is None


def test_an_unchanged_screen_is_not_reported_as_a_tap_the_app_ignored():
    """The planner read "may not have responded" as licence to tap again.

    A value the app redrew unchanged and a tap the app dropped render the same
    pixels, so the screen cannot separate them. Whether to repeat a tap is a
    question about the goal, and the planner constraints own it.
    """
    ctx, scrubber = _context()

    _tap(ctx, scrubber, step=0)

    history = ctx.history.format_for_prompt()
    assert "not evidence that the app ignored this input" in history
    assert "may not have responded" not in history


def test_two_unchanged_taps_nudge_without_blocking_the_second_tap():
    ctx, scrubber = _context()

    first = _tap(ctx, scrubber, step=0)
    second = _tap(ctx, scrubber, step=1)

    first.return_value.execute.assert_called_once()
    second.return_value.execute.assert_called_once()
    assert ctx.pending_loop_nudge is not None
    assert "2 delivered taps" in ctx.pending_loop_nudge
    assert "goal explicitly requires repeated taps" in ctx.pending_loop_nudge


def test_scaled_taps_share_one_unchanged_control_identity():
    ctx, scrubber = _context()

    _tap(ctx, scrubber, step=0, point=(540, 1200), screen_size=(1080, 2400))
    _tap(ctx, scrubber, step=1, point=(270, 600), screen_size=(540, 1200))

    assert ctx.pending_loop_nudge is not None
    assert "2 delivered taps" in ctx.pending_loop_nudge


def test_a_changed_screen_does_not_count_as_an_unchanged_tap():
    ctx, scrubber = _context()

    _tap(ctx, scrubber, step=0, post_phash="next-screen")

    assert ctx.pending_loop_nudge is None


def test_a_driver_error_does_not_count_as_a_delivered_tap():
    ctx, scrubber = _context()

    _tap(ctx, scrubber, step=0, failure=RuntimeError("driver rejected the tap"))

    assert ctx.pending_loop_nudge is None
