"""A control whose screen moves every time a plan is made on it.

A live run aborted five attempts on one control, each reporting the same
displacement. Every abort was treated as a fresh start, so the planner chose
the same control again and the plan was thrown away again. The control is
counted by the position it resolved to, so a rescan that renumbers it does not
reset the count.
"""

import unittest
from types import SimpleNamespace

from yewo.agent.actuation.actions import Coordinates, TapMarkAction
from yewo.agent.actuation.coordinate_actions import TapCoordinateAction
from yewo.agent.memory.history import AgentHistory
from yewo.agent.memory.loop_detector import (
    STALE_REPEATS_BEFORE_WARNING,
    RollingLoopDetector,
)
from yewo.agent.run import outcomes
from yewo.agent.run.context import RunContext
from yewo.agent.step.targeting import Decision

SCREEN_A = "daa82d505aa82d5058a225545822b04d2d57f2af2d57f2af2d57d2af2d57d2ab"
SCREEN_A_JITTER = "daa82d505aa82d5058a225545823b00d2d57f2af2d57f2af2d57d2af2d57d2ab"
SCREEN_B = "d8a82d5058a021504800f2a92d56d2ab2d57d2ab2d57f22fbc7fa55fdaafa555"
SIZE = (1080, 2400)


def _context() -> RunContext:
    return RunContext(history=AgentHistory(), loop_detector=RollingLoopDetector())


def _tap(x: int, y: int) -> TapCoordinateAction:
    return TapCoordinateAction(
        coordinates=Coordinates(x=x, y=y), target="Add", reason="add a schedule"
    )


def _snapshot(phash: str):
    return SimpleNamespace(
        current_phash=phash,
        scan_info=SimpleNamespace(screen_width=SIZE[0], screen_height=SIZE[1]),
    )


class StaleRepeatTests(unittest.TestCase):
    def _abort(
        self,
        ctx: RunContext,
        times: int,
        x: int = 961,
        y: int = 2219,
        screen: str = SCREEN_A,
    ) -> None:
        for _ in range(times):
            resolved = _tap(x, y)
            outcomes.unresolved_control(
                ctx,
                TapMarkAction(mark=12, target="Add", reason="add a schedule"),
                Decision(
                    ok=False,
                    observation="moved",
                    stale=True,
                    action=resolved,
                ),
                _snapshot(screen),
            )

    def test_one_abort_says_nothing(self):
        ctx = _context()
        self._abort(ctx, 1)

        self.assertIsNone(ctx.pending_loop_nudge)

    def test_a_third_abort_on_one_control_warns(self):
        ctx = _context()
        self._abort(ctx, STALE_REPEATS_BEFORE_WARNING)

        self.assertIn("quarantined", ctx.pending_loop_nudge or "")

    def test_the_count_follows_the_control_not_the_step(self):
        """Jitter inside one bucket is the same control."""
        ctx = _context()
        self._abort(ctx, 1, 961, 2219)
        self._abort(ctx, 1, 964, 2221)
        self._abort(ctx, 1, 960, 2218)

        self.assertIn("quarantined", ctx.pending_loop_nudge or "")

    def test_two_controls_are_counted_apart(self):
        ctx = _context()
        self._abort(ctx, 2, 961, 2219)
        self._abort(ctx, 2, 73, 205)

        self.assertNotIn("quarantined", ctx.pending_loop_nudge or "")

    def test_the_same_position_on_two_screens_is_counted_apart(self):
        ctx = _context()
        self._abort(ctx, 2, screen=SCREEN_A)
        self._abort(ctx, 2, screen=SCREEN_B)

        self.assertNotIn("quarantined", ctx.pending_loop_nudge or "")

    def test_small_screen_hash_jitter_keeps_one_stale_count(self):
        ctx = _context()
        self._abort(ctx, 1, screen=SCREEN_A)
        self._abort(ctx, 1, screen=SCREEN_A_JITTER)
        self._abort(ctx, 1, screen=SCREEN_A)

        self.assertIn("quarantined", ctx.pending_loop_nudge or "")

    def test_a_successful_dispatch_does_not_clear_stale_evidence(self):
        ctx = _context()
        self._abort(ctx, STALE_REPEATS_BEFORE_WARNING)
        ctx.pending_loop_nudge = None

        ctx.loop_detector.record_success(_tap(961, 2219), SCREEN_A_JITTER, SIZE)
        self._abort(ctx, 2)

        self.assertIn("quarantined", ctx.pending_loop_nudge or "")

    def test_stale_targets_share_the_bounded_attempt_policy(self):
        ctx = _context()
        self._abort(ctx, 13)

        self.assertTrue(ctx.loop_detector.action_repetition_terminated)

    def test_an_abort_for_another_reason_is_not_counted(self):
        """A target nothing numbered did not go stale."""
        ctx = _context()
        for _ in range(5):
            outcomes.unresolved_control(
                ctx,
                _tap(961, 2219),
                Decision(ok=False, observation="not listed"),
                _snapshot(SCREEN_A),
            )

        self.assertIsNone(ctx.pending_loop_nudge)


if __name__ == "__main__":
    unittest.main()
