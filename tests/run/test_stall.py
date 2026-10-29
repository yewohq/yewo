"""Giving up on a screen that will not take an action."""

import unittest
from types import SimpleNamespace


class StallTests(unittest.TestCase):
    """Steps that never reach the device cost a model call and change nothing."""

    def _stall(self):
        from yewo.agent.memory.stall import Stall

        return Stall()

    def test_a_screen_gets_more_than_one_attempt(self):
        """One refusal is a thing to read and answer, not a reason to stop."""
        stall = self._stall()

        stall.record_blocked("screenA")
        self.assertFalse(stall.is_exhausted)

    def test_a_screen_runs_out_on_the_third_blocked_step(self):
        """The count is written out, so changing it has to be deliberate.

        Reading STALL_LIMIT here instead would move with it and assert nothing.
        """
        stall = self._stall()

        stall.record_blocked("screenA")
        self.assertFalse(stall.is_exhausted)
        stall.record_blocked("screenA")
        self.assertFalse(stall.is_exhausted)
        stall.record_blocked("screenA")
        self.assertTrue(stall.is_exhausted)

    def test_reaching_a_new_screen_starts_the_count_again(self):
        """Scrolling or dismissing a keyboard changes the screen, which is progress."""
        stall = self._stall()
        stall.record_blocked("screenA")
        stall.record_blocked("screenA")

        stall.record_blocked("screenB")
        self.assertFalse(stall.is_exhausted)

    def test_an_action_that_reaches_the_device_clears_the_count(self):
        stall = self._stall()
        stall.record_blocked("screenA")
        stall.record_blocked("screenA")

        stall.action_dispatched()

        stall.record_blocked("screenA")
        self.assertFalse(stall.is_exhausted)

    def test_an_unreadable_screen_never_accumulates(self):
        """Nothing can be compared with it, so it gets the lenient reading."""
        stall = self._stall()

        for _ in range(5):
            stall.record_blocked(None)

        self.assertFalse(stall.is_exhausted)

    def test_a_spent_screen_reports_itself_without_using_an_attempt(self):
        """The execution path asks after the fact; asking must not cost a try."""
        stall = self._stall()
        stall.record_blocked("screenA")
        stall.record_blocked("screenA")

        self.assertFalse(stall.is_exhausted)
        stall.record_blocked("screenA")
        self.assertTrue(stall.is_exhausted)
        self.assertTrue(stall.is_exhausted)

    def test_a_screen_the_agent_came_back_to_is_judged_afresh(self):
        """Leaving and returning is progress, even back to the same pixels."""
        stall = self._stall()
        stall.record_blocked("screenA")
        stall.record_blocked("screenB")

        stall.record_blocked("screenA")
        self.assertFalse(stall.is_exhausted)


class KeyboardRevealTests(unittest.TestCase):
    """Looking behind the keyboard before deciding a screen is hopeless.

    Only what is on screen carries a number, and the keyboard covers the lower
    half. A field underneath it cannot be named, so the planner reaches for
    whatever is visible.
    """

    def _ctx(self):
        from yewo.agent.run.context import RunContext

        return RunContext(history=None, loop_detector=None)

    def _device(self, keyboard_open, closes=True):
        pressed = []

        def dismiss():
            pressed.append("dismiss")
            return closes

        return SimpleNamespace(
            is_keyboard_open=lambda: keyboard_open, dismiss_keyboard=dismiss
        ), pressed

    def _snapshot(self, phash="screenA"):
        return SimpleNamespace(current_phash=phash)

    def _reveal(self, device, snapshot, ctx):
        from yewo.agent.memory.stall import reveal_target_behind_keyboard

        return reveal_target_behind_keyboard(device, snapshot, ctx)

    def test_a_blocked_step_with_the_keyboard_up_closes_it(self):
        device, pressed = self._device(keyboard_open=True)

        self.assertTrue(self._reveal(device, self._snapshot(), self._ctx()))
        self.assertEqual(pressed, ["dismiss"])

    def test_nothing_happens_when_the_keyboard_is_already_down(self):
        """The screen is fully visible, so the target is genuinely absent."""
        device, pressed = self._device(keyboard_open=False)

        self.assertFalse(self._reveal(device, self._snapshot(), self._ctx()))
        self.assertEqual(pressed, [])

    def test_one_screen_gets_one_look_behind_the_keyboard(self):
        """A second failure on the same screen is not about the keyboard."""
        device, pressed = self._device(keyboard_open=True)
        ctx = self._ctx()

        self._reveal(device, self._snapshot(), ctx)

        self.assertFalse(self._reveal(device, self._snapshot(), ctx))
        self.assertEqual(pressed, ["dismiss"])

    def test_a_different_screen_gets_its_own_look(self):
        device, pressed = self._device(keyboard_open=True)
        ctx = self._ctx()

        self._reveal(device, self._snapshot("screenA"), ctx)
        self._reveal(device, self._snapshot("screenB"), ctx)

        self.assertEqual(pressed, ["dismiss", "dismiss"])

    def test_an_unknown_keyboard_state_is_not_treated_as_open(self):
        """is_keyboard_open returns None when no check could answer."""
        device, pressed = self._device(keyboard_open=None)

        self.assertFalse(self._reveal(device, self._snapshot(), self._ctx()))
        self.assertEqual(pressed, [])

    def test_a_close_that_fails_does_not_spend_the_screens_one_attempt(self):
        """iOS hide_keyboard can fail; the screen is no more visible for trying."""
        broken, broken_presses = self._device(keyboard_open=True, closes=False)
        ctx = self._ctx()

        self.assertFalse(self._reveal(broken, self._snapshot(), ctx))

        working, _ = self._device(keyboard_open=True)
        self.assertTrue(self._reveal(working, self._snapshot(), ctx))
        self.assertEqual(broken_presses, ["dismiss"])

    def test_an_unidentified_screen_is_not_recorded_as_dismissed(self):
        """Nothing can match it later, so it would consume every attempt."""
        device, pressed = self._device(keyboard_open=True)
        ctx = self._ctx()

        self.assertFalse(self._reveal(device, self._snapshot(phash=None), ctx))
        self.assertEqual(pressed, [])
        self.assertEqual(ctx.keyboard_dismissed, set())
