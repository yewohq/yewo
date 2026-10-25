"""A cached step that would not run, and the chain a run writes back.

Both are about what a part-replayed run leaves on disk. A path that
replayed three steps and planned two is one path of five, and a file
holding only the planned tail would start in the middle of the scenario.
"""

import unittest
from types import SimpleNamespace

from yewo.agent.actuation.actions import Coordinates, TapAction
from yewo.agent.memory import replay_learning, replay_store, replay_walk

import replay_driver
from yewo.agent.perception.elements import BoundingBox, GUIElement, ScanInfo
from yewo.agent.run.record import StepLog

WIDTH, HEIGHT = 1000, 2000
VALUES = {"email": "qa@example.com", "password": "secret-password123"}


def element(text, role="button", box=(100, 100, 300, 200)):
    left, top, right, bottom = box
    return GUIElement(
        element=text or role,
        bounding_box=BoundingBox(x_min=left, y_min=top, x_max=right, y_max=bottom),
        confidence=0.9,
        type=role,
        text=text,
    )


def scan(count=1, phash="p"):
    return ScanInfo(
        mode="detect",
        reason="step",
        phash=phash,
        screen_width=WIDTH,
        screen_height=HEIGHT,
        element_count=count,
    )


def screen_read(elements, info=None):
    return SimpleNamespace(
        gui_elements=list(elements),
        scan_info=info or scan(len(elements)),
        current_phash="p",
        raw_img_bytes=b"",
    )


def centre_of(target):
    box = target.bounding_box
    return Coordinates(x=(box.x_min + box.x_max) // 2, y=(box.y_min + box.y_max) // 2)


class Device:
    """A device that records what was sent to it."""

    def __init__(self):
        self.sent = []

    def get_screenshot(self):
        return b"not a png, which target_moved reads as no evidence of movement"

    def tap(self, x, y):
        self.sent.append(("tap", x, y))
        return True


def identity():
    return replay_store.TraceIdentity(
        scenario_uuid="s",
        scenario_digest="d",
        platform="android",
        target_profile="Pixel_8",
        detector="d@abc",
        grounding="marks",
    )


class FailedStepTests(unittest.TestCase):
    """A cached step the device would not take.

    The attempt still happened. A run that dropped it would hand the
    planner a screen and no record of a tap, and a non-idempotent tap sent
    twice is the thing a saved path must never cause.
    """

    def setUp(self):
        self.names = replay_store.ValueNames(VALUES)
        self.log = StepLog()
        self.device = Device()

        class Refusing:
            def __init__(self, device_control, scrubber=None):
                pass

            def execute(self, action):
                raise RuntimeError("the driver said no")

        real = replay_walk.ActionOrchestrator
        replay_walk.ActionOrchestrator = Refusing
        self.addCleanup(setattr, replay_walk, "ActionOrchestrator", real)

    def _walk(self):
        sign_in = element("Sign in")
        trace = replay_walk.build_trace(
            identity(),
            [TapAction(coordinates=centre_of(sign_in), reason="r")],
            [(([sign_in], scan()), ([element("Orders")], scan()))],
            self.names,
        )
        replay = replay_learning.DurableReplay(trace=trace, values=dict(VALUES), names=self.names)
        sources, ctx = replay_driver.drive(
            replay, self.device, [screen_read([sign_in]), screen_read([sign_in])]
        )
        return SimpleNamespace(
            sources=sources,
            records=ctx.steps.records,
            replay=replay,
            steps_taken=sources.count("cache"),
        )

    def test_the_attempt_is_counted_against_the_budget(self):
        result = self._walk()

        self.assertEqual(result.steps_taken, 1)

    def test_the_attempt_is_recorded_as_a_step_that_failed(self):
        result = self._walk()

        self.assertFalse(result.records[0].outcome.success)
        self.assertIn("would not run", result.records[0].outcome.observation)

    def test_the_attempt_is_not_learned_from(self):
        """It may have half happened. A path built from it would send a
        step the device already refused once."""
        result = self._walk()

        self.assertEqual(result.replay.recorded_steps, [])

    def test_the_next_step_is_planned_rather_than_replayed(self):
        result = self._walk()

        self.assertEqual(result.sources, ["cache", "model"])


class ChainTests(unittest.TestCase):
    """What a run writes back after replaying part of a path."""

    def setUp(self):
        self.replay = replay_learning.DurableReplay(
            trace=None, values={}, names=replay_store.ValueNames()
        )

    def _tap_on(self, target):
        return TapAction(coordinates=centre_of(target), reason="r")

    def test_a_cached_head_and_a_planned_tail_are_one_chain(self):
        """A file holding only the tail would start in the middle of the
        scenario, and the next run would refuse it on step one."""
        sign_in, orders = element("Sign in"), element("Orders")
        self.replay.record_step(self._tap_on(sign_in), screen_read([sign_in]), 1000.0)
        self.replay.record_step(self._tap_on(orders), screen_read([orders]), 2000.0)
        self.replay.record_arrival(screen_read([element("Order confirmed", "text")]))

        steps, screens = self.replay.select_learned_steps()

        self.assertEqual(len(steps), 2)
        self.assertEqual(len(screens), 2)

    def test_a_step_keeps_its_own_action_to_arrival_duration(self):
        sign_in = element("Sign in")
        self.replay.record_step(self._tap_on(sign_in), screen_read([sign_in]), 1000.0)
        self.replay.record_step(self._tap_on(sign_in), screen_read([sign_in]), 1750.0)
        self.replay.select_learned_steps()

        self.assertEqual(self.replay.selected_arrival_ms, [1000])

    def test_a_run_that_took_no_step_is_not_worth_saving(self):
        self.assertFalse(self.replay.is_worth_saving)

    def test_an_arrival_before_any_step_is_not_kept(self):
        """It is a reading of the screen the scenario started on, and
        nothing arrived there."""
        self.replay.record_arrival(screen_read([element("Sign in")]))

        self.assertEqual(self.replay.recorded_steps, [])

    def test_an_arrival_is_taken_once(self):
        """A second one would pair the final screen with itself."""
        sign_in = element("Sign in")
        self.replay.record_step(self._tap_on(sign_in), screen_read([sign_in]), 1000.0)
        self.replay.record_arrival(screen_read([element("Orders")]))
        self.replay.record_arrival(screen_read([element("Orders")]))

        self.assertEqual(len(self.replay.recorded_steps), 2)


if __name__ == "__main__":
    unittest.main()


class LearningSegmentTests(unittest.TestCase):
    """A failed action ends one segment without discarding later success."""

    def setUp(self):
        self.replay = replay_learning.DurableReplay(
            trace=None, values={}, names=replay_store.ValueNames()
        )
        self.sign_in = element("Sign in")
        self.tap = TapAction(coordinates=centre_of(self.sign_in), reason="r")

    def test_the_failed_step_is_not_in_the_path(self):
        self.replay.record_step(self.tap, screen_read([self.sign_in]), 1000.0)
        self.replay.close_learning_segment(screen_read([element("Orders")]))

        steps, screens = self.replay.select_learned_steps()

        self.assertEqual(len(steps), 1)
        self.assertEqual(len(screens), 1)

    def test_a_successful_retry_after_the_failure_is_learned(self):
        self.replay.record_step(self.tap, screen_read([self.sign_in]), 1000.0)
        self.replay.close_learning_segment(screen_read([element("Orders")]))
        basket = element("Basket")
        self.replay.record_step(
            TapAction(coordinates=centre_of(basket), reason="r"),
            screen_read([basket]),
            3000.0,
        )
        self.replay.record_arrival(screen_read([element("Order confirmed", "text")]))

        steps, _ = self.replay.select_learned_steps()

        self.assertEqual(len(steps), 2)
        self.assertEqual(self.replay.selected_segment_starts, [False, True])

    def test_the_screen_the_failure_read_closes_the_path(self):
        """Without it the last good step has no destination and is lost."""
        self.replay.record_step(self.tap, screen_read([self.sign_in]), 1000.0)
        self.replay.close_learning_segment(screen_read([element("Orders")]))

        _, screens = self.replay.select_learned_steps()

        self.assertEqual([found.text for found in screens[0][1][0]], ["Orders"])

    def test_a_first_step_that_failed_leaves_nothing(self):
        self.replay.close_learning_segment(screen_read([self.sign_in]))

        self.assertFalse(self.replay.is_worth_saving)

    def test_a_rejected_entry_is_replaced_by_its_successful_retry(self):
        self.replay.record_step(self.tap, screen_read([self.sign_in]), 1000.0)
        self.replay.reject_last_step()
        self.replay.record_step(self.tap, screen_read([self.sign_in]), 2000.0)
        self.replay.record_arrival(screen_read([element("Orders")]))

        steps, _ = self.replay.select_learned_steps()

        self.assertEqual(steps, [self.tap])
        self.assertEqual(self.replay.selected_segment_starts, [True])

    def test_success_after_a_first_failure_starts_a_segment(self):
        self.replay.close_learning_segment(screen_read([self.sign_in]))
        self.replay.record_step(self.tap, screen_read([self.sign_in]), 1000.0)
        self.replay.record_arrival(screen_read([element("Orders")]))

        self.replay.select_learned_steps()

        self.assertEqual(self.replay.selected_segment_starts, [True])

    def test_each_failure_closes_only_its_current_segment(self):
        orders = element("Orders")
        basket = element("Basket")
        self.replay.record_step(self.tap, screen_read([self.sign_in]), 1000.0)
        self.replay.close_learning_segment(screen_read([orders]))
        self.replay.record_step(
            TapAction(coordinates=centre_of(orders), reason="r"),
            screen_read([orders]),
            2000.0,
        )
        self.replay.close_learning_segment(screen_read([basket]))
        self.replay.record_step(
            TapAction(coordinates=centre_of(basket), reason="r"),
            screen_read([basket]),
            3000.0,
        )
        self.replay.record_arrival(screen_read([element("Confirmed", "text")]))

        steps, _ = self.replay.select_learned_steps()

        self.assertEqual(len(steps), 3)
        self.assertEqual(self.replay.selected_segment_starts, [False, True, True])
