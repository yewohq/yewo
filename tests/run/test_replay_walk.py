"""Re-walk saved paths and refuse them when they no longer fit."""

import unittest
from types import SimpleNamespace

from yewo.agent.actuation.actions import (
    Coordinates,
    SwipeAction,
    TapAction,
    TypeAction,
)
from yewo.agent.memory import replay_learning, replay_store, replay_walk

import replay_driver
from yewo.agent.perception.elements import BoundingBox, GUIElement, ScanInfo
from yewo.agent.run.record import StepLog, StepSource

WIDTH, HEIGHT = 1000, 2000
A_SCREEN = b"not a png, which target_moved reads as no evidence of movement"
PASSWORD = "secret-password123"
VALUES = {"email": "qa@example.com", "password": PASSWORD}


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


def centre_of(element):
    box = element.bounding_box
    return Coordinates(x=(box.x_min + box.x_max) // 2, y=(box.y_min + box.y_max) // 2)


class Device:
    """A device that records what was sent to it.

    It answers `get_screenshot` because the stale-target check re-reads the
    screen immediately before acting, the same as it does for a planned
    action. `frames` makes it answer with a changed screen.
    """

    def __init__(self, frames=None):
        self.sent = []
        self.frames = list(frames or [])

    def get_screenshot(self):
        # Non-empty by default: an unreadable screen is itself a refusal,
        # and that is a different case from the one under test.
        return self.frames.pop(0) if self.frames else A_SCREEN

    def tap(self, x, y):
        self.sent.append(("tap", x, y))
        return True


class Orchestrator:
    def __init__(self, device_control, scrubber=None):
        self.device = device_control

    def execute(self, action):
        self.device.sent.append(action)


def identity():
    return replay_store.TraceIdentity(
        scenario_uuid="s",
        scenario_digest="d",
        platform="android",
        target_profile="Pixel_8",
        detector="d@abc",
        grounding="marks",
    )


class LearnTests(unittest.TestCase):
    """What a passing scenario leaves behind."""

    def setUp(self):
        self.names = replay_store.ValueNames(VALUES)

    def _learn(self, steps, screens):
        return replay_walk.build_trace(identity(), steps, screens, self.names)

    def test_a_tap_becomes_a_transition_naming_its_target(self):
        sign_in = element("Sign in")
        before = ([sign_in], scan())
        after = ([element("Orders")], scan())
        step = TapAction(coordinates=centre_of(sign_in), reason="r")

        trace = self._learn([step], [(before, after)])

        self.assertEqual(len(trace.transitions), 1)
        self.assertEqual(
            trace.transitions[0].action.target.text,
            replay_store.text_identity("Sign in", self.names),
        )

    def test_a_typed_secret_is_kept_as_its_name(self):
        field = element(PASSWORD, "field")
        step = TypeAction(text=PASSWORD, coordinates=centre_of(field), reason="r")

        trace = self._learn([step], [(([field], scan()), ([field], scan()))])

        written = trace.model_dump_json()
        self.assertNotIn(PASSWORD, written)
        self.assertEqual(trace.transitions[0].action.value_name, "password")

    def test_a_later_learning_segment_is_marked(self):
        first = element("Sign in")
        retry = element("Card Number")
        trace = replay_walk.build_trace(
            identity(),
            [
                TapAction(coordinates=centre_of(first), reason="r"),
                TapAction(coordinates=centre_of(retry), reason="r"),
            ],
            [
                (([first], scan(phash="first")), ([element("Keyboard")], scan(phash="gap"))),
                (([retry], scan(phash="retry")), ([element("Done")], scan(phash="done"))),
            ],
            self.names,
            segment_starts=[False, True],
        )

        self.assertFalse(trace.transitions[0].starts_segment)
        self.assertTrue(trace.transitions[1].starts_segment)

    def test_text_the_journey_never_declared_ends_the_trace(self):
        """Keeping it would mean a literal string on disk."""
        field = element("typed by hand", "field")
        step = TypeAction(text="typed by hand", coordinates=centre_of(field), reason="r")

        self.assertIsNone(self._learn([step], [(([field], scan()), ([field], scan()))]))

    def test_a_destination_is_kept_for_every_transition(self):
        sign_in = element("Sign in")
        step = TapAction(coordinates=centre_of(sign_in), reason="r")

        trace = self._learn([step], [(([sign_in], scan()), ([element("Orders")], scan()))])

        self.assertEqual(
            trace.transitions[0].destination.anchors[0].text,
            replay_store.text_identity("Orders", self.names),
        )

    def test_a_tap_landing_on_nothing_is_not_kept(self):
        """A target read back from no element is a target nobody can find."""
        step = TapAction(coordinates=Coordinates(x=900, y=1900), reason="r")

        self.assertIsNone(self._learn([step], [(([element("Sign in")], scan()), ([], scan(0)))]))

    def test_a_repeated_tap_keeps_its_count_and_its_gap(self):
        """A four-tap at 500ms is a concurrency test. One tap is not."""
        target = element("Place order")
        step = TapAction(coordinates=centre_of(target), count=4, interval_ms=500, reason="r")

        trace = self._learn([step], [(([target], scan()), ([target], scan()))])

        self.assertEqual(trace.transitions[0].action.count, 4)
        self.assertEqual(trace.transitions[0].action.interval_ms, 500)

    def test_a_destination_remembers_what_told_it_apart(self):
        """A failure keeps the surrounding controls and loses only the
        thing that said it worked, so the share alone forgives it."""
        pay = element("Pay")
        trace = self._learn(
            [TapAction(coordinates=centre_of(pay), reason="r")],
            [
                (
                    ([pay, element("Basket", box=(100, 500, 300, 600))], scan(2)),
                    (
                        [element("Order confirmed"), element("Basket", box=(100, 500, 300, 600))],
                        scan(2),
                    ),
                )
            ],
        )

        self.assertEqual(
            trace.transitions[0].destination.distinguishing,
            (("button", replay_store.text_identity("Order confirmed", self.names)),),
        )

    def test_a_swipe_is_saved_as_viewport_fractions(self):
        step = SwipeAction(
            start_x=100,
            start_y=400,
            end_x=900,
            end_y=1600,
            reason="Scroll",
        )

        trace = self._learn(
            [step],
            [
                (
                    ([element("Products")], scan()),
                    ([element("More products")], scan()),
                )
            ],
        )

        saved = trace.transitions[0].action
        self.assertEqual(
            (saved.start_x, saved.start_y, saved.end_x, saved.end_y),
            (100, 200, 900, 800),
        )

    def test_a_swipe_scales_to_the_current_viewport(self):
        cached = replay_store.CachedAction(
            kind="swipe", start_x=100, start_y=200, end_x=900, end_y=800
        )
        current = ScanInfo(
            mode="detect",
            reason="step",
            phash="p",
            screen_width=2000,
            screen_height=4000,
            element_count=1,
        )

        action = replay_walk._resolve_action(cached, None, {}, current)

        self.assertEqual(
            (action.start_x, action.start_y, action.end_x, action.end_y),
            (200, 800, 1800, 3200),
        )

    def test_no_steps_leave_no_trace(self):
        self.assertIsNone(self._learn([], []))


class WalkTests(unittest.TestCase):
    """Re-walking what was saved."""

    def setUp(self):
        self.names = replay_store.ValueNames(VALUES)
        self.device = Device()
        self.log = StepLog()
        self._real = replay_walk.ActionOrchestrator
        replay_walk.ActionOrchestrator = Orchestrator
        self.addCleanup(setattr, replay_walk, "ActionOrchestrator", self._real)

    def _trace(self, before, after, action):
        return replay_walk.build_trace(
            identity(),
            [action],
            [((before, scan(len(before))), (after, scan(len(after))))],
            self.names,
        )

    def _walk(self, trace, screens, values=None):
        """One turn of the loop per screen, through the real service."""
        readings = [
            screen_read(*screen) if isinstance(screen, tuple) else screen for screen in screens
        ]
        replay = replay_learning.DurableReplay(
            trace=trace,
            values=dict(values or VALUES),
            names=replay_store.ValueNames(values or VALUES),
        )
        sources, ctx = replay_driver.drive(replay, self.device, readings)
        self.log = ctx.steps
        return SimpleNamespace(
            sources=sources,
            steps_taken=sources.count("cache"),
            is_whole=all(source == "cache" for source in sources),
            records=ctx.steps.records,
        )

    def test_an_unchanged_screen_replays_the_step(self):
        sign_in = element("Sign in")
        trace = self._trace(
            [sign_in], [element("Orders")], TapAction(coordinates=centre_of(sign_in), reason="r")
        )

        result = self._walk(
            trace,
            [
                ([sign_in], scan()),
                ([element("Orders")], scan()),
            ],
        )

        # The chain holds one step. The second turn has none left, so it
        # falls to the planner rather than being refused.
        self.assertEqual(result.sources, ["cache", "model"])

    def test_a_moved_control_is_tapped_where_it_is_now(self):
        """The whole reason the file holds no coordinate."""
        sign_in = element("Sign in")
        trace = self._trace(
            [sign_in], [element("Orders")], TapAction(coordinates=centre_of(sign_in), reason="r")
        )
        moved = element("Sign in", box=(100, 130, 300, 230))

        self._walk(trace, [([moved], scan()), ([element("Orders")], scan())])

        sent = self.device.sent[0]
        self.assertEqual(sent.coordinates.y, (130 + 230) // 2)

    def test_a_vanished_control_stops_before_acting(self):
        sign_in = element("Sign in")
        trace = self._trace(
            [sign_in], [element("Orders")], TapAction(coordinates=centre_of(sign_in), reason="r")
        )

        result = self._walk(trace, [([element("Something else")], scan())])

        self.assertFalse(result.is_whole)
        self.assertEqual(self.device.sent, [])

    def test_a_twinned_control_stops_before_acting(self):
        sign_in = element("Sign in")
        trace = self._trace(
            [sign_in], [element("Orders")], TapAction(coordinates=centre_of(sign_in), reason="r")
        )
        twinned = [sign_in, element("Sign in", box=(100, 900, 300, 1000))]

        result = self._walk(trace, [(twinned, scan(2))])

        self.assertFalse(result.is_whole)
        self.assertEqual(self.device.sent, [])

    def test_arriving_somewhere_else_stops_after_acting(self):
        """A unique source match only proves something plausible was tapped."""
        sign_in = element("Sign in")
        trace = self._trace(
            [sign_in], [element("Orders")], TapAction(coordinates=centre_of(sign_in), reason="r")
        )

        result = self._walk(
            trace,
            [
                ([sign_in], scan()),
                ([element("Error", "text")], scan()),
            ],
        )

        self.assertFalse(result.is_whole)
        self.assertEqual(len(self.device.sent), 1)
        self.assertFalse(self.log.records[0].outcome.success)

    def test_a_screen_that_cannot_be_re_read_stops_before_acting(self):
        """The same check a planned action gets. A cached step has no
        planner to notice the app moved between reading and acting."""
        sign_in = element("Sign in")
        trace = self._trace(
            [sign_in], [element("Orders")], TapAction(coordinates=centre_of(sign_in), reason="r")
        )
        self.device = Device(frames=[b""])

        result = self._walk(trace, [([sign_in], scan()), ([element("Orders")], scan())])

        self.assertFalse(result.is_whole)
        self.assertEqual(self.device.sent, [])

    def test_a_failure_keeping_the_furniture_is_still_refused(self):
        """The share would pass it. What the step produced is missing."""
        pay = element("Pay")
        basket = element("Basket", box=(100, 500, 300, 600))
        trace = self._trace(
            [pay, basket],
            [element("Order confirmed"), basket],
            TapAction(coordinates=centre_of(pay), reason="r"),
        )

        result = self._walk(
            trace,
            [
                ([pay, basket], scan(2)),
                ([element("Payment declined", "text"), basket], scan(2)),
            ],
        )

        # It acted, then the screen it reached was not the one the step
        # used to reach, so the step is marked failed and the next turn
        # is planned.
        self.assertEqual(result.sources, ["cache", "model"])
        self.assertFalse(result.records[0].outcome.success)

    def test_a_replayed_step_is_recorded_as_a_cached_one(self):
        sign_in = element("Sign in")
        trace = self._trace(
            [sign_in], [element("Orders")], TapAction(coordinates=centre_of(sign_in), reason="r")
        )

        self._walk(trace, [([sign_in], scan()), ([element("Orders")], scan())])

        step = self.log.records[0]
        self.assertEqual(step.source, StepSource.CACHE)
        self.assertEqual(step.n, 1)
        self.assertIsNone(step.decision, "no model saw this screen")
        self.assertEqual(
            step.action.target_label,
            replay_store.text_identity("Sign in", self.names),
        )

    def test_a_repeated_tap_is_replayed_as_a_repeated_tap(self):
        target = element("Place order")
        trace = self._trace(
            [target],
            [target],
            TapAction(coordinates=centre_of(target), count=4, interval_ms=500, reason="r"),
        )

        self._walk(trace, [([target], scan()), ([target], scan())])

        sent = self.device.sent[0]
        self.assertEqual((sent.count, sent.interval_ms), (4, 500))

    def test_a_typed_step_resolves_its_value_at_the_device(self):
        """The file holds a name. The run holds the value."""
        field = element(PASSWORD, "field")
        trace = self._trace(
            [field], [field], TypeAction(text=PASSWORD, coordinates=centre_of(field), reason="r")
        )

        self._walk(trace, [([field], scan()), ([field], scan())])

        self.assertEqual(self.device.sent[0].text, PASSWORD)

    def test_a_typed_step_uses_the_value_this_run_declared(self):
        """A different account walks the same saved path."""
        field = element(PASSWORD, "field")
        trace = self._trace(
            [field], [field], TypeAction(text=PASSWORD, coordinates=centre_of(field), reason="r")
        )
        now = {"password": "a-different-password"}

        self._walk(
            trace,
            [screen_read([element("a-different-password", "field")])] * 2,
            values=now,
        )

        self.assertEqual(self.device.sent[0].text, "a-different-password")


class RealBoundaryTests(unittest.TestCase):
    """What the loop actually hands over, rather than a shape built here.

    The first version of this took a `StepRecord`, whose `action` is an
    `ActionRecord`: words about what was done, with no coordinate and no
    kind an orchestrator accepts. Every test passed and no real run could
    ever have learned a path.
    """

    def test_what_the_loop_records_can_be_turned_into_a_trace(self):
        from yewo.agent.run.record import StepLog

        log = StepLog()
        log.open(1)
        sign_in = element("Sign in")
        action = TapAction(coordinates=centre_of(sign_in), reason="r")
        log.record_action(
            action,
            "tapped Sign in",
            SimpleNamespace(success=True, observation="ok", error=None),
            "Sign in",
        )

        replay = replay_learning.DurableReplay(
            trace=None, values={}, names=replay_store.ValueNames()
        )
        replay.record_step(action, screen_read([sign_in]))
        replay.record_arrival(screen_read([element("Orders")]))
        actions, screens = replay.select_learned_steps()

        trace = replay_walk.build_trace(identity(), actions, screens, replay_store.ValueNames())

        self.assertIsNotNone(trace, "a real run learned nothing")
        self.assertEqual(len(trace.transitions), 1)

    def test_a_step_record_is_not_something_a_path_can_be_built_from(self):
        """Named so the next person does not reach for it."""
        from yewo.agent.run.record import StepLog

        log = StepLog()
        step = log.open(1)
        log.record_action(
            TapAction(coordinates=Coordinates(x=1, y=1), reason="r"),
            "tapped",
            SimpleNamespace(success=True, observation="ok", error=None),
            "Sign in",
        )

        self.assertIsNone(
            replay_walk._build_cached_action(step.action, None, replay_store.ValueNames())
        )


if __name__ == "__main__":
    unittest.main()
