"""Replay resumes a learned path only at a verified segment boundary."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import replay_driver
from yewo.agent.actuation.actions import Coordinates, TapAction
from yewo.agent.memory import replay_learning, replay_store, replay_walk
from yewo.agent.perception.elements import BoundingBox, GUIElement, ScanInfo
from yewo.agent.run import saved_path

WIDTH, HEIGHT = 1000, 2000


def element(text, role="button", box=(100, 100, 300, 200)):
    left, top, right, bottom = box
    return GUIElement(
        element=text,
        bounding_box=BoundingBox(x_min=left, y_min=top, x_max=right, y_max=bottom),
        confidence=0.9,
        type=role,
        text=text,
    )


def scan(phash):
    return ScanInfo(
        mode="detect",
        reason="step",
        phash=phash,
        screen_width=WIDTH,
        screen_height=HEIGHT,
        element_count=1,
    )


def reading(item, phash):
    return SimpleNamespace(
        gui_elements=[item],
        scan_info=scan(phash),
        current_phash=phash,
        raw_img_bytes=b"",
    )


def centre(item):
    box = item.bounding_box
    return Coordinates(x=(box.x_min + box.x_max) // 2, y=(box.y_min + box.y_max) // 2)


def identity():
    return replay_store.TraceIdentity(
        scenario_uuid="s",
        scenario_digest="d",
        platform="android",
        target_profile="Pixel_8",
        detector="d@abc",
        grounding="marks",
    )


class Device:
    def __init__(self):
        self.sent = []

    def get_screenshot(self):
        return b"not a png"

    def tap(self, x, y):
        self.sent.append(("tap", x, y))
        return True


class Orchestrator:
    def __init__(self, device_control, scrubber=None):
        self.device = device_control

    def execute(self, action):
        self.device.sent.append(action)


class ReplaySegmentTests(unittest.TestCase):
    def setUp(self):
        self.names = replay_store.ValueNames({})
        self.device = Device()
        self.patcher = patch("yewo.agent.memory.replay_walk.ActionOrchestrator", Orchestrator)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_a_segment_waits_until_its_source_screen_is_observed(self):
        sign_in = element("Sign in")
        keyboard = element("Keyboard")
        card_field = element("Card Number")
        confirmed = element("Order confirmed", "text")
        trace = replay_walk.build_trace(
            identity(),
            [
                TapAction(coordinates=centre(sign_in), reason="r"),
                TapAction(coordinates=centre(card_field), reason="r"),
            ],
            [
                (([sign_in], scan("sign-in")), ([keyboard], scan("keyboard"))),
                (([card_field], scan("card-field")), ([confirmed], scan("confirmed"))),
            ],
            self.names,
            artifact="build",
            screen_descriptions=["Sign in", "Card form"],
            destination_descriptions=["Keyboard open", "Order confirmed"],
            segment_starts=[False, True],
        )
        replay = replay_learning.DurableReplay(
            trace=trace,
            values={},
            names=self.names,
            current_artifact="build",
        )

        sources, _ = replay_driver.drive(
            replay,
            self.device,
            [
                reading(sign_in, "sign-in"),
                reading(keyboard, "keyboard"),
                reading(card_field, "card-field"),
            ],
        )

        self.assertEqual(sources, ["cache", "model", "cache"])

    def test_a_segment_start_cannot_be_skipped_for_a_later_matching_step(self):
        sign_in = element("Sign in")
        card_field = element("Card Number")
        cvv_field = element("CVV")
        place_order = element("Place Order")
        confirmed = element("Order confirmed", "text")
        trace = replay_walk.build_trace(
            identity(),
            [
                TapAction(coordinates=centre(sign_in), reason="r"),
                TapAction(coordinates=centre(cvv_field), reason="r"),
                TapAction(coordinates=centre(place_order), reason="r"),
            ],
            [
                (([sign_in], scan("sign-in")), ([card_field], scan("card-field"))),
                (([cvv_field], scan("cvv")), ([place_order], scan("place-order"))),
                (([place_order], scan("place-order")), ([confirmed], scan("confirmed"))),
            ],
            self.names,
            artifact="build",
            screen_descriptions=["Sign in", "CVV field", "Place Order"],
            destination_descriptions=["Card form", "Place Order", "Order confirmed"],
            segment_starts=[False, True, False],
        )
        replay = replay_learning.DurableReplay(
            trace=trace,
            values={},
            names=self.names,
            current_artifact="build",
        )
        replay.position = 1

        sources, _ = replay_driver.drive(
            replay,
            self.device,
            [reading(place_order, "place-order")],
        )

        self.assertEqual(sources, ["model"])
        self.assertEqual(replay.position, 1)
        self.assertEqual(self.device.sent, [])

    def test_a_destination_mismatch_does_not_rewind_the_path(self):
        sign_in = element("Sign in")
        orders = element("Orders")
        error = element("Invalid CVV")
        trace = replay_walk.build_trace(
            identity(),
            [TapAction(coordinates=centre(sign_in), reason="r")],
            [
                (([sign_in], scan("sign-in")), ([orders], scan("orders"))),
            ],
            self.names,
            artifact="build",
        )
        replay = replay_learning.DurableReplay(
            trace=trace,
            values={},
            names=self.names,
            current_artifact="build",
        )

        sources, _ = replay_driver.drive(
            replay,
            self.device,
            [reading(sign_in, "sign-in"), reading(error, "error")],
        )

        self.assertEqual(sources, ["cache", "model"])
        self.assertEqual(replay.position, 1)

    def test_a_transient_screen_does_not_consume_the_next_saved_step(self):
        sign_in = element("Sign in")
        orders = element("Orders")
        trace = replay_walk.build_trace(
            identity(),
            [TapAction(coordinates=centre(sign_in), reason="r")],
            [(([sign_in], scan("sign-in")), ([orders], scan("orders")))],
            self.names,
            artifact="build",
            screen_descriptions=["Sign in"],
        )
        replay = replay_learning.DurableReplay(
            trace=trace,
            values={},
            names=self.names,
            current_artifact="build",
        )

        with patch(
            "yewo.agent.memory.replay_walk.take_cached_step",
            return_value=replay_walk.StepAttempt(
                refusal="the saved screen description does not match",
                source_mismatch=True,
            ),
        ):
            sources, _ = replay_driver.drive(
                replay,
                self.device,
                [reading(sign_in, "sign-in")],
            )

        self.assertEqual(sources, ["model"])
        self.assertEqual(replay.position, 0)

    def test_a_forward_match_records_the_transition_that_actually_ran(self):
        sign_in = element("Sign in")
        orders = element("Orders")
        place_order = element("Place Order")
        trace = replay_walk.build_trace(
            identity(),
            [
                TapAction(coordinates=centre(sign_in), reason="r"),
                TapAction(coordinates=centre(place_order), reason="r"),
            ],
            [
                (([sign_in], scan("sign-in")), ([orders], scan("orders"))),
                (
                    ([place_order], scan("place-order")),
                    ([element("Confirmed")], scan("confirmed")),
                ),
            ],
            self.names,
            artifact="build",
            target_descriptions=["Sign in", "Place Order"],
        )
        replay = replay_learning.DurableReplay(
            trace=trace,
            values={},
            names=self.names,
            current_artifact="build",
        )

        sources, _ = replay_driver.drive(
            replay,
            self.device,
            [reading(place_order, "place-order")],
        )

        self.assertEqual(sources, ["cache"])
        self.assertEqual(replay.position, 2)
        self.assertEqual(replay.target_descriptions[-1], "Place Order")

    def test_a_forward_match_prefers_the_closest_valid_saved_screen(self):
        shared = element("Login")
        trace = replay_walk.build_trace(
            identity(),
            [
                TapAction(coordinates=centre(shared), reason="r"),
                TapAction(coordinates=centre(shared), reason="r"),
            ],
            [
                (([shared], scan("0" * 64)), ([shared], scan("8" * 64))),
                (([shared], scan("f" * 64)), ([shared], scan("7" * 64))),
            ],
            self.names,
            artifact="build",
        )
        current = reading(shared, "f" * 64)

        selected = replay_walk.select_transition(trace, 0, current, self.names)

        self.assertEqual(selected, 1)

    def test_cached_actions_do_not_count_the_observed_page_twice(self):
        payment = element("Credit / Debit Card")
        action = TapAction(coordinates=centre(payment), reason="r")
        trace = replay_walk.build_trace(
            identity(),
            [action] * 4,
            [
                (([payment], scan("payment")), ([payment], scan("payment")))
                for _ in range(4)
            ],
            self.names,
            artifact="build",
        )
        replay = replay_learning.DurableReplay(
            trace=trace,
            values={},
            names=self.names,
            current_artifact="build",
        )
        ctx = replay_driver.context()
        agent = replay_driver.agent_for(replay)
        screen = reading(payment, "payment")
        settling = replay_driver.Settling("payment")

        for step in range(4):
            ctx.loop_detector.check(None, screen.current_phash)
            ctx.steps.open(step + 1)
            used_cache = saved_path.try_cached_step(
                agent, screen, self.device, ctx, settling, Path("unused"), step
            )
            self.assertTrue(used_cache)

        self.assertFalse(ctx.loop_detector.terminated())


if __name__ == "__main__":
    unittest.main()
