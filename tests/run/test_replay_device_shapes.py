"""Durable replay behavior derived from real detector output."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from yewo.agent.actuation.actions import (
    Coordinates,
    KeyEventAction,
    TapAction,
    TypeAction,
    WaitAction,
)
import replay_driver
from yewo.agent.memory import (
    replay_builder,
    replay_learning,
    replay_match,
    replay_store,
    replay_walk,
)
from yewo.agent.perception.elements import BoundingBox, GUIElement, ScanInfo

WIDTH, HEIGHT = 1080, 2400
TEST_KEY = b"real detector fixture digest key"


def element(text, box):
    return GUIElement(
        element=text or "View",
        bounding_box=BoundingBox(
            x_min=box[0], y_min=box[1], x_max=box[2], y_max=box[3]
        ),
        confidence=0.9,
        type="yolo_detection",
        text=text,
    )


def scan(count, phash="screen"):
    return ScanInfo(
        mode="full",
        reason="first_pass",
        phash=phash,
        screen_width=WIDTH,
        screen_height=HEIGHT,
        element_count=count,
    )


def identity():
    return replay_store.TraceIdentity(
        scenario_uuid="checkout",
        scenario_digest="digest",
        platform="android",
        target_profile="sdk_gphone64_arm64",
        detector="orasul/deki-yolo@revision",
        grounding="coordinates",
    )


def build_trace(action, before, after):
    return replay_walk.build_trace(
        identity(),
        [action],
        [
            (
                (before, scan(len(before))),
                (after, scan(len(after))),
            )
        ],
        replay_store.ValueNames(),
        key=TEST_KEY,
    )


class NestedDetectionTests(unittest.TestCase):
    def test_the_innermost_named_box_owns_the_tap(self):
        container = element("Account form", (50, 900, 950, 1500))
        control = element("Sign in", (300, 1100, 700, 1300))
        label = element("Sign in", (420, 1160, 580, 1240))

        trace = build_trace(
            TapAction(coordinates=Coordinates(x=500, y=1200), reason="sign in"),
            [container, control, label],
            [element("Orders", (100, 300, 500, 500))],
        )

        self.assertEqual(
            trace.transitions[0].action.target.bounds,
            replay_store.relative_bounds(label.bounding_box, WIDTH, HEIGHT),
        )

    def test_an_unnamed_box_cannot_authorize_a_tap(self):
        icon = element("", (420, 1160, 580, 1240))

        with self.assertLogs(replay_builder.logger, level="INFO") as messages:
            trace = build_trace(
                TapAction(coordinates=Coordinates(x=500, y=1200), reason="open"),
                [icon],
                [element("Orders", (100, 300, 500, 500))],
            )

        self.assertIsNone(trace)
        self.assertIn("screen has no named anchors", " ".join(messages.output))

    def test_a_named_container_cannot_stand_in_for_an_unnamed_inner_control(self):
        container = element("Account form", (50, 900, 950, 1500))
        control = element("", (420, 1160, 580, 1240))

        trace = build_trace(
            TapAction(coordinates=Coordinates(x=500, y=1200), reason="open"),
            [container, control],
            [element("Orders", (100, 300, 500, 500))],
        )

        self.assertIsNone(trace)

    def test_an_empty_field_is_anchored_to_its_label(self):
        field = element("", (61, 1187, 1014, 1334))
        label = element("Email Address", (190, 1233, 495, 1291))
        email = "qa.tester@example.com"
        trace = replay_walk.build_trace(
            identity(),
            [
                TypeAction(
                    text=email,
                    coordinates=Coordinates(x=537, y=1260),
                    reason="enter email",
                )
            ],
            [
                (
                    ([field, label], scan(2)),
                    ([element("Password", (191, 1423, 404, 1479))], scan(1)),
                )
            ],
            replay_store.ValueNames({"email": email}),
            key=TEST_KEY,
        )

        target = trace.transitions[0].action.target
        self.assertEqual(target.text, "")
        self.assertEqual(
            target.context.text,
            replay_store.text_identity(
                "Email Address", replay_store.ValueNames(), TEST_KEY
            ),
        )

    def test_a_contextual_field_tracks_its_label_after_both_move(self):
        field = element("", (61, 1187, 1014, 1334))
        label = element("Email Address", (190, 1233, 495, 1291))
        anchors = replay_store.anchors_of(
            [field, label], scan(2), replay_store.ValueNames(), TEST_KEY
        )
        target = replay_walk._resolve_target_anchor(
            TapAction(coordinates=Coordinates(x=537, y=1260), reason="focus"),
            anchors,
            scan(2),
        )
        moved_field = element("", (81, 1197, 1034, 1344))
        moved_label = element("Email Address", (210, 1243, 515, 1301))

        match, refusal = replay_match.find_target(
            target,
            [moved_field, moved_label],
            scan(2),
            replay_store.ValueNames(),
            TEST_KEY,
        )

        self.assertIsNone(refusal)
        self.assertEqual(match.element.bounding_box, moved_field.bounding_box)

    def test_a_contextual_field_is_refused_when_its_label_disappears(self):
        field = element("", (61, 1187, 1014, 1334))
        label = element("Email Address", (190, 1233, 495, 1291))
        anchors = replay_store.anchors_of(
            [field, label], scan(2), replay_store.ValueNames(), TEST_KEY
        )
        target = replay_walk._resolve_target_anchor(
            TapAction(coordinates=Coordinates(x=537, y=1260), reason="focus"),
            anchors,
            scan(2),
        )

        match, refusal = replay_match.find_target(
            target,
            [field],
            scan(1),
            replay_store.ValueNames(),
            TEST_KEY,
        )

        self.assertIsNone(match)
        self.assertIn("context", refusal.reason)

    def test_equidistant_named_contexts_do_not_authorize_an_unnamed_target(self):
        field = element("", (400, 1100, 680, 1300))
        left_label = element("First", (200, 1150, 350, 1250))
        right_label = element("Second", (730, 1150, 880, 1250))

        trace = build_trace(
            TapAction(coordinates=Coordinates(x=540, y=1200), reason="focus"),
            [field, left_label, right_label],
            [element("Next", (100, 300, 500, 500))],
        )

        self.assertIsNone(trace)

    def test_equal_innermost_boxes_are_ambiguous(self):
        first = element("Continue", (300, 1100, 700, 1300))
        second = element("Submit", (300, 1100, 700, 1300))

        with self.assertLogs(replay_builder.logger, level="INFO") as messages:
            trace = build_trace(
                TapAction(coordinates=Coordinates(x=500, y=1200), reason="continue"),
                [first, second],
                [element("Orders", (100, 300, 500, 500))],
            )

        self.assertIsNone(trace)
        self.assertIn("no unique named target", " ".join(messages.output))


class TargetlessActionTests(unittest.TestCase):
    def _walk_wait(self, trace, source_phash):
        logo = element("", (402, 1035, 717, 1354))
        email = element("Email", (100, 900, 900, 1100))
        readings = [
            SimpleNamespace(
                gui_elements=[logo],
                scan_info=scan(1, phash=source_phash),
                current_phash=source_phash,
                raw_img_bytes=b"",
            ),
            SimpleNamespace(
                gui_elements=[email],
                scan_info=scan(1, phash="login"),
                current_phash="login",
                raw_img_bytes=b"",
            ),
        ]

        replay = replay_learning.DurableReplay(
            trace=trace, values={}, names=replay_store.ValueNames(), key=TEST_KEY
        )
        with (
            patch("yewo.agent.memory.replay_walk.ActionOrchestrator") as orchestrator,
            patch("yewo.agent.memory.replay_walk.time.sleep"),
        ):
            sources, ctx = replay_driver.drive(
                replay, object(), readings, settle_phash="login"
            )
        return (
            SimpleNamespace(sources=sources, is_whole=sources[0] == "cache"),
            orchestrator.return_value.execute,
        )

    def test_a_wait_can_start_on_an_unnamed_splash_screen(self):
        logo = element("", (402, 1035, 717, 1354))
        email = element("Email", (100, 900, 900, 1100))

        trace = replay_walk.build_trace(
            identity(),
            [WaitAction(duration=2.0, reason="loading")],
            [
                (
                    ([logo], scan(1, phash="splash")),
                    ([email], scan(1, phash="login")),
                )
            ],
            replay_store.ValueNames(),
            key=TEST_KEY,
        )

        self.assertEqual(trace.transitions[0].action.kind, "wait")
        self.assertEqual(trace.transitions[0].action.duration, 2.0)

        result, execute = self._walk_wait(trace, "splash")

        self.assertTrue(result.is_whole)
        self.assertIsInstance(execute.call_args.args[0], WaitAction)

    def test_a_changed_unnamed_screen_does_not_replay_the_wait(self):
        logo = element("", (402, 1035, 717, 1354))
        email = element("Email", (100, 900, 900, 1100))
        trace = replay_walk.build_trace(
            identity(),
            [WaitAction(duration=2.0, reason="loading")],
            [
                (
                    ([logo], scan(1, phash="splash")),
                    ([email], scan(1, phash="login")),
                )
            ],
            replay_store.ValueNames(),
            key=TEST_KEY,
        )

        result, execute = self._walk_wait(trace, "different")

        self.assertFalse(result.is_whole)
        execute.assert_not_called()

    def test_only_android_back_is_kept_as_a_key_event(self):
        field = element("Card number", (100, 900, 900, 1100))

        back_trace = build_trace(
            KeyEventAction(key=4, reason="hide keyboard"), [field], [field]
        )
        home_trace = build_trace(
            KeyEventAction(key=3, reason="go home"), [field], [field]
        )

        self.assertEqual(back_trace.transitions[0].action.key, 4)
        self.assertIsNone(home_trace)

    def test_android_back_replays_without_an_element_target(self):
        field = element("Card number", (100, 900, 900, 1100))
        trace = build_trace(
            KeyEventAction(key=4, reason="hide keyboard"), [field], [field]
        )
        readings = [
            SimpleNamespace(
                gui_elements=[field],
                scan_info=scan(1),
                current_phash="screen",
                raw_img_bytes=b"",
            ),
            SimpleNamespace(
                gui_elements=[field],
                scan_info=scan(1),
                current_phash="screen",
                raw_img_bytes=b"",
            ),
        ]

        replay = replay_learning.DurableReplay(
            trace=trace, values={}, names=replay_store.ValueNames(), key=TEST_KEY
        )
        with patch(
            "yewo.agent.memory.replay_walk.ActionOrchestrator"
        ) as orchestrator:
            sources, ctx = replay_driver.drive(
                replay, object(), readings, settle_phash="screen"
            )

        self.assertEqual(sources[0], "cache")
        self.assertIsInstance(
            orchestrator.return_value.execute.call_args.args[0], KeyEventAction
        )


if __name__ == "__main__":
    unittest.main()
