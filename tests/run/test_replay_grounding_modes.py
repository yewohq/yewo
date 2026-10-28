"""Durable replay uses the evidence available in each grounding mode."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from yewo.agent.actuation.actions import (
    Coordinates,
    TapAction,
    TapDescribeAction,
    TapMarkAction,
)
from yewo.agent.grounding import (
    GROUNDING_COORDINATES,
    GROUNDING_DESCRIBE,
    GROUNDING_MARKS,
)
from yewo.agent.grounding.marks import Mark
from yewo.agent.grounding.placement import Placement
from yewo.agent.memory import replay_store, replay_walk
from yewo.agent.perception.elements import BoundingBox, GUIElement, ScanInfo
from yewo.agent.perception.blind import BlindPerception, for_grounding
from yewo.agent.step import targeting


class ReplayPerceptionTests(unittest.TestCase):
    def test_only_marks_uses_detected_elements_during_replay(self):
        detected = object()
        session = SimpleNamespace()

        with patch(
            "yewo.agent.perception.screen.ScreenPerception", return_value=detected
        ) as screen_perception:
            for grounding in (
                GROUNDING_COORDINATES,
                GROUNDING_MARKS,
                GROUNDING_DESCRIBE,
            ):
                with self.subTest(grounding=grounding):
                    perception = for_grounding(grounding, session)
                    if grounding == GROUNDING_MARKS:
                        self.assertIs(perception, detected)
                    else:
                        self.assertIsInstance(perception, BlindPerception)
                        self.assertEqual(perception.mode, grounding)

        self.assertEqual(screen_perception.call_count, 1)

    def test_describe_without_replay_keeps_its_detector_free_path(self):
        perception = for_grounding(GROUNDING_DESCRIBE, SimpleNamespace())

        self.assertIsInstance(perception, BlindPerception)


class GroundedPathTests(unittest.TestCase):
    def test_each_mode_learns_the_same_semantic_target(self):
        target = GUIElement(
            element="Button",
            bounding_box=BoundingBox(x_min=100, y_min=200, x_max=300, y_max=300),
            confidence=0.9,
            type="button",
            text="Continue",
        )
        before = ScanInfo(
            mode="detect",
            reason="step",
            phash="before",
            screen_width=1000,
            screen_height=2000,
            element_count=1,
        )
        after = ScanInfo(
            mode="detect",
            reason="step",
            phash="after",
            screen_width=1000,
            screen_height=2000,
            element_count=1,
        )
        snapshot = SimpleNamespace(
            marks=[Mark(index=1, element=target)],
            raw_img_bytes=b"screen",
            gui_elements=[target],
        )
        device = SimpleNamespace(get_screenshot=lambda: b"screen")
        grounder = SimpleNamespace(
            ground=lambda _image, _description: Placement(
                box=target.bounding_box, present=1.0, box_confidence=1.0
            )
        )
        actions = {
            GROUNDING_COORDINATES: TapAction(
                coordinates=Coordinates(x=200, y=250),
                target="Continue button",
                reason="continue",
            ),
            GROUNDING_MARKS: TapMarkAction(mark=1, target="Continue button", reason="continue"),
            GROUNDING_DESCRIBE: TapDescribeAction(target="Continue button", reason="continue"),
        }

        with patch("yewo.agent.step.targeting.stale_target_reason", return_value=None):
            for grounding, proposed in actions.items():
                with self.subTest(grounding=grounding):
                    decision = targeting.evaluate(proposed, snapshot, device, grounding, grounder)
                    self.assertTrue(decision.ok)
                    if grounding == GROUNDING_DESCRIBE:
                        self.assertEqual(decision.target_bounds, target.bounding_box)
                    trace = replay_walk.build_trace(
                        replay_store.TraceIdentity(
                            scenario_uuid="checkout",
                            scenario_digest="digest",
                            platform="android",
                            target_profile="Pixel",
                            detector="detector@revision",
                            perception_pipeline="test-pipeline",
                            grounding=grounding,
                        ),
                        [decision.action or proposed],
                        [(([target], before), ([target], after))],
                        replay_store.ValueNames(),
                        key=b"mode test key",
                        target_descriptions=["Continue button"],
                        screen_descriptions=["Welcome screen", "Next screen"],
                    )
                    self.assertIsNotNone(trace)
                    transition = trace.transitions[0]
                    self.assertEqual(transition.action.described, "Continue button")
                    self.assertEqual(transition.source.description, "Welcome screen")
                    self.assertEqual(transition.destination.description, "Next screen")

    def test_detector_free_modes_learn_without_elements(self):
        before = ScanInfo(
            mode="coordinates",
            reason="step",
            phash="before",
            screen_width=1000,
            screen_height=2000,
            element_count=0,
        )
        after = ScanInfo(
            mode="coordinates",
            reason="step",
            phash="after",
            screen_width=1000,
            screen_height=2000,
            element_count=0,
        )
        action = TapAction(
            coordinates=Coordinates(x=200, y=250),
            target="Continue button",
            reason="continue",
        )

        for grounding in (GROUNDING_COORDINATES, GROUNDING_DESCRIBE):
            with self.subTest(grounding=grounding):
                trace = replay_walk.build_trace(
                    replay_store.TraceIdentity(
                        scenario_uuid="checkout",
                        scenario_digest="digest",
                        platform="android",
                        target_profile="Pixel",
                        detector="",
                        perception_pipeline="test-pipeline",
                        grounding=grounding,
                    ),
                    [action],
                    [(([], before), ([], after))],
                    replay_store.ValueNames(),
                    target_descriptions=["Continue button"],
                    screen_descriptions=["Welcome screen", "Next screen"],
                )

                self.assertIsNotNone(trace)
                transition = trace.transitions[0]
                self.assertFalse(transition.source.anchors)
                self.assertEqual(transition.action.described, "Continue button")


if __name__ == "__main__":
    unittest.main()
