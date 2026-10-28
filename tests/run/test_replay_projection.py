"""Replay uses the same actionable target projection as the planner."""

import unittest

from yewo.agent.actuation.actions import Coordinates, TapAction
from yewo.agent.memory import replay_store, replay_walk
from yewo.agent.perception.elements import BoundingBox, GUIElement, ScanInfo

WIDTH, HEIGHT = 1000, 2000


def element(text: str, box: tuple[int, int, int, int], *roles: str) -> GUIElement:
    left, top, right, bottom = box
    return GUIElement(
        element=text,
        bounding_box=BoundingBox(x_min=left, y_min=top, x_max=right, y_max=bottom),
        confidence=0.9,
        type="yolo_detection",
        text=text,
        semantic_roles=frozenset(roles),
    )


def scan(element_count: int, **updates) -> ScanInfo:
    return ScanInfo(
        mode="detect",
        reason="step",
        phash="screen",
        screen_width=WIDTH,
        screen_height=HEIGHT,
        element_count=element_count,
        **updates,
    )


def centre(target: GUIElement) -> Coordinates:
    box = target.bounding_box
    return Coordinates(x=(box.x_min + box.x_max) // 2, y=(box.y_min + box.y_max) // 2)


def identity() -> replay_store.TraceIdentity:
    return replay_store.TraceIdentity(
        scenario_uuid="scenario",
        scenario_digest="digest",
        platform="android",
        target_profile="Pixel_8",
        detector="detector@revision",
        perception_pipeline="test-pipeline",
        grounding="marks",
    )


class ReplayProjectionTests(unittest.TestCase):
    def setUp(self):
        self.names = replay_store.ValueNames()
        self.keyboard_region = element(
            "Keyboard", (0, 1200, 1000, 2000), "keyboard_region"
        )
        self.keyboard_key = element("A", (100, 1300, 200, 1400), "keyboard_key")

    def _learn(self, target, before_elements, before_scan):
        return replay_walk.build_trace(
            identity(),
            [TapAction(coordinates=centre(target), reason="test")],
            [
                (
                    (before_elements, before_scan),
                    ([element("Saved", (700, 100, 900, 200))], scan(1)),
                )
            ],
            self.names,
        )

    def test_keyboard_structure_is_not_learned_as_an_app_target(self):
        save = element("Save", (700, 100, 900, 200))
        trace = self._learn(
            save,
            [self.keyboard_region, self.keyboard_key, save],
            scan(3, keyboard_effective_visible=True),
        )

        source = trace.transitions[0].source
        self.assertEqual(len(source.anchors), 1)
        self.assertEqual(
            source.anchors[0].text,
            replay_store.text_identity("Save", self.names),
        )
        self.assertTrue(source.keyboard_visible)

    def test_explicit_keyboard_targets_can_be_learned(self):
        expanded_scan = scan(
            2,
            keyboard_effective_visible=True,
            keyboard_targets_expanded=True,
        )

        trace = self._learn(
            self.keyboard_key,
            [self.keyboard_region, self.keyboard_key],
            expanded_scan,
        )

        self.assertIsNotNone(trace)
        self.assertEqual(len(trace.transitions[0].source.anchors), 1)


if __name__ == "__main__":
    unittest.main()
