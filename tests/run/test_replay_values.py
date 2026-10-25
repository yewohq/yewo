"""Replay persistence when a journey declares few or no values."""

import unittest

from yewo.agent.actuation.actions import Coordinates, TapAction, TypeAction
from yewo.agent.memory import replay_store, replay_walk
from yewo.agent.perception.elements import BoundingBox, GUIElement, ScanInfo

WIDTH, HEIGHT = 1000, 2000


def element(text, role="button"):
    return GUIElement(
        element=text or role,
        bounding_box=BoundingBox(x_min=100, y_min=100, x_max=300, y_max=200),
        confidence=0.9,
        type=role,
        text=text,
    )


def scan(count):
    return ScanInfo(
        mode="detect",
        reason="step",
        phash="p",
        screen_width=WIDTH,
        screen_height=HEIGHT,
        element_count=count,
    )


def centre_of(target):
    box = target.bounding_box
    return Coordinates(
        x=(box.x_min + box.x_max) // 2,
        y=(box.y_min + box.y_max) // 2,
    )


def identity():
    return replay_store.TraceIdentity(
        scenario_uuid="s",
        scenario_digest="d",
        platform="android",
        target_profile="Pixel_8",
        detector="d@abc",
        grounding="marks",
    )


class AppsWithoutValuesTests(unittest.TestCase):
    def _trace(self, action, before, after, values=None):
        return replay_walk.build_trace(
            identity(),
            [action],
            [((before, scan(len(before))), (after, scan(len(after))))],
            replay_store.ValueNames(values or {}),
        )

    def test_a_tap_caches_with_no_declared_values_at_all(self):
        add = element("Add to basket")

        trace = self._trace(
            TapAction(coordinates=centre_of(add), reason="r"),
            [add],
            [element("Basket")],
        )

        self.assertIsNotNone(trace)
        self.assertEqual(len(trace.transitions), 1)

    def test_typing_text_the_journey_never_named_does_not_cache(self):
        search = element("Search", "field")

        trace = self._trace(
            TypeAction(text="pizza", coordinates=centre_of(search), reason="r"),
            [search],
            [element("Results")],
        )

        self.assertIsNone(trace)

    def test_naming_the_value_makes_typing_cacheable(self):
        search = element("Search", "field")

        trace = self._trace(
            TypeAction(text="pizza", coordinates=centre_of(search), reason="r"),
            [search],
            [element("Results")],
            values={"search_term": "pizza"},
        )

        self.assertIsNotNone(trace)
        self.assertEqual(trace.transitions[0].action.value_name, "search_term")
        postcondition = trace.transitions[0].action.postconditions[0]
        self.assertEqual(postcondition.value_name, "search_term")
        self.assertNotIn("pizza", postcondition.model_dump_json())

    def test_an_explicit_mask_policy_is_saved_as_a_specification_not_evidence(self):
        password = element("Password", "field")

        trace = replay_walk.build_trace(
            identity(),
            [TypeAction(text="correct horse", coordinates=centre_of(password), reason="r")],
            [
                (
                    ([password], scan(1)),
                    ([element("••••••••••••", "field")], scan(1)),
                )
            ],
            replay_store.ValueNames({"credential": "correct horse"}),
            verification_policies={"credential": "masked"},
        )

        self.assertIsNotNone(trace)
        postcondition = trace.transitions[0].action.postconditions[0]
        self.assertEqual(postcondition.comparison, "masked")
        self.assertNotIn("correct horse", trace.model_dump_json())

    def test_undeclared_screen_text_is_digested(self):
        anchors = replay_store.anchors_of(
            [element("Order #4471 confirmed", "text")],
            scan(1),
            replay_store.ValueNames({}),
        )

        self.assertNotIn("4471", anchors[0].text)
        self.assertNotIn("Order", anchors[0].text)
