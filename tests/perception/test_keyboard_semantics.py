import unittest

from yewo.agent.perception.elements import BoundingBox, GUIElement
from yewo.agent.perception.keyboard import (
    annotate_keyboard_keys,
    observe,
    target_elements,
)


def _element(label, box, text="", roles=()):
    return GUIElement(
        element=label,
        type="yolo_detection",
        confidence=0.9,
        text=text,
        semantic_roles=frozenset(roles),
        bounding_box=BoundingBox(x_min=box[0], y_min=box[1], x_max=box[2], y_max=box[3]),
    )


class KeyboardSemanticsTests(unittest.TestCase):
    def test_region_owns_contained_detections_without_using_screen_position(self):
        region = _element("Surface", (300, 300, 800, 700), roles=("keyboard_region",))
        contained = _element("Text", (350, 350, 450, 400), text="space")
        app = _element("Text", (20, 900, 200, 960), text="Place order")

        result = observe([region, contained, app], native_visible=False)

        self.assertTrue(result.visual_visible)
        self.assertTrue(result.effective_visible)
        self.assertTrue(result.state_disagrees)
        self.assertEqual(result.owned_elements, (region, contained))
        self.assertEqual(result.app_elements, (app,))

    def test_native_visibility_alone_does_not_claim_app_elements(self):
        bottom_control = _element("Text", (10, 1800, 300, 1900), text="Continue")

        result = observe([bottom_control], native_visible=True)

        self.assertTrue(result.effective_visible)
        self.assertEqual(result.app_elements, (bottom_control,))
        self.assertEqual(result.owned_elements, ())

    def test_explicit_key_is_owned_without_a_region(self):
        key = _element("AnyLabel", (10, 10, 40, 40), roles=("keyboard_key",))

        result = observe([key])

        self.assertTrue(result.visual_visible)
        self.assertEqual(result.key_elements, (key,))

    def test_expansion_offers_explicit_keys_but_not_incidental_region_content(self):
        region = _element("Surface", (0, 0, 200, 200), roles=("keyboard_region",))
        key = _element("Key", (10, 10, 50, 50), text="A", roles=("keyboard_key",))
        label = _element("Text", (60, 10, 100, 50), text="suggestion")
        app = _element("Button", (0, 300, 100, 350), text="Submit")

        collapsed, _ = target_elements([region, key, label, app], True)
        expanded, _ = target_elements(
            [region, key, label, app], True, include_keyboard_targets=True
        )

        self.assertEqual(collapsed, [app])
        self.assertEqual([element.text for element in expanded], ["Submit", "[key] A"])

    def test_annotation_uses_semantics_and_preserves_inputs(self):
        region = _element("Surface", (0, 0, 200, 200), roles=("keyboard_region",))
        key = _element("Text", (10, 10, 30, 30), text="A")

        annotated = annotate_keyboard_keys([region, key], False, 2400)

        self.assertEqual([element.text for element in annotated], ["[keyboard]", "[key] A"])
        self.assertEqual([region.text, key.text], ["", "A"])


if __name__ == "__main__":
    unittest.main()
