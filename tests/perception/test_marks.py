"""Numbering what is on the screen, and collapsing what is not a target."""

import unittest

# A phone screenshot. Collapsing reads a container's size against the screen,
# so a fixture without one treats every box as a full-page region.
SCREEN_AREA = 1080 * 2400


class MarkTests(unittest.TestCase):
    """The number drawn on the screenshot and the number in the prompt must agree."""

    def _el(self, name, text, box):
        from yewo.agent.perception.elements import BoundingBox, GUIElement

        return GUIElement(
            element=name,
            type="yolo_detection",
            confidence=0.9,
            text=text,
            bounding_box=BoundingBox(x_min=box[0], y_min=box[1], x_max=box[2], y_max=box[3]),
        )

    def _screen(self):
        return [
            self._el("Button", "Continue", (480, 2180, 600, 2220)),
            self._el("Text", "Credit / Debit Card", (380, 660, 500, 700)),
            self._el("View", "", (500, 2190, 560, 2210)),
        ]

    def test_numbers_run_in_reading_order(self):
        from yewo.agent.grounding.marks import build

        numbered = [(m.index, m.element.text) for m in build(self._screen())]

        self.assertEqual(numbered[0], (1, "Credit / Debit Card"))
        self.assertEqual(numbered[-1][1], "Continue")

    def test_repeat_detections_get_one_number(self):
        from yewo.agent.grounding.marks import build

        doubled = self._screen() + [self._el("Text", "Credit / Debit Card", (381, 661, 501, 701))]

        self.assertEqual(len(build(doubled)), len(build(self._screen())))

    def test_a_label_found_twice_leaves_its_button_one_number(self):
        """A nested repeat overlaps unevenly, so an even-overlap test keeps it.

        Two labels inside one box read as a row, and a row keeps its parts
        numbered separately. The button then offers three numbers, two of
        which point at text rather than at the control.
        """
        from yewo.agent.grounding.marks import build

        button = [
            self._el("View", "", (361, 1751, 508, 1830)),
            self._el("Text", "Clear", (386, 1770, 482, 1813)),
            self._el("Text", "Clear", (385, 1765, 499, 1816)),
        ]

        numbered = build(button, screen_area=SCREEN_AREA)

        self.assertEqual(len(numbered), 1)
        self.assertEqual(numbered[0].element.text, "Clear")

    def test_a_rule_found_twice_along_its_length_gets_one_number(self):
        """A shift along a long edge costs an even-overlap test most of its score.

        Both boxes cover nearly all of the shorter one, and they are the same
        size, so they are one divider rather than a divider inside something.
        """
        from yewo.agent.grounding.marks import build

        shifted = [
            self._el("Line", "", (148, 455, 834, 466)),
            self._el("Line", "", (259, 456, 938, 466)),
        ]

        self.assertEqual(len(build(shifted, screen_area=SCREEN_AREA)), 1)

    def test_two_separate_controls_of_one_kind_keep_their_numbers(self):
        """Side by side, they share no area, whatever their class and size."""
        from yewo.agent.grounding.marks import build

        side_by_side = [
            self._el("View", "", (100, 100, 180, 180)),
            self._el("View", "", (200, 100, 280, 180)),
        ]

        self.assertEqual(len(build(side_by_side, screen_area=SCREEN_AREA)), 2)

    def test_an_untexted_box_inside_another_is_folded_not_dropped(self):
        """Dropping one of the pair loses whichever of the two it keeps.

        A container and the part inside it are folded by `_collapse_nests`
        instead, which keeps the container's outline and takes only the touch
        point from the part.
        """
        from yewo.agent.grounding.marks import build

        nested = [
            self._el("View", "", (100, 100, 900, 900)),
            self._el("Line", "", (120, 120, 880, 130)),
        ]

        numbered = build(nested, screen_area=SCREEN_AREA)

        self.assertEqual(len(numbered), 1)
        self.assertEqual(numbered[0].element.bounding_box.x_max, 900)

    def test_the_offered_list_carries_no_coordinates(self):
        """A coordinate here is one the model can echo back as if it saw it."""
        from yewo.agent.grounding.marks import build, describe

        listing = describe(build(self._screen()), height=2400)

        self.assertNotIn("2201", listing)
        self.assertNotIn("center", listing)
        self.assertIn("lower", listing)

    def test_the_detected_text_is_never_listed(self):
        """The model is looking at the screen those words are printed on."""
        from yewo.agent.grounding.marks import build, describe

        listing = describe(build(self._screen()), height=2400)

        self.assertNotIn("Continue", listing)
        self.assertNotIn("Credit", listing)

    def test_every_mark_is_listed(self):
        """The planner may answer only with a number from this list."""
        from yewo.agent.grounding.marks import build, describe

        marks = build(self._screen())
        listing = describe(marks, height=2400)

        for mark in marks:
            self.assertIn(f"{mark.index} |", listing)

    def test_a_number_resolves_to_its_element(self):
        from yewo.agent.grounding.marks import build, find

        marks = build(self._screen())
        continue_mark = next(m for m in marks if m.element.text == "Continue")

        self.assertEqual(find(marks, continue_mark.index).center, (540, 2200))
        self.assertIsNone(find(marks, 99))

    def test_marks_mode_never_offers_coordinates_to_the_model(self):
        from yewo.agent.plan.planner import GROUNDING_MARKS, DevicePlanner

        planner = DevicePlanner.__new__(DevicePlanner)
        planner.grounding = GROUNDING_MARKS

        vocabulary = planner._action_vocabulary()

        self.assertIn("`mark` (int)", vocabulary)
        self.assertIn("target_not_listed", vocabulary)
        self.assertNotIn('"x": int', vocabulary)


class KeyboardMarkTests(unittest.TestCase):
    """An open keyboard is most of the screen, and numbering it buries the app."""

    def _el(self, text, y, *roles):
        from yewo.agent.perception.elements import BoundingBox, GUIElement

        return GUIElement(
            element="Text",
            type="yolo_detection",
            confidence=0.9,
            text=text,
            semantic_roles=frozenset(roles),
            bounding_box=BoundingBox(x_min=10, y_min=y, x_max=60, y_max=y + 40),
        )

    def _screen(self):
        return [
            self._el("Email Address", 800),
            self._el("Login", 1300),
            *[self._el(letter, 1800, "keyboard_key") for letter in "qwertyuiop"],
            self._el("?!", 2200, "keyboard_key"),
        ]

    def test_keys_are_not_numbered_while_the_keyboard_is_open(self):
        from yewo.agent.grounding.marks import build
        from yewo.agent.perception.keyboard import without_keyboard_keys

        kept = without_keyboard_keys(self._screen(), keyboard_open=True, screen_height=2400)
        texts = [m.element.text for m in build(kept)]

        self.assertEqual(texts, ["Email Address", "Login"])

    def test_visual_semantics_override_a_stale_native_closed_state(self):
        from yewo.agent.grounding.marks import build
        from yewo.agent.perception.keyboard import without_keyboard_keys

        kept = without_keyboard_keys(self._screen(), keyboard_open=False, screen_height=2400)

        self.assertEqual([mark.element.text for mark in build(kept)], ["Email Address", "Login"])

    def test_a_long_label_low_on_screen_is_not_mistaken_for_a_key(self):
        from yewo.agent.perception.keyboard import without_keyboard_keys

        elements = [self._el("Place Order", 2200)]

        self.assertEqual(
            len(without_keyboard_keys(elements, keyboard_open=True, screen_height=2400)), 1
        )


class UnresolvedMarkTests(unittest.TestCase):
    """A number naming nothing must be reportable, not fatal."""

    def _action(self):
        from yewo.agent.actuation.actions import TapMarkAction

        return TapMarkAction(mark=47, target="Continue button", reason="continue")

    def test_an_unresolved_mark_can_be_described(self):
        from yewo.agent.step import narration

        self.assertIn("element 47", narration.describe(self._action()))

    def test_an_unresolved_mark_can_be_hashed_for_loop_detection(self):
        from yewo.agent.actuation.actions import TapMarkAction
        from yewo.agent.memory.loop_detector import RollingLoopDetector

        same = RollingLoopDetector.hash_action(self._action())
        other = RollingLoopDetector.hash_action(
            TapMarkAction(mark=48, target="Continue button", reason="continue")
        )

        self.assertEqual(same, RollingLoopDetector.hash_action(self._action()))
        self.assertNotEqual(same, other)

    def test_a_not_listed_action_can_be_described_and_hashed(self):
        from yewo.agent.step import narration
        from yewo.agent.actuation.actions import TargetNotListedAction
        from yewo.agent.memory.loop_detector import RollingLoopDetector

        action = TargetNotListedAction(
            description="blue button", region="lower_center", reason="r"
        )

        self.assertIn("target_not_listed", narration.describe(action))
        self.assertTrue(RollingLoopDetector.hash_action(action))


class NestCollapseTests(unittest.TestCase):
    """A field is a container holding an icon and a value — one control, not three."""

    def _el(self, kind, text, box):
        from yewo.agent.perception.elements import BoundingBox, GUIElement

        return GUIElement(
            element=kind,
            type="yolo_detection",
            confidence=0.9,
            text=text,
            bounding_box=BoundingBox(x_min=box[0], y_min=box[1], x_max=box[2], y_max=box[3]),
        )

    def _field(self):
        return [
            self._el("View", "", (60, 800, 1020, 970)),  # the field container
            self._el("ImageView", "", (80, 860, 140, 920)),  # the icon
            self._el("Text", "qa@example.com", (160, 860, 700, 910)),  # the value
        ]

    def test_a_field_becomes_one_mark(self):
        from yewo.agent.grounding.marks import build

        marks = build(self._field(), screen_area=1080 * 2400)

        self.assertEqual(len(marks), 1)

    def test_the_collapsed_mark_keeps_the_name(self):
        from yewo.agent.grounding.marks import build

        self.assertEqual(
            build(self._field(), screen_area=1080 * 2400)[0].element.text,
            "qa@example.com",
        )

    def test_the_collapsed_mark_is_touched_where_its_label_is(self):
        from yewo.agent.grounding.marks import build

        mark = build(self._field(), screen_area=1080 * 2400)[0]

        self.assertEqual(mark.center, (430, 885))

    def test_a_layout_region_keeps_its_controls_separate(self):
        from yewo.agent.grounding.marks import build

        page = [self._el("View", "", (0, 0, 1080, 2400)), *self._field()]

        self.assertGreater(len(build(page, screen_area=1080 * 2400)), 1)

    def test_a_toolbar_wrapper_uses_the_inner_named_control_box(self):
        """A broad bottom bar must not become the outline for its MOVE child."""
        from yewo.agent.grounding.marks import build

        elements = [
            self._el("View", "", (121, 2206, 1079, 2399)),
            self._el("View", "", (299, 2227, 534, 2320)),
            self._el("Text", "MOVE", (354, 2250, 483, 2298)),
            self._el("View", "", (38, 2218, 281, 2322)),
            self._el("Text", "CANCEL", (72, 2248, 243, 2300)),
        ]

        move = next(
            mark
            for mark in build(elements, screen_area=1080 * 2400)
            if mark.element.text == "MOVE"
        )

        self.assertEqual(
            tuple(move.element.bounding_box),
            (299, 2227, 534, 2320),
        )
        self.assertEqual(move.center, (418, 2274))


class ListRowTests(NestCollapseTests):
    """A row of separate controls is not one control, however neatly it is boxed."""

    def _card(self):
        return [
            self._el("View", "", (40, 600, 1040, 700)),  # the card
            self._el("Text", "Burger Palace", (160, 620, 500, 650)),
            self._el("Text", "Fast Food 1.2 km", (160, 655, 500, 685)),
            self._el("Text", "Menu", (820, 630, 960, 680)),  # the only live part
        ]

    def test_a_card_keeps_its_button_reachable(self):
        from yewo.agent.grounding.marks import build

        texts = [m.element.text for m in build(self._card(), screen_area=1080 * 2400)]

        self.assertIn("Menu", texts)
        self.assertIn("Burger Palace", texts)

    def test_a_single_value_wrapper_still_collapses(self):
        from yewo.agent.grounding.marks import build

        self.assertEqual(len(build(self._field(), screen_area=1080 * 2400)), 1)

    def test_a_container_with_its_own_label_is_left_alone(self):
        from yewo.agent.grounding.marks import build

        labelled = [
            self._el("View", "Total", (40, 600, 1040, 700)),
            self._el("Text", "£35.00", (820, 630, 960, 680)),
        ]

        self.assertEqual(len(build(labelled, screen_area=1080 * 2400)), 2)


class KeyboardBandTests(unittest.TestCase):
    """The strip with no numbers has to say why it has no numbers."""

    def _key(self, text, y):
        from yewo.agent.perception.elements import BoundingBox, GUIElement

        return GUIElement(
            element="Text",
            type="yolo_detection",
            confidence=0.9,
            text=text,
            semantic_roles=frozenset({"keyboard_key"}),
            bounding_box=BoundingBox(x_min=10, y_min=y, x_max=60, y_max=y + 40),
        )

    def _app_control(self, text, y):
        return self._key(text, y).model_copy(update={"semantic_roles": frozenset()})

    def test_the_band_spans_the_keys(self):
        from yewo.agent.perception.keyboard import keyboard_band

        band = keyboard_band([self._key("q", 1800), self._key("m", 2200)], 1080)

        self.assertEqual(band, (10, 1800, 60, 2240))

    def test_there_is_no_band_without_a_keyboard(self):
        from yewo.agent.perception.keyboard import keyboard_band

        self.assertIsNone(keyboard_band([], 1080))

    def test_a_wide_key_inside_the_band_is_absorbed(self):
        """Semantic keys do not depend on label length."""
        from yewo.agent.perception.keyboard import split_keyboard

        elements = [
            self._key("q", 1700),
            self._key("?123", 1950),
            self._key("m", 2200),
        ]

        kept, keys = split_keyboard(elements, keyboard_open=True, screen_height=2400)

        self.assertEqual(kept, [])
        self.assertEqual(len(keys), 3)

    def test_an_app_control_above_the_keyboard_survives(self):
        from yewo.agent.perception.keyboard import split_keyboard

        elements = [
            self._app_control("Place Order", 1400),
            self._key("q", 1700),
            self._key("m", 2200),
        ]

        kept, keys = split_keyboard(elements, keyboard_open=True, screen_height=2400)

        self.assertEqual([e.text for e in kept], ["Place Order"])
        self.assertEqual(len(keys), 2)
