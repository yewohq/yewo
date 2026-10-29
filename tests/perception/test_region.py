"""Bounding a missing control with the controls that were found around it."""

import unittest

WIDTH, HEIGHT = 1080, 2400


def _el(box, text=""):
    from yewo.agent.perception.elements import BoundingBox, GUIElement

    return GUIElement(
        element="Text",
        type="yolo_detection",
        confidence=0.9,
        text=text,
        bounding_box=BoundingBox(x_min=box[0], y_min=box[1], x_max=box[2], y_max=box[3]),
    )


def _form():
    """A trainer form whose repetitions field was never detected."""
    return [
        _el((42, 900, 420, 940), "Exercise:"),
        _el((42, 950, 1038, 1010)),
        _el((42, 1100, 480, 1162), "Repeations:"),
        # Nothing between here and the next label. That gap is the target.
        _el((42, 1280, 420, 1320), "Pause [s]:"),
        _el((42, 1340, 1038, 1400)),
    ]


class DeriveTests(unittest.TestCase):
    def test_gap_between_two_rows_becomes_the_region(self):
        from yewo.agent.grounding.region import derive

        derived = derive("middle_left", _form(), (WIDTH, HEIGHT))

        self.assertIsNotNone(derived)
        self.assertEqual(derived.anchor_above[3], 1162)
        self.assertEqual(derived.anchor_below[1], 1280)

    def test_width_comes_from_the_sibling_fields(self):
        from yewo.agent.grounding.region import SIBLING_SPAN, derive

        derived = derive("middle_left", _form(), (WIDTH, HEIGHT))

        self.assertEqual(derived.width_rule, SIBLING_SPAN)
        self.assertEqual((derived.crop[0], derived.crop[2]), (36, 1044))

    def test_the_region_contains_the_point_the_control_sits_at(self):
        from yewo.agent.grounding.region import derive

        crop = derive("middle_left", _form(), (WIDTH, HEIGHT)).crop

        self.assertLess(crop[0], 540)
        self.assertGreater(crop[2], 540)
        self.assertLess(crop[1], 1221)
        self.assertGreater(crop[3], 1221)

    def test_the_region_is_a_fraction_of_the_third_it_replaces(self):
        from yewo.agent.grounding.region import derive

        crop = derive("middle_left", _form(), (WIDTH, HEIGHT)).crop
        height = crop[3] - crop[1]

        self.assertLess(height, HEIGHT // 3)

    def test_one_detection_cannot_bound_anything(self):
        from yewo.agent.grounding.region import derive

        self.assertIsNone(derive("middle_left", [_el((42, 500, 480, 562))], (WIDTH, HEIGHT)))

    def test_a_gap_open_at_one_end_is_refused(self):
        """The band's own edge is a screen fraction, which is what this avoids."""
        from yewo.agent.grounding.region import derive

        one_anchor = [_el((42, 200, 1038, 240)), _el((42, 1000, 1038, 1040))]

        self.assertIsNone(derive("middle_left", one_anchor, (WIDTH, HEIGHT)))

    def test_spacing_between_controls_is_not_a_missing_control(self):
        from yewo.agent.grounding.region import derive

        tight = [
            _el((42, 1000, 1038, 1060)),
            _el((42, 1070, 1038, 1130)),
            _el((42, 1140, 1038, 1200)),
        ]

        self.assertIsNone(derive("middle_left", tight, (WIDTH, HEIGHT)))

    def test_empty_screen_is_not_a_region(self):
        from yewo.agent.grounding.region import derive

        wide_apart = [_el((42, 820, 1038, 860)), _el((42, 1760, 1038, 1800))]

        self.assertIsNone(derive("middle_left", wide_apart, (WIDTH, HEIGHT)))

    def test_detections_on_one_line_do_not_open_a_gap_against_each_other(self):
        from yewo.agent.grounding.region import derive

        split_row = _form() + [_el((500, 1105, 700, 1160), "3")]

        derived = derive("middle_left", split_row, (WIDTH, HEIGHT))

        self.assertEqual(derived.anchor_above[3], 1162)

    def test_the_hint_chooses_between_gaps_without_becoming_an_edge(self):
        from yewo.agent.grounding.region import derive

        rows = [
            _el((42, 200, 1038, 240)),
            _el((42, 500, 1038, 540)),
            _el((42, 1000, 1038, 1040)),
            _el((42, 1300, 1038, 1340)),
            _el((42, 1900, 1038, 1940)),
            _el((42, 2200, 1038, 2240)),
        ]

        upper = derive("upper_left", rows, (WIDTH, HEIGHT))
        middle = derive("middle_left", rows, (WIDTH, HEIGHT))
        lower = derive("lower_left", rows, (WIDTH, HEIGHT))

        self.assertEqual(upper.anchor_above[3], 240)
        self.assertEqual(middle.anchor_above[3], 1040)
        self.assertEqual(lower.anchor_above[3], 1940)

    def test_anchors_bound_the_width_when_no_sibling_field_was_found(self):
        from yewo.agent.grounding.region import ANCHOR_SPAN, derive

        labels_only = [
            _el((300, 1100, 700, 1162), "Repeations:"),
            _el((320, 1280, 680, 1320), "Pause [s]:"),
        ]

        derived = derive("middle_left", labels_only, (WIDTH, HEIGHT))

        self.assertEqual(derived.width_rule, ANCHOR_SPAN)
        self.assertEqual((derived.crop[0], derived.crop[2]), (294, 706))

    def test_an_unknown_region_name_derives_nothing(self):
        from yewo.agent.grounding.region import derive

        self.assertIsNone(derive("middle", _form(), (WIDTH, HEIGHT)))

    def test_named_anchors_choose_the_requested_gap_when_one_band_has_several(self):
        from yewo.agent.grounding.region import derive_between

        elements = [
            _el((42, 900, 1038, 950)),
            _el((42, 1100, 1038, 1150)),
            _el((42, 1300, 1038, 1350)),
            _el((42, 1500, 1038, 1550)),
        ]

        derived = derive_between(
            (42, 1300, 1038, 1350),
            (42, 1500, 1038, 1550),
            elements,
            (WIDTH, HEIGHT),
        )

        self.assertEqual(derived.anchor_above, (42, 1300, 1038, 1350))
        self.assertEqual(derived.anchor_below, (42, 1500, 1038, 1550))

    def test_named_anchor_derivation_requires_both_surrounding_controls(self):
        from yewo.agent.grounding.region import derive_between

        self.assertIsNone(
            derive_between((42, 1100, 480, 1162), None, _form(), (WIDTH, HEIGHT))
        )


class WidthTests(unittest.TestCase):
    """Where the left and right edges come from when the siblings vary."""

    def _gap(self, extra):
        from yewo.agent.grounding.region import derive

        anchors = [
            _el((300, 1100, 700, 1162), "Repeations:"),
            _el((320, 1280, 680, 1320), "Pause [s]:"),
        ]
        return derive("middle_left", anchors + extra, (WIDTH, HEIGHT))

    def test_an_odd_number_of_siblings_takes_the_middle_measurement(self):
        from yewo.agent.grounding.region import SIBLING_SPAN

        siblings = [
            _el((40, 400, 1030, 460)),
            _el((42, 1340, 1038, 1400)),
            _el((44, 1500, 1040, 1560)),
        ]

        derived = self._gap(siblings)

        self.assertEqual(derived.width_rule, SIBLING_SPAN)
        self.assertEqual(derived.width_samples, 3)
        self.assertEqual((derived.crop[0], derived.crop[2]), (36, 1044))

    def test_one_sibling_field_is_enough_to_measure_from(self):
        from yewo.agent.grounding.region import SINGLE_SIBLING

        derived = self._gap([_el((42, 1340, 1038, 1400))])

        self.assertEqual(derived.width_rule, SINGLE_SIBLING)
        self.assertEqual(derived.width_samples, 1)
        self.assertEqual((derived.crop[0], derived.crop[2]), (36, 1044))

    def test_a_region_too_narrow_to_hold_a_control_widens_to_its_anchors(self):
        from yewo.agent.grounding.region import ANCHOR_SPAN, derive

        narrow = [
            _el((500, 1100, 540, 1162)),
            _el((900, 1280, 1000, 1320)),
        ]

        derived = derive("middle_left", narrow, (WIDTH, HEIGHT))

        self.assertEqual(derived.width_rule, ANCHOR_SPAN)
        self.assertEqual((derived.crop[0], derived.crop[2]), (494, 1006))

    def test_anchors_with_no_width_derive_no_region(self):
        from yewo.agent.grounding.region import derive

        empty = [_el((0, 1100, 0, 1162)), _el((0, 1280, 0, 1320))]

        self.assertIsNone(derive("middle_left", empty, (WIDTH, HEIGHT)))

    def test_a_band_with_nothing_in_it_derives_no_region(self):
        from yewo.agent.grounding.region import derive

        outside = [_el((42, 100, 1038, 160)), _el((42, 200, 1038, 260))]

        self.assertIsNone(derive("middle_left", outside, (WIDTH, HEIGHT)))

    def test_a_screen_with_no_size_derives_no_region(self):
        from yewo.agent.grounding.region import derive

        self.assertIsNone(derive("middle_left", _form(), (0, 0)))


class ScaleTests(unittest.TestCase):
    def test_a_thin_band_is_magnified_more_than_a_third_of_screen(self):
        from yewo.agent.grounding.region import scale_for

        band = scale_for((36, 1156, 1044, 1286))
        third = scale_for((0, 600, 1080, 1400))

        self.assertGreater(band, third)

    def test_magnification_stays_inside_the_bounds_detection_was_tuned_for(self):
        from yewo.agent.grounding.region import MAX_SCALE, MIN_SCALE, scale_for

        for crop in ((0, 0, 4, 4), (0, 0, 1080, 2400), (0, 0, 1080, 60)):
            with self.subTest(crop=crop):
                self.assertGreaterEqual(scale_for(crop), MIN_SCALE)
                self.assertLessEqual(scale_for(crop), MAX_SCALE)

    def test_an_empty_crop_cannot_divide_by_zero(self):
        from yewo.agent.grounding.region import MIN_SCALE, scale_for

        self.assertEqual(scale_for((10, 10, 10, 10)), MIN_SCALE)
