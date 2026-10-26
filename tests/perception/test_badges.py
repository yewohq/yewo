"""Where a number is drawn, and whether the element it names can be told."""

import unittest
from itertools import pairwise

from PIL import Image, ImageDraw

from yewo.agent.grounding import badges
from yewo.agent.perception.elements import BoundingBox

import fakes


SCREEN = (1080, 2400)
# A colour palette: 84px swatches with a 12px gap, three rows deep. The gap
# is narrower than a two-digit badge, so nothing fits beside a swatch.
SWATCH_SIDE = 84
SWATCH_GAP = 12
SWATCH_LEFT = 169
SWATCH_TOP = 1417


def _swatch_row(count: int = 8, top: int = SWATCH_TOP) -> list:
    pitch = SWATCH_SIDE + SWATCH_GAP
    return [
        fakes.element(
            box=(
                SWATCH_LEFT + column * pitch,
                top,
                SWATCH_LEFT + column * pitch + SWATCH_SIDE,
                top + SWATCH_SIDE,
            ),
            name="View",
            text="",
        )
        for column in range(count)
    ]


def _box(x_min: int, y_min: int, x_max: int, y_max: int) -> BoundingBox:
    return BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def _canvas(color=(255, 255, 255)) -> Image.Image:
    return Image.new("RGB", SCREEN, color)


def _is_dark(pixel) -> bool:
    return sum(pixel) < 120


class BadgePointSizeTests(unittest.TestCase):
    """A number sized for the screen covers a swatch sized for a finger."""

    def test_a_large_element_takes_the_largest_badge(self):
        size = badges._badge_point_size(_box(0, 0, 400, 400))

        self.assertEqual(size, badges.BADGE_POINT_SIZE_MAX)

    def test_a_small_element_takes_the_smallest_badge(self):
        size = badges._badge_point_size(_box(0, 0, 20, 20))

        self.assertEqual(size, badges.BADGE_POINT_SIZE_MIN)

    def test_the_badge_follows_the_shortest_side(self):
        """A wide, short row is limited by its height, not its width."""
        wide_and_short = badges._badge_point_size(_box(0, 0, 900, 45))
        square = badges._badge_point_size(_box(0, 0, 45, 45))

        self.assertEqual(wide_and_short, square)

    def test_the_keyboard_band_still_scales_with_the_screenshot(self):
        """One label over a large area, so it is read at screen distance."""
        small, large = badges._band_point_size(800), badges._band_point_size(2400)

        self.assertGreater(large, small)


class BadgePlacementTests(unittest.TestCase):
    """A badge beside a box lands on the next box when elements are dense.

    A number on the wrong swatch is answered confidently, and the wrong
    colour is tapped.
    """

    def test_a_badge_sits_inside_the_element_it_names(self):
        box = _box(545, 1417, 629, 1501)

        slot = badges._inside_slot(box, badge_width=24, badge_height=20, align_right=False)

        self.assertIsNotNone(slot)
        assert slot is not None
        self.assertGreaterEqual(slot[0], box.x_min)
        self.assertGreaterEqual(slot[1], box.y_min)
        self.assertLessEqual(slot[2], box.x_max)
        self.assertLessEqual(slot[3], box.y_max)

    def test_a_text_element_keeps_the_start_of_its_label(self):
        """Labels are left-aligned, so the first word is what names the target."""
        box = _box(100, 200, 500, 260)

        slot = badges._inside_slot(box, badge_width=24, badge_height=20, align_right=True)

        assert slot is not None
        self.assertGreater(slot[0], (box.x_min + box.x_max) // 2)
        self.assertLessEqual(slot[2], box.x_max)

    def test_an_element_too_small_for_a_badge_takes_none(self):
        """A scrollbar sliver cannot hold a number without becoming one."""
        slot = badges._inside_slot(
            _box(770, 380, 780, 1040), badge_width=24, badge_height=20, align_right=False
        )

        self.assertIsNone(slot)

    def test_an_element_mostly_covered_by_its_badge_takes_none(self):
        slot = badges._inside_slot(
            _box(0, 0, 30, 30), badge_width=24, badge_height=20, align_right=False
        )

        self.assertIsNone(slot)

    def test_a_badge_with_nowhere_free_still_gets_a_slot(self):
        """Every element keeps a number, even where the screen leaves no room.

        An unnumbered element cannot be named, so a crowded slot is the better
        of the two failures.
        """
        box = _box(0, 0, 10, 10)
        everywhere = [(0, 0, SCREEN[0], SCREEN[1])]

        slot = badges._outside_slot(
            box,
            badge_width=24,
            badge_height=20,
            placed_slots=everywhere,
            element_boxes=everywhere,
            screen=SCREEN,
        )

        self.assertEqual(slot, (0, 0, 24, 20))

    def test_an_outside_badge_avoids_the_neighbouring_element(self):
        """Avoiding only other badges is what put a number on the wrong box."""
        crowded = _box(100, 100, 110, 400)
        neighbour = (40, 60, 99, 400)

        slot = badges._outside_slot(
            crowded,
            badge_width=24,
            badge_height=20,
            placed_slots=[],
            element_boxes=[neighbour],
            screen=SCREEN,
        )

        self.assertFalse(badges._overlaps_any(slot, [neighbour]))


class BadgeContrastTests(unittest.TestCase):
    """A dark badge on a dark control is as unreadable as no badge."""

    def test_a_dark_element_takes_a_light_badge(self):
        navy = Image.new("RGB", (40, 40), (0, 0, 128))

        background, ink = badges._select_badge_colors(navy, (0, 0, 40, 40))

        self.assertEqual(background, badges.BADGE_LIGHT)
        self.assertEqual(ink, badges.BADGE_DARK)

    def test_a_light_element_takes_a_dark_badge(self):
        cream = Image.new("RGB", (40, 40), (240, 230, 140))

        background, ink = badges._select_badge_colors(cream, (0, 0, 40, 40))

        self.assertEqual(background, badges.BADGE_DARK)
        self.assertEqual(ink, badges.BADGE_LIGHT)


class RenderedPaletteTests(unittest.TestCase):
    """The whole overlay, on a grid too dense for a badge to sit beside."""

    def _rendered_row(self) -> tuple:
        row = _swatch_row()
        return badges.render(_canvas(), fakes.numbered(*row)), row

    def test_no_badge_lands_in_the_gap_between_two_swatches(self):
        marked, row = self._rendered_row()
        pixels = marked.load()
        assert pixels is not None

        for left, right in pairwise(row):
            gap_start = left.bounding_box.x_max + 2
            gap_end = right.bounding_box.x_min - 2
            dark = [
                (x, y)
                for x in range(gap_start, gap_end)
                for y in range(SWATCH_TOP, SWATCH_TOP + SWATCH_SIDE)
                if _is_dark(pixels[x, y])
            ]
            self.assertEqual(dark, [], f"a badge reached the gap at x={gap_start}")

    def test_no_badge_lands_above_the_row_it_belongs_to(self):
        """Placing a badge outside the top-left corner lands it a row up."""
        marked, _ = self._rendered_row()
        pixels = marked.load()
        assert pixels is not None

        band = [
            (x, y)
            for x in range(SWATCH_LEFT - 60, SWATCH_LEFT + 8 * (SWATCH_SIDE + SWATCH_GAP))
            for y in range(SWATCH_TOP - 40, SWATCH_TOP)
            if _is_dark(pixels[x, y])
        ]

        self.assertEqual(band, [], "a badge was drawn above its own row")

    def test_every_swatch_is_still_numbered(self):
        marked, row = self._rendered_row()
        pixels = marked.load()
        assert pixels is not None

        for element in row:
            box = element.bounding_box
            dark = any(
                _is_dark(pixels[x, y])
                for x in range(box.x_min, box.x_max)
                for y in range(box.y_min, box.y_max)
            )
            self.assertTrue(dark, f"the swatch at x={box.x_min} lost its number")

    def test_the_screenshot_keeps_its_size(self):
        marked, _ = self._rendered_row()

        self.assertEqual(marked.size, SCREEN)

    def test_an_element_too_thin_for_a_badge_is_still_numbered(self):
        """A scrollbar has no room inside, so its number goes beside it."""
        sliver = fakes.element(box=(770, 380, 780, 1040), name="Line", text="")

        marked = badges.render(_canvas(), fakes.numbered(sliver))
        pixels = marked.load()
        assert pixels is not None

        beside = any(
            _is_dark(pixels[x, y])
            for x in range(700, 800)
            for y in range(340, 420)
        )
        self.assertTrue(beside, "the sliver was left without a number")

    def test_an_open_keyboard_is_banded_rather_than_numbered(self):
        """The band says which half of the screen is keys, in one label."""
        row = _swatch_row()
        band = (0, 1800, SCREEN[0], 2400)

        marked = badges.render(_canvas(), fakes.numbered(*row), band=band)
        pixels = marked.load()
        assert pixels is not None

        banded = sum(
            1
            for x in range(0, SCREEN[0], 7)
            for y in range(1810, 2390, 7)
            if pixels[x, y] != (255, 255, 255)
        )
        self.assertGreater(banded, 0, "the keyboard area was left unmarked")


class BadgeFontTests(unittest.TestCase):
    """A number the model cannot read is worse than no number at all.

    The planner answers with an index, so an unreadable badge does not fail. It
    comes back as a wrong element, which reads as an app defect.
    """

    def test_the_font_ships_with_the_package(self):
        """A system font path differs per platform, and containers carry none."""
        self.assertTrue(
            badges.FONT_PATH.is_file(), f"{badges.FONT_PATH} is not in the package"
        )

    def test_every_glyph_the_badges_use_is_in_the_font(self):
        """A glyph the face lacks draws as an empty box, which names nothing."""
        font = badges._font(badges.BADGE_POINT_SIZE_MAX)
        draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))

        for label in [str(digit) for digit in range(10)] + ["KEYBOARD"]:
            left, _, right, _ = draw.textbbox((0, 0), label, font=font)
            self.assertGreater(right - left, 0, f"{label!r} drew no pixels")

    def test_the_smallest_badge_still_draws_readable_digits(self):
        """The floor is the size the model was shown to still resolve."""
        font = badges._font(badges.BADGE_POINT_SIZE_MIN)
        draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))

        _, top, _, bottom = draw.textbbox((0, 0), "18", font=font)

        self.assertGreaterEqual(bottom - top, 8)


if __name__ == "__main__":
    unittest.main()
