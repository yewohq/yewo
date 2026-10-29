"""Drawing the number that names each detected element on the screenshot.

The number is the only way the planner can name a target. Each badge must show
which box it belongs to. A badge beside a box can land on a neighbouring box.
This occurs when the gap is narrower than the badge, as in icon grids and
colour palettes. A badge inside its box has a clear association. Place it
there when the element is large enough.

Badge text is sized from the element rather than the screen. A number large
enough to read on a full-screen dialog covers a small swatch completely.
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageStat

if TYPE_CHECKING:
    from yewo.agent.grounding.marks import Mark

logger = logging.getLogger(__name__)

Rectangle = tuple[int, int, int, int]

BADGE_PAD = 3
# Clear of the element's own outline, which is drawn on the boundary.
BADGE_INSET = 3
BADGE_POINT_SIZE_MIN = 12
BADGE_POINT_SIZE_MAX = 16
# Badge text as a fraction of the element's shortest side.
BADGE_SIZE_RATIO = 0.28
# Above this share of the element, a badge hides more than it names.
BADGE_COVERAGE_LIMIT = 0.25
# Perceived brightness, 0-255, below which a badge needs a light background.
BADGE_CONTRAST_LUMINANCE = 140
BADGE_LIGHT = (255, 255, 255)
BADGE_DARK = (18, 18, 18)
ELEMENT_OUTLINE_COLOR = (214, 16, 46)
BAND_COLOR = (36, 82, 168)

FONT_PATH = Path(__file__).parent / "fonts" / "Aileron-Bold.otf"
"""Aileron Bold, shipped with the package. Public domain, so it carries no notice.

The badge numbers are the only way the planner can name a target, so they have
to be legible at screenshot resolution on every machine. A system font cannot
promise that: the path differs per platform and a slim container image often
carries none.
"""


@lru_cache(maxsize=16)
def _font(point_size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATH, point_size)


def font(point_size: int) -> ImageFont.FreeTypeFont:
    """The badge typeface at one size, for anything else that draws on a screenshot.

    Shared so an overlay drawn outside this module cannot end up on a system
    font that the badges themselves refuse to depend on.
    """
    return _font(point_size)


def _badge_point_size(box) -> int:
    """Badge text sized to the element, so a small control keeps its picture."""
    shortest_side = min(box.x_max - box.x_min, box.y_max - box.y_min)
    return max(
        BADGE_POINT_SIZE_MIN,
        min(BADGE_POINT_SIZE_MAX, round(shortest_side * BADGE_SIZE_RATIO)),
    )


def _band_point_size(longest_screen_side: int) -> int:
    """The keyboard band carries one label over a large area, so it scales."""
    return max(10, int(15 * (longest_screen_side / 1200.0)))


def _select_badge_colors(image: Image.Image, slot: Rectangle) -> tuple:
    """Background and ink that stay legible on the pixels the badge covers.

    Sampled from the unmarked screenshot. Sampling the annotated copy would
    read an earlier badge instead of the element underneath.
    """
    red, green, blue = ImageStat.Stat(image.crop(slot).convert("RGB")).mean[:3]
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    if luminance < BADGE_CONTRAST_LUMINANCE:
        return BADGE_LIGHT, BADGE_DARK
    return BADGE_DARK, BADGE_LIGHT


def _overlaps_any(slot: Rectangle, rectangles: Sequence[Rectangle]) -> bool:
    return any(
        slot[0] < other[2]
        and other[0] < slot[2]
        and slot[1] < other[3]
        and other[1] < slot[3]
        for other in rectangles
    )


def _inside_slot(
    box, badge_width: int, badge_height: int, align_right: bool
) -> Optional[Rectangle]:
    """A slot within the element, or None when the element cannot hold one.

    Text is left-aligned, so a badge over the start of a label costs the
    planner the word it navigates by. It goes to the trailing end instead.
    """
    box_width = box.x_max - box.x_min
    box_height = box.y_max - box.y_min
    if (
        badge_width + BADGE_INSET * 2 > box_width
        or badge_height + BADGE_INSET * 2 > box_height
        or badge_width * badge_height > box_width * box_height * BADGE_COVERAGE_LIMIT
    ):
        return None
    top = box.y_min + BADGE_INSET
    left = (
        box.x_max - BADGE_INSET - badge_width
        if align_right
        else box.x_min + BADGE_INSET
    )
    return (left, top, left + badge_width, top + badge_height)


def _outside_slot(
    box,
    badge_width: int,
    badge_height: int,
    placed_slots: Sequence[Rectangle],
    element_boxes: Sequence[Rectangle],
    screen: tuple[int, int],
) -> Rectangle:
    """A slot beside an element too small to hold its own number.

    Checked against the other elements as well as the other badges. A number
    on a neighbouring control names the wrong one, which is a worse failure
    than two numbers crowding each other.
    """
    screen_width, screen_height = screen
    candidates = (
        (box.x_min - badge_width, box.y_min - badge_height),
        (box.x_min, box.y_min - badge_height),
        (box.x_max, box.y_min - badge_height),
        (box.x_min - badge_width, box.y_min),
        (box.x_max, box.y_min),
    )
    for left, top in candidates:
        left = max(0, min(left, screen_width - badge_width))
        top = max(0, min(top, screen_height - badge_height))
        slot = (left, top, left + badge_width, top + badge_height)
        if not _overlaps_any(slot, placed_slots) and not _overlaps_any(
            slot, element_boxes
        ):
            return slot
    left = max(0, min(box.x_min, screen_width - badge_width))
    top = max(0, min(box.y_min, screen_height - badge_height))
    return (left, top, left + badge_width, top + badge_height)


def _draw_badge(draw, image: Image.Image, slot: Rectangle, label: str, font) -> None:
    background, ink = _select_badge_colors(image, slot)
    left, top, _, _ = draw.textbbox((0, 0), label, font=font)
    draw.rectangle(list(slot), fill=background)
    draw.text(
        (slot[0] + BADGE_PAD - left, slot[1] + BADGE_PAD - top),
        label,
        fill=ink,
        font=font,
    )


def _draw_band(draw, band: Rectangle, font, screen: tuple[int, int]) -> None:
    x0, y0, x1, y1 = band
    draw.rectangle([x0, y0, x1, y1], fill=BAND_COLOR + (36,))
    draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=BAND_COLOR, width=4)

    label = "KEYBOARD"
    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    width, height = right - left + BADGE_PAD * 2, bottom - top + BADGE_PAD * 2
    # Bottom-right: the top-left of a keyboard is its most-used keys.
    band_left = max(0, x1 - width)
    band_top = max(0, min(y1 - height, screen[1] - height))
    draw.rectangle(
        [band_left, band_top, band_left + width, band_top + height], fill=BAND_COLOR
    )
    draw.text(
        (band_left + BADGE_PAD - left, band_top + BADGE_PAD - top),
        label,
        fill=BADGE_LIGHT,
        font=font,
    )


def render(
    image: Image.Image,
    marks: "Sequence[Mark]",
    outline_color: tuple = ELEMENT_OUTLINE_COLOR,
    band: Optional[Rectangle] = None,
) -> Image.Image:
    """Draws each element's box and the number that names it."""
    marked = image.copy()
    draw = ImageDraw.Draw(marked, "RGBA")
    screen = marked.size
    if band:
        _draw_band(draw, band, _font(_band_point_size(max(screen))), screen)
    outline_width = max(1, int(1.5 * (max(screen) / 1000.0)))
    element_boxes: list[Rectangle] = [
        (box.x_min, box.y_min, box.x_max, box.y_max)
        for box in (mark.element.bounding_box for mark in marks)
    ]
    placed_slots: list[Rectangle] = []

    for mark in marks:
        box = mark.element.bounding_box
        draw.rectangle(
            [box.x_min, box.y_min, box.x_max, box.y_max],
            outline=outline_color,
            width=outline_width,
        )
        label = str(mark.index)
        font = _font(_badge_point_size(box))
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        badge_width = int(right - left) + BADGE_PAD * 2
        badge_height = int(bottom - top) + BADGE_PAD * 2
        slot = _inside_slot(
            box, badge_width, badge_height, align_right=bool(mark.element.text)
        )
        if slot is None:
            slot = _outside_slot(
                box, badge_width, badge_height, placed_slots, element_boxes, screen
            )
        placed_slots.append(slot)
        _draw_badge(draw, image, slot, label, font)
    return marked
