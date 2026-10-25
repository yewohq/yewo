"""Numbering the detected elements so a target can be named rather than guessed.

A coordinate answer cannot be checked. Any pair of numbers is structurally
valid, so an element the planner cannot find becomes a tap into empty space.
Numbering the detections makes the answer a choice from a known set, and an
index that names nothing fails before anything touches the device.

The screenshot labels and the prompt labels are generated here, from one pass,
so the two cannot drift apart.
"""

import io
import logging
from copy import copy
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from PIL import Image, ImageDraw, ImageFont

from yewo.agent.actuation.actions import (
    TARGETED_ACTIONS,
    TEXT_FIELD_ACTIONS,
    Coordinates,
)
from yewo.agent.grounding import GROUNDING_MARKS
from yewo.agent.perception import keyboard
from yewo.agent.perception.elements import GUIElement

logger = logging.getLogger(__name__)

DUPLICATE_OVERLAP = 0.80
# How much of a box must sit inside another before it counts as part of it.
CONTAINED_FRACTION = 0.90
# A box bigger than this is a layout region, not a control, so it keeps its parts.
MAX_CONTAINER_FRACTION = 0.25
ROW_BAND = 25
BADGE_PAD = 3
BAND_COLOR = (36, 82, 168)
# Where the soft keyboard starts, and how short a label has to be to be a key.


@dataclass(frozen=True)
class Mark:
    """One numbered element: what the model picks and what we tap.

    `point` can differ from the box centre. A collapsed control is drawn around
    its container, but touched where its label is. The label is the part that
    reliably responds: the centre of a wide list row often lands on padding.
    """

    index: int
    element: GUIElement
    point: Optional[tuple[int, int]] = None

    refuses_text: bool = False
    """Typing here was tried and focus never moved, so it holds no text.

    The fact rides on the mark so the planner's prompt and the guard before
    dispatch read the same thing, without either being told separately.
    """

    @property
    def center(self) -> tuple[int, int]:
        if self.point is not None:
            return self.point
        box = self.element.bounding_box
        return (box.x_min + box.x_max) // 2, (box.y_min + box.y_max) // 2


Box = tuple[int, int, int, int]
"""An element's place on screen, which outlives the number drawn on it."""


def box_of(element: GUIElement) -> Box:
    """The element's bounds, in the form a fact about it is remembered by."""
    bounds = element.bounding_box
    return (bounds.x_min, bounds.y_min, bounds.x_max, bounds.y_max)


def is_same_element(element: GUIElement, remembered: Box) -> bool:
    """Whether a freshly detected element is the one something was learned about.

    Numbers run in reading order over whatever the detector returned, so one
    box more or fewer renumbers everything below it. A clock ticking in the
    status bar is enough to change what the detector finds, while the screen
    fingerprint stays identical. A number is therefore no way to recognise an
    element across two looks at one screen. Its place on screen is.
    """
    left, top, right, bottom = remembered
    width = max(0, min(element.bounding_box.x_max, right) - max(element.bounding_box.x_min, left))
    height = max(0, min(element.bounding_box.y_max, bottom) - max(element.bounding_box.y_min, top))
    intersection = width * height
    if not intersection:
        return False
    union = (
        _area(element.bounding_box) + (right - left) * (bottom - top) - intersection
    )
    return intersection / float(union or 1) >= DUPLICATE_OVERLAP


def _overlap(a, b) -> float:
    left, top = max(a.x_min, b.x_min), max(a.y_min, b.y_min)
    right, bottom = min(a.x_max, b.x_max), min(a.y_max, b.y_max)
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    areas = (a.x_max - a.x_min) * (a.y_max - a.y_min) + (b.x_max - b.x_min) * (
        b.y_max - b.y_min
    )
    return intersection / float(areas - intersection or 1)


def _center(element) -> tuple[int, int]:
    box = element.bounding_box
    return (box.x_min + box.x_max) // 2, (box.y_min + box.y_max) // 2


def _area(box) -> int:
    return max(1, (box.x_max - box.x_min) * (box.y_max - box.y_min))


def _contained_in(child, parent) -> bool:
    overlap_w = max(0, min(child.x_max, parent.x_max) - max(child.x_min, parent.x_min))
    overlap_h = max(0, min(child.y_max, parent.y_max) - max(child.y_min, parent.y_min))
    return (overlap_w * overlap_h) / _area(child) >= CONTAINED_FRACTION


def _is_one_control(parent: GUIElement, children: List[GUIElement]) -> bool:
    """Whether a box and its contents are one thing, or several things in a container.

    A field is a wrapper around a single value: no label of its own, one child
    that carries text. A list row is not. It holds several labels and a
    control, and folding them together loses the control, which is the only
    part that responds.
    """
    if (parent.text or "").strip():
        return False
    return sum(1 for child in children if (child.text or "").strip()) <= 1


def _collapse_nests(elements: List[GUIElement], screen_area: int) -> List[tuple]:
    """Folds a control's parts into the control, as one numbered thing.

    A text field is not one box on screen. It is a container holding an icon,
    the typed value, and sometimes a label. The detector finds all of them, so
    numbering each would offer four choices for one field. Three of them land
    somewhere that may not respond.

    The container keeps the outline, since that is the thing being pointed at.
    The parts give it its name and the spot to touch. An empty container has no
    name, and its centre is often padding.
    """
    # Without a screen size the guard has to come from the elements themselves,
    # or a full-screen container would swallow every control on the page.
    if not screen_area:
        screen_area = max(
            (_area(el.bounding_box) for el in elements), default=0
        )
    limit = screen_area * MAX_CONTAINER_FRACTION
    ordered = sorted(elements, key=lambda el: _area(el.bounding_box), reverse=True)

    taken: set[int] = set()
    groups: List[tuple] = []
    for parent in ordered:
        if id(parent) in taken:
            continue
        if limit and _area(parent.bounding_box) > limit:
            continue
        children = [
            child
            for child in elements
            if child is not parent
            and id(child) not in taken
            and _contained_in(child.bounding_box, parent.bounding_box)
        ]
        if not children or not _is_one_control(parent, children):
            continue
        taken.add(id(parent))
        taken.update(id(child) for child in children)
        groups.append((parent, children))

    collapsed = []
    for element in elements:
        if id(element) in taken:
            continue
        collapsed.append((element, []))
    for parent, children in groups:
        collapsed.append((parent, children))
    return collapsed


def _named(parent: GUIElement, children: List[GUIElement]) -> GUIElement:
    """The container, wearing the names of its parts."""
    texts = [c.text.strip() for c in children if (c.text or "").strip()]
    if (parent.text or "").strip():
        texts.insert(0, parent.text.strip())
    if not texts:
        return parent
    named = copy(parent)
    named.text = " ".join(dict.fromkeys(texts))
    return named


def _touch_point(parent: GUIElement, children: List[GUIElement]) -> Optional[tuple]:
    for child in children:
        if (child.text or "").strip():
            return _center(child)
    return None


def build(elements: List[GUIElement], screen_area: int = 0) -> List[Mark]:
    """Numbers the elements in reading order, one number per control.

    Reading order rather than detection order, so the numbers run down the
    screen the way a person scans it. A misread number then lands on a
    neighbour rather than somewhere unrelated.
    """
    kept: List[GUIElement] = []
    for element in elements or []:
        if any(
            element.text == other.text
            and _overlap(element.bounding_box, other.bounding_box) >= DUPLICATE_OVERLAP
            for other in kept
        ):
            continue
        kept.append(element)

    groups = _collapse_nests(kept, screen_area)
    groups.sort(key=lambda g: (_center(g[0])[1] // ROW_BAND, _center(g[0])[0]))
    return [
        Mark(
            index=position,
            element=_named(parent, children),
            point=_touch_point(parent, children),
        )
        for position, (parent, children) in enumerate(groups, 1)
    ]


def find(marks: Sequence[Mark], index: int) -> Optional[Mark]:
    """The mark with this number, or None if the model named one that isn't there."""
    for mark in marks:
        if mark.index == index:
            return mark
    return None


def _region(element, height: int) -> str:
    _, cy = _center(element)
    if height <= 0:
        return "unknown"
    position = cy / height
    if position < 0.33:
        return "upper"
    if position < 0.66:
        return "middle"
    return "lower"


def describe(marks: List[Mark], height: int) -> str:
    """The list offered to the planner alongside the numbered screenshot.

    Each line carries a number and a coarse vertical region, which is what the
    keyboard rules need. Never coordinates: a coordinate in the prompt is one
    the model can echo back, and an echoed coordinate cannot be told from a
    guessed one.

    Never include the detected text either. The model already sees the screen
    that contains the words. Our OCR reads English only, so its label can be
    wrong when the screenshot is correct. The numbers are listed because the
    planner may answer only with a number from this list.

    A mark that already refused text says so. Nothing on screen separates a
    field from a picture of one, so the only evidence is having tried.
    """
    lines = [
        f"{mark.index} | {_region(mark.element, height)}"
        + (" | holds no text" if mark.refuses_text else "")
        for mark in marks
    ]
    return "\n".join(lines) or "None detected."


FONT_PATH = Path(__file__).parent / "fonts" / "Aileron-Bold.otf"
"""Aileron Bold, shipped with the package. Public domain, so it carries no notice.

The badge numbers are the only way the planner can name a target, so they have
to be legible at screenshot resolution on every machine. A system font cannot
promise that: the path differs per platform and a slim container image often
carries none.
"""


@lru_cache(maxsize=8)
def _font(scale_ref: int) -> ImageFont.FreeTypeFont:
    """The badge face, sized for this screen."""
    size = max(10, int(15 * (scale_ref / 1200.0)))
    return ImageFont.truetype(FONT_PATH, size)


def _free_slot(box, text_w, text_h, placed, screen):
    """Finds somewhere for a badge that does not sit on another badge.

    Numbers that overlap each other are worse than no numbers: two legible
    digits read as one illegible one, and the model picks neither.
    """
    width, height = screen
    pad = BADGE_PAD * 2
    w, h = text_w + pad, text_h + pad
    candidates = (
        (box.x_min - w, box.y_min - h),  # outside the top-left corner
        (box.x_min, box.y_min - h),  # directly above
        (box.x_max, box.y_min - h),  # outside the top-right corner
        (box.x_min - w, box.y_min),  # to the left
        (box.x_max, box.y_min),  # to the right
        (box.x_min, box.y_min),  # inside, last resort
    )
    for x, y in candidates:
        x = max(0, min(x, width - w))
        y = max(0, min(y, height - h))
        slot = (x, y, x + w, y + h)
        if not any(
            slot[0] < other[2]
            and other[0] < slot[2]
            and slot[1] < other[3]
            and other[1] < slot[3]
            for other in placed
        ):
            return slot
    x = max(0, min(box.x_min, width - w))
    y = max(0, min(box.y_min, height - h))
    return (x, y, x + w, y + h)


def _draw_band(draw, band, font, screen) -> None:
    x0, y0, x1, y1 = band
    draw.rectangle([x0, y0, x1, y1], fill=BAND_COLOR + (36,))
    draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=BAND_COLOR, width=4)

    label = "KEYBOARD"
    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    w, h = right - left + BADGE_PAD * 2, bottom - top + BADGE_PAD * 2
    # Bottom-right: the top-left of a keyboard is its most-used keys.
    bx, by = max(0, x1 - w), max(0, min(y1 - h, screen[1] - h))
    draw.rectangle([bx, by, bx + w, by + h], fill=BAND_COLOR)
    draw.text(
        (bx + BADGE_PAD - left, by + BADGE_PAD - top), label, fill=(255, 255, 255), font=font
    )


def render(
    image: Image.Image,
    marks: List[Mark],
    color=(214, 16, 46),
    band: Optional[tuple] = None,
) -> Image.Image:
    """Draws each number near its element, avoiding other badges.

    Small, and outside the box where there is room. The number has to be
    readable without hiding the thing it names. A badge covering a label costs
    the planner the word it navigates by.
    """
    marked = image.copy()
    draw = ImageDraw.Draw(marked, "RGBA")
    screen = marked.size
    font = _font(max(screen))
    if band:
        _draw_band(draw, band, font, screen)
    outline = max(1, int(1.5 * (max(screen) / 1000.0)))
    placed = []

    for mark in marks:
        box = mark.element.bounding_box
        draw.rectangle(
            [box.x_min, box.y_min, box.x_max, box.y_max], outline=color, width=outline
        )

        label = str(mark.index)
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        slot = _free_slot(box, right - left, bottom - top, placed, screen)
        placed.append(slot)

        draw.rectangle(list(slot), fill=color)
        draw.text(
            (slot[0] + BADGE_PAD - left, slot[1] + BADGE_PAD - top),
            label,
            fill=(255, 255, 255),
            font=font,
        )
    return marked


def resolve_mark(action, element_marks, grounding: str = "") -> tuple:
    """Turns a chosen number into the point to touch.

    Returns (reason the step cannot proceed, action to run instead). A marks
    action becomes its coordinate form here, because every mode ends in a point.
    the number only decides who worked that point out.

    In marks mode the number is the only accepted way to name a target. A
    response carrying coordinates instead is refused rather than obeyed.
    Accepting it would reinstate the invented coordinate that the numbering
    exists to prevent, and we cannot tell a read-off coordinate from a guessed
    one.
    """
    if action.type not in TARGETED_ACTIONS:
        return None, None

    index = getattr(action, "mark", None)
    if grounding == GROUNDING_MARKS:
        if index is None:
            return (
                "Not executed. This action needs `mark`. `mark` is the number on "
                "the screenshot. Do not send coordinates. If the element you need "
                "has no number, answer target_not_listed."
            ), None
    elif index is None:
        if getattr(action, "coordinates", None) is None:
            return "Not executed. This action needs `coordinates` of the target.", None
        return None, None

    mark = find(element_marks, index)
    if mark is None:
        available = len(element_marks)
        logger.warning("Element %s was chosen but only 1-%d exist.", index, available)
        return (
            f"Not executed. There is no element {index}. The numbers on this "
            f"screen are 1 to {available}. Choose one of them."
        ), None

    if mark.refuses_text and action.type in TEXT_FIELD_ACTIONS:
        return (
            f"Not executed. Element {index} holds no text. Typing there did not "
            "move the cursor, twice. Choose a different number. If no number is "
            "the field you need, the field is off screen. Close the keyboard "
            "with key_event 4, or scroll. Then look again."
        ), None

    x, y = mark.center
    # Every marks variant carries this, set by actions._marks_variant.
    resolved = type(action).coordinate_form(
        coordinates=Coordinates(x=x, y=y), **action.model_dump(exclude={"mark", "type"})
    )
    logger.info("Element %s resolves to (%d, %d).", index, x, y)
    return None, resolved


def refuse_mark(
    strikes: Dict[str, List[Box]],
    refused: Dict[str, List[Box]],
    phash: Optional[str],
    index: Optional[int],
    element_marks: Sequence[Mark],
) -> None:
    """Records a failure to put text somewhere, and bans the place on the second.

    Focus that did not move proves the tap reached nothing focusable. It does
    not prove the element holds no text. A tap into a real field's padding has
    the same result. Banning after one failure can hide a genuine editable
    field for the rest of the screen. The first failure only marks the place.

    The failure is kept by place rather than by number. Numbers follow the
    detector's reading order. They can change when a detection appears or
    disappears between two observations of one screen.
    """
    mark = find(element_marks, index) if index is not None else None
    if phash is None or mark is None:
        return

    box = box_of(mark.element)
    first = strikes.setdefault(phash, [])
    if not any(is_same_element(mark.element, seen) for seen in first):
        first.append(box)
        logger.info("Element %d took no text. Trying it once more.", index)
        return

    refused.setdefault(phash, []).append(box)
    logger.info("Element %d took no text twice. Its place will be skipped.", index)


def number(
    elements: List[GUIElement],
    screenshot: bytes,
    keyboard_open: Optional[bool],
    screen_height: Optional[int],
    refused: Sequence[Box] = (),
) -> tuple[List[Mark], bytes]:
    """The numbered targets, and the screenshot carrying their numbers.

    Built together so that a number in the prompt is a number the planner can
    see. A place that already refused text keeps its number and is marked, so
    the planner is told rather than left to choose it again.
    """
    targets, keys = keyboard.split_keyboard(elements, keyboard_open, screen_height)
    image = Image.open(io.BytesIO(screenshot))
    width, height = image.size

    built = [
        replace(mark, refuses_text=True)
        if any(is_same_element(mark.element, box) for box in refused)
        else mark
        for mark in build(targets, screen_area=width * height)
    ]

    numbered = io.BytesIO()
    render(image, built, band=keyboard.keyboard_band(keys, width)).save(
        numbered, format="PNG"
    )
    return built, numbered.getvalue()
