"""Candidate points for a control the detector never found.

Nothing here is a target. A cell is a place worth trying once. It sits inside
a region that two detected controls bound. It stays a candidate until the
device says otherwise.

A separate type from `Mark` is the guarantee. A cell cannot reach the planner's
numbered list through the wrong function call. It is not a mark, and it never
becomes one.

The overlay this draws is evidence of an intervention, not a view anyone plans
from. It is written to the run directory and shown to nobody.
"""

import io
import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

from PIL import Image, ImageDraw

from yewo.agent.grounding.badges import font
from yewo.agent.perception.elements import GUIElement
from yewo.agent.run.control_discovery import MAX_PROBE_ATTEMPTS

logger = logging.getLogger(__name__)

Box = tuple[int, int, int, int]

MAX_CANDIDATES = MAX_PROBE_ATTEMPTS
"""Ceiling on both the grid and the probe budget, deliberately one number.

A grid larger than the budget would offer cells that are never tried, and a
report listing them reads as coverage that did not happen."""

MIN_CELL_PIXELS = 48
"""Smallest cell worth a probe, in full-screen pixels.

Below a touch target two cells land on the same control, so the second probe
can only repeat the first."""

CELL_COLOR = (36, 82, 168)
HIT_COLOR = (16, 148, 72)
MISS_COLOR = (150, 150, 150)
EVIDENCE_COLOR = (214, 16, 46)


@dataclass(frozen=True)
class Candidate:
    """One cell of the grid, in full-screen coordinates."""

    index: int
    box: Box
    detection: Optional[GUIElement] = None
    """A detection the magnified pass found overlapping this cell.

    Present or absent decides what may be done with the cell. A cell with
    evidence was seen and not numbered. A cell without it was never seen at
    all, and the only safe probe for one of those is a probe that can be
    undone."""

    @property
    def point(self) -> tuple[int, int]:
        return (self.box[0] + self.box[2]) // 2, (self.box[1] + self.box[3]) // 2


def _grid_shape(crop: Box) -> tuple[int, int]:
    """Rows and columns that keep every cell at least a touch target wide.

    Sized from the region rather than fixed, so a one-line band is a row of
    cells and a block is a block of them. The product is capped, and the cap
    is reported by the caller when it bites.
    """
    width, height = crop[2] - crop[0], crop[3] - crop[1]
    columns = max(1, min(MAX_CANDIDATES, width // MIN_CELL_PIXELS))
    rows = max(1, min(MAX_CANDIDATES, height // MIN_CELL_PIXELS))
    while rows * columns > MAX_CANDIDATES:
        if columns >= rows:
            columns -= 1
        else:
            rows -= 1
    return rows, columns


def cell_capacity(crop: Box) -> int:
    """How many cells the region has room for, before the budget is applied.

    Separate from `build` so the difference can be reported. A silently
    truncated grid looks exactly like a region that was fully covered.
    """
    width, height = crop[2] - crop[0], crop[3] - crop[1]
    return max(1, width // MIN_CELL_PIXELS) * max(1, height // MIN_CELL_PIXELS)


def _distance_from_centre(box: Box, crop: Box) -> float:
    centre_x, centre_y = (crop[0] + crop[2]) / 2, (crop[1] + crop[3]) / 2
    x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return (x - centre_x) ** 2 + (y - centre_y) ** 2


def _detection_at(box: Box, elements: Sequence[GUIElement]) -> Optional[GUIElement]:
    """The detection covering this cell's centre, preferring the most confident."""
    x, y = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
    covering = [
        element
        for element in elements
        if element.bounding_box.x_min <= x <= element.bounding_box.x_max
        and element.bounding_box.y_min <= y <= element.bounding_box.y_max
    ]
    if not covering:
        return None
    return max(covering, key=lambda element: element.confidence)


def build(crop: Box, elements: Sequence[GUIElement] = ()) -> List[Candidate]:
    """Cells over the region, ordered from its centre outwards.

    Centre first because a control that fills a gap between two labels is
    centred in it far more often than it is cornered. The order is the probe
    order, so the cheapest outcome is also the most likely one.
    """
    rows, columns = _grid_shape(crop)
    left, top, right, bottom = crop
    width, height = right - left, bottom - top
    boxes = [
        (
            left + column * width // columns,
            top + row * height // rows,
            left + (column + 1) * width // columns,
            top + (row + 1) * height // rows,
        )
        for row in range(rows)
        for column in range(columns)
    ]
    boxes.sort(key=lambda box: (_distance_from_centre(box, crop), box[1], box[0]))
    return [
        Candidate(index=position + 1, box=box, detection=_detection_at(box, elements))
        for position, box in enumerate(boxes)
    ]


def overlay(
    close_up: Image.Image,
    candidates: Sequence[Candidate],
    crop: Box,
    scale: int,
    verdicts: Optional[dict] = None,
) -> bytes:
    """Draw the grid on the magnified crop, for the run directory only.

    Colour carries the outcome so the file answers the question it exists for:
    which cells were tried, and what the device said to each.
    """
    canvas = close_up.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    verdicts = verdicts or {}
    for candidate in candidates:
        box = [
            (candidate.box[0] - crop[0]) * scale,
            (candidate.box[1] - crop[1]) * scale,
            (candidate.box[2] - crop[0]) * scale,
            (candidate.box[3] - crop[1]) * scale,
        ]
        verdict = verdicts.get(candidate.index)
        if verdict == "accepted":
            color = HIT_COLOR
        elif verdict:
            color = MISS_COLOR
        elif candidate.detection is not None:
            color = EVIDENCE_COLOR
        else:
            color = CELL_COLOR
        draw.rectangle(box, outline=color, width=2)
        draw.text((box[0] + 4, box[1] + 2), str(candidate.index), fill=color, font=font(16))
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return buffer.getvalue()
