"""Deriving the screen region a missing control must occupy.

A coarse third is a guess dressed as a location. It is the same rectangle on
every screen. It answers "where should I look" with "somewhere over there". A
magnified third still holds mostly the elements that were already numbered.

What the screen does know is where the controls around the gap are. A label
above and a label below bound a missing control vertically. The fields that
were detected elsewhere on the form say how wide a control on this screen is.
Those are measurements, not guesses, so a region built from them is bounded by
evidence and can be refused when the evidence is not there.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

from yewo.agent.perception.elements import GUIElement

logger = logging.getLogger(__name__)

Box = tuple[int, int, int, int]

MIN_GAP_SHARE = 0.015
"""Shortest vertical gap that could hold a control, as a share of the screen.

Below a touch target's height the gap is spacing between two controls, not a
control nobody detected."""

MIN_GAP_PIXELS = 24
"""Floor under MIN_GAP_SHARE, so a short screen still rejects hairline gaps."""

MAX_GAP_SHARE = 0.35
"""Tallest gap still worth calling a region.

A gap larger than this is empty screen. Cropping to it would be as coarse as
the third it replaces. The caller is told that nothing was derived, and the
coarse crop stands."""

FULL_WIDTH_SHARE = 0.5
"""How wide an element must be to count as a sibling field rather than a label."""

MIN_REGION_SHARE = 0.10
"""Narrowest derived region worth cropping, as a share of screen width."""

ROW_MERGE_SLACK = 4
"""Vertical pixels of overlap tolerated before two elements are one row.

Detections on one line rarely share an exact top edge. Without slack each one
opens a gap against its neighbour and the real gap is never the largest."""

PAD = 6
"""Pixels kept outside the derived box, so a control's border is not clipped."""

TARGET_LONG_SIDE = 4000
TARGET_SHORT_SIDE = 600
"""What magnification aims at. The binding constraint wins, so a thin band is
not blown up to an image whose width dwarfs anything the detector has seen."""

MIN_SCALE = 2
MAX_SCALE = 6

_ROWS = {"upper": 0, "middle": 1, "lower": 2}
_COLUMNS = {"left": 0, "center": 1, "right": 2}
_ALL_REGIONS = frozenset(
    f"{vertical}_{horizontal}" for vertical in _ROWS for horizontal in _COLUMNS
)

SIBLING_SPAN = "sibling_span"
SINGLE_SIBLING = "single_sibling"
ANCHOR_SPAN = "anchor_span"


@dataclass(frozen=True)
class DerivedRegion:
    """A crop built from the controls around it, and what built it."""

    crop: Box
    anchor_above: Box
    """The row whose bottom edge is the region's top."""
    anchor_below: Box
    """The row whose top edge is the region's bottom."""
    width_rule: str
    """Which measurement set the left and right edges."""
    width_samples: int
    """How many sibling fields the width was taken from."""

    def as_record(self) -> dict:
        """What the artifact stores, so a crop can be re-derived by hand."""
        return {
            "crop": list(self.crop),
            "anchor_above": list(self.anchor_above),
            "anchor_below": list(self.anchor_below),
            "width_rule": self.width_rule,
            "width_samples": self.width_samples,
        }


def _bounds_of(element: GUIElement) -> Box:
    bounds = element.bounding_box
    return (bounds.x_min, bounds.y_min, bounds.x_max, bounds.y_max)


def _hint_band(region: str, height: int) -> tuple[int, int]:
    """The vertical slice of screen the coarse region name points at.

    The hint chooses which gaps are candidates. It never becomes an edge of the
    crop, because the whole point is that its edges are arbitrary.
    """
    vertical = region.split("_", 1)[0]
    row = _ROWS[vertical]
    pad = height // 12
    return max(0, row * height // 3 - pad), min(height, (row + 1) * height // 3 + pad)


def _merge_rows(boxes: Sequence[Box]) -> List[Box]:
    """Collapse detections that sit on one line into a single occupied row.

    Merged by vertical overlap alone. Two elements side by side are one row
    here, even when nothing links them horizontally. A control cannot hide
    between them. The gap this looks for runs the width of the screen.
    """
    if not boxes:
        return []
    merged: List[list] = []
    for box in sorted(boxes, key=lambda b: (b[1], b[3])):
        if merged and box[1] < merged[-1][3] + ROW_MERGE_SLACK:
            row = merged[-1]
            row[0] = min(row[0], box[0])
            row[1] = min(row[1], box[1])
            row[2] = max(row[2], box[2])
            row[3] = max(row[3], box[3])
            continue
        merged.append(list(box))
    return [tuple(row) for row in merged]


def _find_gaps(rows: Sequence[Box], height: int) -> List[tuple[Box, Box]]:
    """Pairs of rows with room for an undetected control between them.

    Only interior gaps. A gap open at one end has one anchor, and one anchor
    fixes one edge. The other edge would fall back to a screen fraction. That
    is the guess this module exists to avoid.
    """
    floor = max(MIN_GAP_PIXELS, round(MIN_GAP_SHARE * height))
    ceiling = round(MAX_GAP_SHARE * height)
    pairs = []
    for above, below in zip(rows, rows[1:], strict=False):
        span = below[1] - above[3]
        if floor <= span <= ceiling:
            pairs.append((above, below))
    return pairs


def _select_nearest_gap(pairs: Sequence[tuple[Box, Box]], band: tuple[int, int]) -> tuple[Box, Box]:
    """The candidate gap closest to the middle of the region the planner named.

    The hint discriminates vertically only. Every gap here runs the full width,
    so a horizontal half of the hint would rank them all the same.
    """
    middle = (band[0] + band[1]) / 2
    return min(pairs, key=lambda pair: abs((pair[0][3] + pair[1][1]) / 2 - middle))


def _median_of(values: Sequence[int]) -> int:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) // 2


def _measure_extents(boxes: Sequence[Box], above: Box, below: Box, width: int) -> tuple[int, int, str, int]:
    """How wide a control on this screen is, measured from the ones that exist.

    Labels are narrow and fields are wide, so the siblings worth measuring are
    the wide ones. With none to measure, the two anchors still bound the
    region, which is narrower than the truth but never wider than the evidence.
    """
    siblings = [box for box in boxes if box[2] - box[0] >= FULL_WIDTH_SHARE * width]
    if len(siblings) >= 2:
        left = _median_of([box[0] for box in siblings])
        right = _median_of([box[2] for box in siblings])
        rule, samples = SIBLING_SPAN, len(siblings)
    elif siblings:
        left, right = siblings[0][0], siblings[0][2]
        rule, samples = SINGLE_SIBLING, 1
    else:
        left = min(above[0], below[0])
        right = max(above[2], below[2])
        rule, samples = ANCHOR_SPAN, 0

    if right - left < MIN_REGION_SHARE * width:
        left = min(left, above[0], below[0])
        right = max(right, above[2], below[2])
        rule, samples = ANCHOR_SPAN, 0
    return left, right, rule, samples


def derive(
    region: str, elements: Sequence[GUIElement], size: tuple[int, int]
) -> Optional[DerivedRegion]:
    """The region a missing control occupies, or None when nothing says.

    None is an answer. It means this screen's detections do not bound the gap.
    The caller should keep whatever coarse crop it already had. A rectangle
    built from fewer than two measurements is worth less.
    """
    width, height = size
    if width <= 0 or height <= 0 or region not in _ALL_REGIONS:
        return None

    boxes = [_bounds_of(element) for element in elements]
    if len(boxes) < 2:
        return None

    band = _hint_band(region, height)
    in_band = [box for box in boxes if box[3] > band[0] and box[1] < band[1]]
    pairs = _find_gaps(_merge_rows(in_band), height)
    if not pairs:
        logger.info("No gap in the %s band is bounded by two rows.", region)
        return None

    above, below = _select_nearest_gap(pairs, band)
    left, right, rule, samples = _measure_extents(boxes, above, below, width)
    crop = (
        max(0, left - PAD),
        max(0, above[3] - PAD),
        min(width, right + PAD),
        min(height, below[1] + PAD),
    )
    if crop[2] - crop[0] < MIN_REGION_SHARE * width:
        # Widening to the anchors did not reach a width a control could
        # occupy. The height is guaranteed by the gap floor, so only this
        # side can fail.
        return None
    return DerivedRegion(
        crop=crop,
        anchor_above=above,
        anchor_below=below,
        width_rule=rule,
        width_samples=samples,
    )


def derive_between(
    anchor_above: Optional[Box],
    anchor_below: Optional[Box],
    elements: Sequence[GUIElement],
    size: tuple[int, int],
) -> Optional[DerivedRegion]:
    """Build a crop from the two numbered controls named by the planner.

    The numbers are resolved to bounds before this function is called. Text is
    never matched, so the contract is unchanged for translated, icon-only, or
    unlabelled interfaces.
    """
    width, height = size
    if width <= 0 or height <= 0 or anchor_above is None or anchor_below is None:
        return None
    span = anchor_below[1] - anchor_above[3]
    floor = max(MIN_GAP_PIXELS, round(MIN_GAP_SHARE * height))
    ceiling = round(MAX_GAP_SHARE * height)
    if span < floor or span > ceiling:
        return None
    boxes = [_bounds_of(element) for element in elements]
    left, right, rule, samples = _measure_extents(
        boxes, anchor_above, anchor_below, width
    )
    crop = (
        max(0, left - PAD),
        max(0, anchor_above[3] - PAD),
        min(width, right + PAD),
        min(height, anchor_below[1] + PAD),
    )
    if crop[2] - crop[0] < MIN_REGION_SHARE * width:
        return None
    return DerivedRegion(
        crop=crop,
        anchor_above=anchor_above,
        anchor_below=anchor_below,
        width_rule=rule,
        width_samples=samples,
    )



def scale_for(crop: Box) -> int:
    """How far to magnify a crop before the detector sees it again.

    Driven by the crop, not fixed. A derived band is a fraction of the third
    it replaces, and the magnification that suited the third leaves it tiny.
    The binding side wins. A wide thin band then stays close to the image
    shapes the detector was trained on.
    """
    width, height = crop[2] - crop[0], crop[3] - crop[1]
    if width <= 0 or height <= 0:
        return MIN_SCALE
    long_side, short_side = max(width, height), min(width, height)
    fit = min(TARGET_LONG_SIDE / long_side, TARGET_SHORT_SIDE / short_side)
    return max(MIN_SCALE, min(MAX_SCALE, int(fit)))
