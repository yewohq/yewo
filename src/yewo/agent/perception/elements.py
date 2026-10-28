"""What perception gives the rest of the agent: where things are, what they are,
and what the pass that found them cost.

The contract lives here, not in `detector`, so that no producer owns it.
"""

from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel, Field

NAVIGATION_THRESHOLD = 80
"""pHash-256 distance above which a frame is a screen the run just arrived at."""

FIRST_PASS = "first_pass"
NAVIGATION = "navigation"
LAYOUT_SHIFT = "layout_shift"
OCR_REFRESH = "ocr_refresh"
STABLE = "stable"

ARRIVALS = (FIRST_PASS, NAVIGATION, LAYOUT_SHIFT)
"""The reasons that mean a pass found a screen the run was not already on.

Kept beside the names rather than copied by each consumer. A consumer holding
its own list cannot see a producer report a reason the list does not have, and
the audit that reason was meant to trigger then never runs.
"""


def reason_for(distance: Optional[int]) -> str:
    """Why a pass happened, as far as one frame and the last one can say.

    Distance alone cannot tell a reflow from a stable screen, so nothing here
    answers `layout_shift`. That needs the elements, and a mode that detects
    none has none to compare.
    """
    if distance is None:
        return FIRST_PASS
    return NAVIGATION if distance >= NAVIGATION_THRESHOLD else STABLE


class BoundingBox(BaseModel):
    """A detected element's box in screenshot pixels, top-left origin."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def __iter__(self):
        """Unpacks in the order that PIL's crop expects."""
        yield self.x_min
        yield self.y_min
        yield self.x_max
        yield self.y_max


class GUIElement(BaseModel):
    element: str
    bounding_box: BoundingBox
    confidence: float
    type: str
    text: str = ""
    semantic_roles: frozenset[str] = Field(default_factory=frozenset)


@dataclass
class ScanInfo:
    """What one pass over the screen cost and found."""

    mode: str
    reason: str
    phash: str
    phash_distance: Optional[int] = None
    screen_width: int = 0
    screen_height: int = 0
    element_count: int = 0
    matched: Optional[int] = None
    new: Optional[int] = None
    removed: Optional[int] = None
    match_rate: Optional[float] = None
    dirty_ocr_count: Optional[int] = None
    uncovered_mae: Optional[float] = None
    keyboard_native_visible: Optional[bool] = None
    keyboard_visual_visible: Optional[bool] = None
    keyboard_effective_visible: Optional[bool] = None
    keyboard_owned_count: int = 0
    keyboard_state_disagrees: bool = False
    keyboard_targets_expanded: bool = False
    detector_reused: bool = False
    ocr_input_regions: int = 0
    ocr_input_pixels: int = 0
    changed_pixel_fraction: Optional[float] = None
    incremental_surface_roles: tuple[str, ...] = ()
    full_scan_fallback: Optional[str] = None
    thresholds: dict[str, float | int] = field(default_factory=dict)
