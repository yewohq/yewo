"""Reading the screen without detecting anything on it.

Describe mode names its target rather than choosing one from a list, so no
detector runs and no element list is built. What the loop still needs is the
frame, its hash, and a scan record saying a pass happened.
"""

import io
import logging
from typing import Optional

import imagehash
from PIL import Image

from yewo.agent.perception.elements import GUIElement, ScanInfo, reason_for

logger = logging.getLogger(__name__)


class BlindPerception:
    """Perception for a mode that places targets by name.

    Answers the same three things `ScreenPerception` answers, so the loop
    cannot tell which it was given. The element list is always empty, and the
    annotated image is the raw one, because nothing was drawn on it.
    """

    def __init__(self, device_session, mode: str = "describe"):
        self.device_session = device_session
        self.mode = mode
        self.current_hash: Optional["imagehash.ImageHash"] = None
        self.last_scan_info: Optional[ScanInfo] = None
        self._prev_hash: Optional["imagehash.ImageHash"] = None

    def reset(self) -> None:
        self.current_hash = None
        self._prev_hash = None

    def perceive(self) -> tuple[list[GUIElement], Image.Image, Image.Image]:
        png = self.device_session.device_control.get_screenshot()
        raw = Image.open(io.BytesIO(png))

        self._prev_hash, self.current_hash = self.current_hash, imagehash.phash(raw, hash_size=16)
        distance = self.current_hash - self._prev_hash if self._prev_hash is not None else None
        width, height = raw.size
        self.last_scan_info = ScanInfo(
            mode=self.mode,
            reason=reason_for(distance),
            phash=str(self.current_hash),
            phash_distance=distance,
            screen_width=width,
            screen_height=height,
            element_count=0,
        )
        return [], raw, raw


def for_grounding(
    grounding: str,
    device_session,
    model_reference: str | None = None,
):
    """The perception a grounding mode needs.

    A mode that names its target reads the screen and detects nothing. Every
    other mode chooses from detections, so it needs them found.
    """
    from yewo.agent.grounding import GROUNDING_DESCRIBE
    from yewo.agent.perception.screen import ScreenPerception

    from yewo.agent.grounding import GROUNDING_COORDINATES

    if grounding in (GROUNDING_COORDINATES, GROUNDING_DESCRIBE):
        return BlindPerception(device_session, grounding)
    return ScreenPerception(device_session, model_reference)
