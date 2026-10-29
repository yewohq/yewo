"""Builders for one fused perception frame, as the detector and OCR report it.

Every control surface test starts from a frame rather than a screenshot, so
these call the real constructors. A detector class that stops carrying a role,
or an OCR line that changes shape, breaks the build here instead of letting a
projection keep passing against a frame production no longer produces.
"""

from yewo.agent.perception.elements import BoundingBox, GUIElement
from yewo.agent.perception.fusion import PerceptionFrame, TextAssociation
from yewo.agent.perception.ocr import OCRLine, OCRPoint

DIGEST = "a" * 64


def detected(
    box: tuple[int, int, int, int],
    name: str = "View",
    text: str = "",
    roles: frozenset[str] | set[str] = frozenset(),
    confidence: float = 0.9,
    kind: str = "yolo_detection",
) -> GUIElement:
    """One element as the detector reports it, after fusion attached its text."""
    return GUIElement(
        element=name,
        type=kind,
        confidence=confidence,
        text=text,
        semantic_roles=frozenset(roles),
        bounding_box=BoundingBox(
            x_min=box[0], y_min=box[1], x_max=box[2], y_max=box[3]
        ),
    )


def read(box: tuple[int, int, int, int], text: str, confidence: float = 0.95) -> OCRLine:
    """One OCR line, from the axis-aligned box its four points describe."""
    left, top, right, bottom = box
    return OCRLine(
        box=(
            OCRPoint(x=left, y=top),
            OCRPoint(x=right, y=top),
            OCRPoint(x=right, y=bottom),
            OCRPoint(x=left, y=bottom),
        ),
        text=text,
        confidence=confidence,
    )


def frame(
    elements: tuple[GUIElement, ...] = (),
    text_lines: tuple[OCRLine, ...] = (),
    associations: tuple[TextAssociation, ...] = (),
) -> PerceptionFrame:
    """One fused observation of a screen."""
    return PerceptionFrame(
        image_digest=DIGEST,
        detector_identity="yewo-ui@test",
        ocr_identity="rapidocr@test",
        elements=elements,
        text_lines=text_lines,
        associations=associations,
    )
