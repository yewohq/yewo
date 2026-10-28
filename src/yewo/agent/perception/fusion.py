"""Geometry-only fusion of independent UI detections and OCR observations."""

from __future__ import annotations

import math
from collections import defaultdict

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yewo.agent.perception.elements import BoundingBox, GUIElement
from yewo.agent.perception.ocr import OCRLine, OCRResult

MINIMUM_LINE_COVERAGE = 0.5
FUSION_VERSION = 1


class TextAssociation(BaseModel):
    """One OCR line geometrically associated with one detected UI element."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    element_index: int = Field(ge=0)
    line_index: int = Field(ge=0)
    line_coverage: float = Field(ge=0, le=1)


class PerceptionFrame(BaseModel):
    """Independent model evidence and its derived spatial associations."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = FUSION_VERSION
    image_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    detector_identity: str = Field(min_length=1)
    ocr_identity: str = Field(min_length=1)
    elements: tuple[GUIElement, ...]
    text_lines: tuple[OCRLine, ...]
    associations: tuple[TextAssociation, ...]

    @model_validator(mode="after")
    def _validate_association_indices(self) -> "PerceptionFrame":
        for association in self.associations:
            if association.element_index >= len(self.elements):
                raise ValueError("text association refers to a missing element")
            if association.line_index >= len(self.text_lines):
                raise ValueError("text association refers to a missing OCR line")
        return self

    def unmatched_text(self) -> tuple[str, ...]:
        """OCR text that remains evidence but is not attached to a UI target."""
        associated = {association.line_index for association in self.associations}
        return tuple(
            line.text.strip()
            for index, line in enumerate(self.text_lines)
            if index not in associated and line.text.strip()
        )

    def ocr_redaction_evidence(self) -> tuple[GUIElement, ...]:
        """Represent OCR regions only for secret masking, never for targeting."""
        evidence: list[GUIElement] = []
        for line in self.text_lines:
            left, top, right, bottom = _line_bounds(line)
            evidence.append(
                GUIElement(
                    element="ScreenTextEvidence",
                    bounding_box=BoundingBox(
                        x_min=int(left),
                        y_min=int(top),
                        x_max=math.ceil(right),
                        y_max=math.ceil(bottom),
                    ),
                    confidence=line.confidence,
                    type="ocr_evidence",
                    text=line.text,
                )
            )
        return tuple(evidence)


def fuse_perception(
    elements: list[GUIElement],
    ocr_result: OCRResult,
    image_digest: str,
    detector_identity: str,
) -> PerceptionFrame:
    """Attach OCR text by geometry without consulting detector class names."""
    fused_elements = [element.model_copy(deep=True) for element in elements]
    associations: list[TextAssociation] = []
    lines_by_element: dict[int, list[tuple[int, OCRLine]]] = defaultdict(list)

    for line_index, line in enumerate(ocr_result.lines):
        match = _best_element(line, fused_elements)
        if match is None:
            continue
        element_index, coverage = match
        associations.append(
            TextAssociation(
                element_index=element_index,
                line_index=line_index,
                line_coverage=round(coverage, 4),
            )
        )
        lines_by_element[element_index].append((line_index, line))

    for element_index, associated_lines in lines_by_element.items():
        associated_lines.sort(key=lambda item: _reading_order(item[1]))
        texts = [line.text.strip() for _, line in associated_lines if line.text.strip()]
        fused_elements[element_index].text = " ".join(dict.fromkeys(texts))

    return PerceptionFrame(
        image_digest=image_digest,
        detector_identity=detector_identity,
        ocr_identity=ocr_result.engine_identity,
        elements=tuple(fused_elements),
        text_lines=ocr_result.lines,
        associations=tuple(associations),
    )


def _best_element(
    line: OCRLine,
    elements: list[GUIElement],
) -> tuple[int, float] | None:
    line_left, line_top, line_right, line_bottom = _line_bounds(line)
    line_area = max(0.0, line_right - line_left) * max(0.0, line_bottom - line_top)
    if line_area == 0:
        return None

    candidates: list[tuple[int, float, int]] = []
    for index, element in enumerate(elements):
        box = element.bounding_box
        intersection_width = max(0.0, min(line_right, box.x_max) - max(line_left, box.x_min))
        intersection_height = max(0.0, min(line_bottom, box.y_max) - max(line_top, box.y_min))
        coverage = intersection_width * intersection_height / line_area
        if coverage < MINIMUM_LINE_COVERAGE:
            continue
        element_area = max(0, box.x_max - box.x_min) * max(0, box.y_max - box.y_min)
        candidates.append((element_area, -coverage, index))

    if not candidates:
        return None
    _, negative_coverage, index = min(candidates)
    return index, -negative_coverage


def _line_bounds(line: OCRLine) -> tuple[float, float, float, float]:
    xs = [point.x for point in line.box]
    ys = [point.y for point in line.box]
    return min(xs), min(ys), max(xs), max(ys)


def _reading_order(line: OCRLine) -> tuple[float, float]:
    left, top, _, _ = _line_bounds(line)
    return top, left
