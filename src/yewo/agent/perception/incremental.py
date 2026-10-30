"""Conservative change planning and OCR evidence reuse between screenshots."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from yewo.agent.perception.elements import BoundingBox, GUIElement
from yewo.agent.perception.ocr import OCRLine, OCRPoint, OCRResult, OCRTiming

PIXEL_DELTA_THRESHOLD = 2
MAX_CHANGED_FRACTION = 0.15
MAX_DIRTY_REGIONS = 8
DETECTOR_REUSE_SURFACE_FRACTION = 0.15
SURFACE_BOUNDARY_MARGIN = 4
OCR_PADDING = 24
OCR_MIN_WIDTH = 192
OCR_MIN_HEIGHT = 72
CHANGE_TILE_SIZE = 64


@dataclass(frozen=True)
class IncrementalPlan:
    """The model work justified by the pixels that changed."""

    regions: tuple[tuple[int, int, int, int], ...]
    changed_pixels: int
    changed_fraction: float
    reuse_detector: bool = False
    surface_roles: frozenset[str] = frozenset()
    full_scan_reason: str | None = None

    @property
    def ocr_input_pixels(self) -> int:
        return sum((right - left) * (bottom - top) for left, top, right, bottom in self.regions)


def plan_incremental_refresh(
    previous: Image.Image,
    current: Image.Image,
    elements: list[GUIElement],
    reusable_surface_roles: frozenset[str],
) -> IncrementalPlan:
    """Plan a partial refresh, falling back whenever containment is uncertain."""
    if previous.size != current.size:
        return IncrementalPlan((), 0, 1.0, full_scan_reason="screen_dimensions_changed")

    previous_array = np.asarray(previous.convert("RGB"), dtype=np.int16)
    current_array = np.asarray(current.convert("RGB"), dtype=np.int16)
    changed = np.max(np.abs(previous_array - current_array), axis=2) > PIXEL_DELTA_THRESHOLD
    changed_pixels = int(np.count_nonzero(changed))
    screen_pixels = max(changed.size, 1)
    changed_fraction = changed_pixels / screen_pixels
    if changed_pixels == 0:
        return IncrementalPlan((), 0, 0.0)
    if changed_fraction > MAX_CHANGED_FRACTION:
        return IncrementalPlan(
            (),
            changed_pixels,
            changed_fraction,
            full_scan_reason="changed_area_too_large",
        )

    raw_regions = _changed_regions(changed)
    if len(raw_regions) > MAX_DIRTY_REGIONS:
        return IncrementalPlan(
            (),
            changed_pixels,
            changed_fraction,
            full_scan_reason="too_many_dirty_regions",
        )

    width, height = current.size
    ocr_regions = tuple(_ocr_region(region, elements, width, height) for region in raw_regions)
    ocr_regions = _merge_regions(ocr_regions)
    surface_roles = _containing_surface_roles(
        raw_regions,
        elements,
        reusable_surface_roles,
        changed_pixels,
    )
    return IncrementalPlan(
        regions=ocr_regions,
        changed_pixels=changed_pixels,
        changed_fraction=changed_fraction,
        reuse_detector=bool(surface_roles),
        surface_roles=surface_roles,
    )


def merge_ocr_evidence(
    previous_lines: tuple[OCRLine, ...],
    refreshed: list[tuple[tuple[int, int, int, int], OCRResult]],
    engine_identity: str,
) -> OCRResult:
    """Keep untouched OCR lines and map crop readings into screen coordinates."""
    dirty_regions = [region for region, _ in refreshed]
    lines = [
        line
        for line in previous_lines
        if not any(_line_intersects(line, region) for region in dirty_regions)
    ]
    timings: list[OCRTiming] = []
    for (left, top, _, _), result in refreshed:
        lines.extend(_translate_line(line, left, top) for line in result.lines)
        timings.append(result.timing)
    lines.sort(key=_line_order)
    return OCRResult(
        engine_identity=engine_identity,
        lines=tuple(lines),
        timing=_sum_timings(timings),
    )


def rescale_ocr_result(result: OCRResult, coordinate_scale: float) -> OCRResult:
    """Map OCR coordinates from a calibrated crop back to native crop pixels."""
    if coordinate_scale <= 0:
        raise ValueError("OCR coordinate scale must be positive")
    return OCRResult(
        engine_identity=result.engine_identity,
        lines=tuple(_rescale_line(line, coordinate_scale) for line in result.lines),
        timing=result.timing,
    )


def _rescale_line(line: OCRLine, scale: float) -> OCRLine:
    points = tuple(
        OCRPoint(x=point.x * scale, y=point.y * scale) for point in line.box
    )
    return OCRLine(
        box=(points[0], points[1], points[2], points[3]),
        text=line.text,
        confidence=line.confidence,
    )


def _changed_regions(mask: np.ndarray) -> tuple[tuple[int, int, int, int], ...]:
    height, width = mask.shape
    rows = (height + CHANGE_TILE_SIZE - 1) // CHANGE_TILE_SIZE
    columns = (width + CHANGE_TILE_SIZE - 1) // CHANGE_TILE_SIZE
    dirty_tiles = {
        (row, column)
        for row in range(rows)
        for column in range(columns)
        if mask[
            row * CHANGE_TILE_SIZE : min((row + 1) * CHANGE_TILE_SIZE, height),
            column * CHANGE_TILE_SIZE : min((column + 1) * CHANGE_TILE_SIZE, width),
        ].any()
    }
    regions = []
    while dirty_tiles:
        seed = dirty_tiles.pop()
        component = {seed}
        pending = [seed]
        while pending:
            row, column = pending.pop()
            for neighbour in (
                (row - 1, column),
                (row + 1, column),
                (row, column - 1),
                (row, column + 1),
            ):
                if neighbour in dirty_tiles:
                    dirty_tiles.remove(neighbour)
                    component.add(neighbour)
                    pending.append(neighbour)
        top = min(row for row, _ in component) * CHANGE_TILE_SIZE
        bottom = min((max(row for row, _ in component) + 1) * CHANGE_TILE_SIZE, height)
        left = min(column for _, column in component) * CHANGE_TILE_SIZE
        right = min((max(column for _, column in component) + 1) * CHANGE_TILE_SIZE, width)
        ys, xs = np.nonzero(mask[top:bottom, left:right])
        regions.append(
            (
                left + int(xs.min()),
                top + int(ys.min()),
                left + int(xs.max()) + 1,
                top + int(ys.max()) + 1,
            )
        )
    return tuple(sorted(regions, key=lambda region: (region[1], region[0])))


def _ocr_region(
    changed: tuple[int, int, int, int],
    elements: list[GUIElement],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    candidates = [
        element.bounding_box
        for element in elements
        if not any(role.endswith("_region") for role in element.semantic_roles)
        and _regions_intersect(changed, _box_tuple(element.bounding_box))
    ]
    source = min(candidates, key=_box_area) if candidates else changed
    region = _box_tuple(source) if isinstance(source, BoundingBox) else source
    return _expand_region(region, width, height)


def _containing_surface_roles(
    regions: tuple[tuple[int, int, int, int], ...],
    elements: list[GUIElement],
    reusable_roles: frozenset[str],
    changed_pixels: int,
) -> frozenset[str]:
    candidates = []
    for element in elements:
        roles = element.semantic_roles & reusable_roles
        if not roles:
            continue
        box = _box_tuple(element.bounding_box)
        if not all(_strictly_contains(box, region) for region in regions):
            continue
        if changed_pixels / max(_box_area(box), 1) > DETECTOR_REUSE_SURFACE_FRACTION:
            continue
        candidates.append(roles)
    if not candidates:
        return frozenset()
    return frozenset.intersection(*candidates)


def _expand_region(
    region: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int]:
    left, top, right, bottom = region
    target_width = max(right - left + 2 * OCR_PADDING, OCR_MIN_WIDTH)
    target_height = max(bottom - top + 2 * OCR_PADDING, OCR_MIN_HEIGHT)
    center_x = (left + right) // 2
    center_y = (top + bottom) // 2
    left = max(0, center_x - target_width // 2)
    top = max(0, center_y - target_height // 2)
    right = min(width, left + target_width)
    bottom = min(height, top + target_height)
    left = max(0, right - target_width)
    top = max(0, bottom - target_height)
    return left, top, right, bottom


def _merge_regions(
    regions: tuple[tuple[int, int, int, int], ...],
) -> tuple[tuple[int, int, int, int], ...]:
    merged: list[tuple[int, int, int, int]] = []
    for region in regions:
        touching = [existing for existing in merged if _regions_intersect(existing, region)]
        if not touching:
            merged.append(region)
            continue
        for existing in touching:
            merged.remove(existing)
        merged.append(
            (
                min([region[0], *(item[0] for item in touching)]),
                min([region[1], *(item[1] for item in touching)]),
                max([region[2], *(item[2] for item in touching)]),
                max([region[3], *(item[3] for item in touching)]),
            )
        )
    return tuple(sorted(merged, key=lambda item: (item[1], item[0])))


def _strictly_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return (
        inner[0] >= outer[0] + SURFACE_BOUNDARY_MARGIN
        and inner[1] >= outer[1] + SURFACE_BOUNDARY_MARGIN
        and inner[2] <= outer[2] - SURFACE_BOUNDARY_MARGIN
        and inner[3] <= outer[3] - SURFACE_BOUNDARY_MARGIN
    )


def _regions_intersect(a, b) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _box_tuple(box: BoundingBox) -> tuple[int, int, int, int]:
    return box.x_min, box.y_min, box.x_max, box.y_max


def _box_area(box) -> int:
    if isinstance(box, BoundingBox):
        box = _box_tuple(box)
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _line_bounds(line: OCRLine) -> tuple[float, float, float, float]:
    return line.bounds


def _line_intersects(line: OCRLine, region: tuple[int, int, int, int]) -> bool:
    return _regions_intersect(_line_bounds(line), region)


def _translate_line(line: OCRLine, offset_x: int, offset_y: int) -> OCRLine:
    points = tuple(OCRPoint(x=point.x + offset_x, y=point.y + offset_y) for point in line.box)
    return OCRLine(
        box=(points[0], points[1], points[2], points[3]),
        text=line.text,
        confidence=line.confidence,
    )


def _line_order(line: OCRLine) -> tuple[float, float]:
    left, top, _, _ = _line_bounds(line)
    return top, left


def _sum_timings(timings: list[OCRTiming]) -> OCRTiming:
    def total(field: str) -> float | None:
        values = [getattr(timing, field) for timing in timings]
        present = [value for value in values if value is not None]
        return sum(present) if present else None

    return OCRTiming(
        detection_ms=total("detection_ms"),
        classification_ms=total("classification_ms"),
        recognition_ms=total("recognition_ms"),
        engine_total_ms=sum(timing.engine_total_ms for timing in timings),
        wall_ms=sum(timing.wall_ms for timing in timings),
    )
