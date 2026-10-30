"""Canonical Yewo colors and packaged mark."""

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
import re
from xml.etree import ElementTree

from PIL import ImageDraw


MOSS = (47, 58, 43)
SAND = (239, 227, 200)
CLAY = (201, 120, 74)
_MARK_CROP = (18.0, 21.0, 52.0, 47.0)


@dataclass(frozen=True)
class _MarkDefinition:
    paths: tuple[str, str]
    stroke_width: float
    translate_y: float


@lru_cache(maxsize=1)
def _mark_definition() -> _MarkDefinition:
    source = resources.files(__package__).joinpath("yewo-mark-for-dark-bg.svg")
    root = ElementTree.fromstring(source.read_text(encoding="utf-8"))
    group = next(child for child in root if child.tag.endswith("g"))
    paths = tuple(child for child in group if child.tag.endswith("path"))
    translation = group.attrib.get("transform", "")
    match = re.fullmatch(r"translate\(0,(-?\d+(?:\.\d+)?)\)", translation)
    if len(paths) != 2 or match is None:
        raise ValueError("The packaged Yewo mark has an unsupported SVG structure.")
    return _MarkDefinition(
        paths=(paths[0].attrib["d"], paths[1].attrib["d"]),
        stroke_width=float(paths[0].attrib["stroke-width"]),
        translate_y=float(match.group(1)),
    )


def _path_points(path: str) -> list[tuple[float, float]]:
    tokens = re.findall(r"[MLQHVZ]|-?\d+(?:\.\d+)?", path)
    points: list[tuple[float, float]] = []
    current = (0.0, 0.0)
    index = 0
    while index < len(tokens):
        command = tokens[index]
        index += 1
        if command in {"M", "L"}:
            current = (float(tokens[index]), float(tokens[index + 1]))
            index += 2
            points.append(current)
        elif command == "H":
            current = (float(tokens[index]), current[1])
            index += 1
            points.append(current)
        elif command == "V":
            current = (current[0], float(tokens[index]))
            index += 1
            points.append(current)
        elif command == "Q":
            control = (float(tokens[index]), float(tokens[index + 1]))
            end = (float(tokens[index + 2]), float(tokens[index + 3]))
            index += 4
            start = current
            for step in range(1, 9):
                ratio = step / 8
                inverse = 1 - ratio
                points.append(
                    (
                        inverse * inverse * start[0]
                        + 2 * inverse * ratio * control[0]
                        + ratio * ratio * end[0],
                        inverse * inverse * start[1]
                        + 2 * inverse * ratio * control[1]
                        + ratio * ratio * end[1],
                    )
                )
            current = end
        else:
            raise ValueError(f"Unsupported Yewo mark command: {command}")
    return points


def draw_mark(
    drawing: ImageDraw.ImageDraw,
    left: int,
    top: int,
    size: int,
) -> None:
    """Draw the canonical Yewo mark in a square raster region."""
    mark = _mark_definition()
    crop_x, crop_y, crop_width, crop_height = _MARK_CROP
    scale = size / crop_width
    stroke = max(2, round(mark.stroke_width * scale))
    for path, color in zip(mark.paths, (SAND, CLAY), strict=True):
        points = [
            (
                left + (x - crop_x) * scale,
                top + (y + mark.translate_y - crop_y) * size / crop_height,
            )
            for x, y in _path_points(path)
        ]
        drawing.line(points, fill=color, width=stroke, joint="curve")
        radius = stroke / 2
        for x, y in (points[0], points[-1]):
            drawing.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


__all__ = ["CLAY", "MOSS", "SAND", "draw_mark"]
