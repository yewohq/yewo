"""The packaged Yewo mark shared by reports and generated screenshots."""

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
import re
from xml.etree import ElementTree

from PIL import ImageDraw


MOSS = (47, 58, 43)
SAND = (239, 227, 200)
CLAY = (201, 120, 74)
MARK_CROP = (18.0, 21.0, 52.0, 47.0)


@dataclass(frozen=True)
class MarkDefinition:
    """Vector data read from the packaged brand artifact."""

    paths: tuple[str, str]
    colors: tuple[str, str]
    stroke_width: float
    translate_y: float


@lru_cache(maxsize=1)
def mark_source() -> str:
    """Return the exact SVG artifact shipped with Yewo."""
    asset = resources.files(__package__).joinpath("yewo-mark-for-dark-bg.svg")
    return asset.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def mark_definition() -> MarkDefinition:
    """Read the paths and presentation values from the SVG artifact."""
    root = ElementTree.fromstring(mark_source())
    group = next(child for child in root if child.tag.endswith("g"))
    paths = tuple(child for child in group if child.tag.endswith("path"))
    translate = group.attrib.get("transform", "")
    match = re.fullmatch(r"translate\(0,(-?\d+(?:\.\d+)?)\)", translate)
    if len(paths) != 2 or match is None:
        raise ValueError("The packaged Yewo mark has an unsupported SVG structure")
    return MarkDefinition(
        paths=(paths[0].attrib["d"], paths[1].attrib["d"]),
        colors=(paths[0].attrib["stroke"], paths[1].attrib["stroke"]),
        stroke_width=float(paths[0].attrib["stroke-width"]),
        translate_y=float(match.group(1)),
    )


def mark_svg(
    width: int,
    height: int,
    *,
    foreground: str,
    accent: str,
    stroke_width: float,
) -> str:
    """Render the packaged mark paths as inline SVG."""
    mark = mark_definition()
    crop_x, crop_y, crop_width, crop_height = MARK_CROP
    return (
        f'<svg width="{width}" height="{height}" '
        f'viewBox="{crop_x:g} {crop_y:g} {crop_width:g} {crop_height:g}" '
        'aria-hidden="true"><g '
        f'transform="translate(0,{mark.translate_y:g})" fill="none" '
        'stroke-linecap="round" stroke-linejoin="round">'
        f'<path d="{mark.paths[0]}" stroke="{foreground}" stroke-width="{stroke_width:g}"/>'
        f'<path d="{mark.paths[1]}" stroke="{accent}" stroke-width="{stroke_width:g}"/>'
        "</g></svg>"
    )


def _path_points(path: str) -> list[tuple[float, float]]:
    """Sample the line and quadratic commands used by the Yewo mark."""
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
                t = step / 8
                inverse = 1 - t
                points.append(
                    (
                        inverse * inverse * start[0]
                        + 2 * inverse * t * control[0]
                        + t * t * end[0],
                        inverse * inverse * start[1]
                        + 2 * inverse * t * control[1]
                        + t * t * end[1],
                    )
                )
            current = end
        else:
            raise ValueError(f"Unsupported Yewo mark command: {command}")
    return points


def draw_mark(
    draw: ImageDraw.ImageDraw,
    left: int,
    top: int,
    size: int,
    *,
    foreground: tuple[int, int, int] = SAND,
    accent: tuple[int, int, int] = CLAY,
) -> None:
    """Draw the packaged Yewo mark into a square raster region."""
    mark = mark_definition()
    crop_x, crop_y, crop_width, crop_height = MARK_CROP
    scale = size / crop_width
    stroke = max(2, round(mark.stroke_width * scale))
    for path, color in zip(mark.paths, (foreground, accent), strict=True):
        points = [
            (
                left + (x - crop_x) * scale,
                top + (y + mark.translate_y - crop_y) * size / crop_height,
            )
            for x, y in _path_points(path)
        ]
        draw.line(points, fill=color, width=stroke, joint="curve")
        radius = stroke / 2
        for x, y in (points[0], points[-1]):
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
