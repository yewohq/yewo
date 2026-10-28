"""Render two saved perception runs as side-by-side detection overlays."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from tools.perception_eval.corpus import MANIFEST_PATH, PerceptionCase, read_corpus
from tools.perception_eval.metrics import (
    STRUCTURAL_ELEMENTS,
    CaseObservation,
    EvaluationRun,
    ObservedElement,
    score,
)

RUNS_ROOT = MANIFEST_PATH.parent / "screens" / "runs"
PANEL_MAX_WIDTH = 720
PANEL_MAX_HEIGHT = 1400
HEADER_HEIGHT = 58
PANEL_GAP = 12
DETECTION_COLORS = ((31, 119, 180), (255, 127, 14))
TARGET_COVERED_COLOR = (20, 150, 70)
TARGET_MISSED_COLOR = (220, 35, 45)


def render_comparison(
    left_run: EvaluationRun,
    right_run: EvaluationRun,
    output_directory: Path,
    manifest_path: Path = MANIFEST_PATH,
) -> list[Path]:
    """Render every corpus case and return the created comparison images."""
    corpus = read_corpus(manifest_path)
    left_metrics = score(corpus, left_run)
    right_metrics = score(corpus, right_run)
    left_by_case_id = {observation.case_id: observation for observation in left_run.observations}
    right_by_case_id = {observation.case_id: observation for observation in right_run.observations}
    output_directory.mkdir(parents=True, exist_ok=False)

    comparison_paths: list[Path] = []
    for case in corpus.cases:
        comparison = _render_case_comparison(
            case,
            left_run,
            left_by_case_id[case.id],
            right_run,
            right_by_case_id[case.id],
        )
        comparison_path = output_directory / f"{case.id}.png"
        comparison.save(comparison_path)
        comparison_paths.append(comparison_path)

    _render_overview(comparison_paths, output_directory / "overview.png")
    summary = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "manifest": str(manifest_path),
        "left": {
            "detector": left_run.detector,
            "inference_image_size": left_run.inference_image_size,
            "metrics": left_metrics.model_dump(),
        },
        "right": {
            "detector": right_run.detector,
            "inference_image_size": right_run.inference_image_size,
            "metrics": right_metrics.model_dump(),
        },
        "images": [path.name for path in comparison_paths],
    }
    (output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return comparison_paths


def _render_case_comparison(
    case: PerceptionCase,
    left_run: EvaluationRun,
    left_observation: CaseObservation,
    right_run: EvaluationRun,
    right_observation: CaseObservation,
) -> Image.Image:
    source = Image.open(case.screenshot).convert("RGB")
    scale = min(PANEL_MAX_WIDTH / source.width, PANEL_MAX_HEIGHT / source.height)
    panel_size = (round(source.width * scale), round(source.height * scale))
    panels = [
        _render_panel(source, panel_size, scale, case, left_run, left_observation, 0),
        _render_panel(source, panel_size, scale, case, right_run, right_observation, 1),
    ]
    canvas = Image.new(
        "RGB",
        (panel_size[0] * 2 + PANEL_GAP, panel_size[1] + HEADER_HEIGHT),
        "white",
    )
    canvas.paste(panels[0], (0, 0))
    canvas.paste(panels[1], (panel_size[0] + PANEL_GAP, 0))
    return canvas


def _render_panel(
    source: Image.Image,
    panel_size: tuple[int, int],
    scale: float,
    case: PerceptionCase,
    run: EvaluationRun,
    observation: CaseObservation,
    color_index: int,
) -> Image.Image:
    panel = Image.new("RGB", (panel_size[0], panel_size[1] + HEADER_HEIGHT), "white")
    resized = source.resize(panel_size, Image.Resampling.LANCZOS)
    panel.paste(resized, (0, HEADER_HEIGHT))
    draw = ImageDraw.Draw(panel)
    header_font = _load_font(16)
    label_font = _load_font(11)
    target_font = _load_font(12)
    detector_name = _short_detector_name(run.detector)
    draw.text(
        (8, 7),
        f"{detector_name} · imgsz {run.inference_image_size} · {case.id}",
        fill=(20, 20, 20),
        font=header_font,
    )
    draw.text(
        (8, 31),
        f"boxes {len(observation.elements)} · marks {observation.mark_count} · "
        f"keyboard {observation.visual_keyboard_visible}",
        fill=(70, 70, 70),
        font=label_font,
    )
    detection_color = DETECTION_COLORS[color_index]
    for element in observation.elements:
        _draw_detection(draw, element, scale, detection_color, label_font)
    for target in case.target_points:
        is_covered = any(
            element.element not in STRUCTURAL_ELEMENTS
            and element.bounding_box.contains(target)
            for element in observation.elements
        )
        _draw_target(draw, target.name, target.x, target.y, scale, is_covered, target_font)
    return panel


def _draw_detection(
    draw: ImageDraw.ImageDraw,
    element: ObservedElement,
    scale: float,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    box = element.bounding_box
    coordinates = (
        round(box.x_min * scale),
        round(box.y_min * scale) + HEADER_HEIGHT,
        round(box.x_max * scale),
        round(box.y_max * scale) + HEADER_HEIGHT,
    )
    draw.rectangle(coordinates, outline=color, width=2)
    text_suffix = f" {element.text[:22]}" if element.text else ""
    label = f"{element.element}{text_suffix} {element.confidence:.2f}"
    label_box = draw.textbbox((coordinates[0], coordinates[1]), label, font=font)
    label_y = max(HEADER_HEIGHT, coordinates[1] - (label_box[3] - label_box[1]) - 2)
    background = (
        coordinates[0],
        label_y,
        coordinates[0] + label_box[2] - label_box[0] + 4,
        label_y + label_box[3] - label_box[1] + 2,
    )
    draw.rectangle(background, fill=color)
    draw.text((coordinates[0] + 2, label_y), label, fill="white", font=font)


def _draw_target(
    draw: ImageDraw.ImageDraw,
    name: str,
    x: int,
    y: int,
    scale: float,
    is_covered: bool,
    font: ImageFont.ImageFont,
) -> None:
    center_x = round(x * scale)
    center_y = round(y * scale) + HEADER_HEIGHT
    color = TARGET_COVERED_COLOR if is_covered else TARGET_MISSED_COLOR
    radius = 6
    draw.ellipse(
        (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
        outline=color,
        width=3,
    )
    draw.line((center_x - 9, center_y, center_x + 9, center_y), fill=color, width=2)
    draw.line((center_x, center_y - 9, center_x, center_y + 9), fill=color, width=2)
    draw.text((center_x + 8, center_y + 4), name, fill=color, font=font, stroke_width=2, stroke_fill="white")


def _render_overview(comparison_paths: list[Path], output_path: Path) -> None:
    thumbnails: list[tuple[str, Image.Image]] = []
    for comparison_path in comparison_paths:
        image = Image.open(comparison_path).convert("RGB")
        image.thumbnail((700, 700), Image.Resampling.LANCZOS)
        thumbnails.append((comparison_path.stem, image.copy()))
    font = _load_font(16)
    cell_width = 720
    cell_height = max(image.height for _, image in thumbnails) + 42
    columns = 2
    rows = (len(thumbnails) + columns - 1) // columns
    overview = Image.new("RGB", (cell_width * columns, cell_height * rows), "white")
    draw = ImageDraw.Draw(overview)
    for index, (case_id, thumbnail) in enumerate(thumbnails):
        x = (index % columns) * cell_width + 10
        y = (index // columns) * cell_height + 8
        draw.text((x, y), case_id, fill=(20, 20, 20), font=font)
        overview.paste(thumbnail, (x, y + 26))
    overview.save(output_path)


def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _short_detector_name(detector: str) -> str:
    repository = detector.split("@", maxsplit=1)[0]
    return repository.removeprefix("hf://").rsplit("/", maxsplit=1)[-1]


def _read_run(path: Path) -> EvaluationRun:
    return EvaluationRun.model_validate_json(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path, help="Saved run for the left panel")
    parser.add_argument("right", type=Path, help="Saved run for the right panel")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--output-root", type=Path, default=RUNS_ROOT)
    arguments = parser.parse_args()
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S%z")
    output_directory = arguments.output_root / timestamp
    comparison_paths = render_comparison(
        _read_run(arguments.left),
        _read_run(arguments.right),
        output_directory,
        arguments.manifest,
    )
    print(f"Rendered {len(comparison_paths)} comparisons in {output_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
