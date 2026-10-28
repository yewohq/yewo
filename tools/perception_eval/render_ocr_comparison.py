"""Render two whole-screen OCR runs as side-by-side text-box overlays."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from tools.perception_eval.corpus import MANIFEST_PATH, PerceptionCase, read_corpus
from tools.perception_eval.ocr_evaluation import (
    OCRCaseObservation,
    OCREvaluationRun,
    score_ocr,
)
from yewo.agent.perception.ocr import OCRLine

RUNS_ROOT = MANIFEST_PATH.parent / "screens" / "runs"
PANEL_MAX_WIDTH = 720
PANEL_MAX_HEIGHT = 1400
HEADER_HEIGHT = 62
PANEL_GAP = 12
BOX_COLORS = ((31, 119, 180), (255, 127, 14))
Font = ImageFont.ImageFont | ImageFont.FreeTypeFont


def render_ocr_comparison(
    left_run: OCREvaluationRun,
    right_run: OCREvaluationRun,
    output_directory: Path,
    manifest_path: Path = MANIFEST_PATH,
) -> list[Path]:
    """Render every corpus screenshot with each OCR engine's own boxes."""
    corpus = read_corpus(manifest_path)
    left_metrics = score_ocr(corpus, left_run)
    right_metrics = score_ocr(corpus, right_run)
    output_directory.mkdir(parents=True, exist_ok=False)

    comparison_paths: list[Path] = []
    for case, left, right in zip(
        corpus.cases,
        left_run.observations,
        right_run.observations,
        strict=True,
    ):
        comparison = _render_case(case, left_run, left, right_run, right)
        comparison_path = output_directory / f"{case.id}.png"
        comparison.save(comparison_path)
        comparison_paths.append(comparison_path)

    _render_overview(comparison_paths, output_directory / "overview.png")
    (output_directory / "easyocr.json").write_text(
        left_run.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (output_directory / "rapidocr.json").write_text(
        right_run.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "manifest": str(manifest_path),
        "left": {"ocr": left_run.ocr, "metrics": left_metrics.model_dump()},
        "right": {"ocr": right_run.ocr, "metrics": right_metrics.model_dump()},
        "images": [path.name for path in comparison_paths],
    }
    (output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return comparison_paths


def _render_case(
    case: PerceptionCase,
    left_run: OCREvaluationRun,
    left: OCRCaseObservation,
    right_run: OCREvaluationRun,
    right: OCRCaseObservation,
) -> Image.Image:
    with Image.open(case.screenshot) as image:
        source = image.convert("RGB")
    scale = min(PANEL_MAX_WIDTH / source.width, PANEL_MAX_HEIGHT / source.height)
    panel_size = (round(source.width * scale), round(source.height * scale))
    panels = (
        _render_panel(source, panel_size, scale, case.id, left_run.ocr, left, 0),
        _render_panel(source, panel_size, scale, case.id, right_run.ocr, right, 1),
    )
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
    case_id: str,
    identity: str,
    observation: OCRCaseObservation,
    color_index: int,
) -> Image.Image:
    panel = Image.new("RGB", (panel_size[0], panel_size[1] + HEADER_HEIGHT), "white")
    panel.paste(source.resize(panel_size, Image.Resampling.LANCZOS), (0, HEADER_HEIGHT))
    draw = ImageDraw.Draw(panel)
    header_font = _load_font(16)
    label_font = _load_font(11)
    draw.text((8, 6), f"{identity} · {case_id}", fill=(20, 20, 20), font=header_font)
    draw.text(
        (8, 33),
        f"lines {len(observation.result.lines)} · wall {observation.result.timing.wall_ms:.1f} ms",
        fill=(70, 70, 70),
        font=label_font,
    )
    for line in observation.result.lines:
        _draw_line(draw, line, scale, BOX_COLORS[color_index], label_font)
    return panel


def _draw_line(
    draw: ImageDraw.ImageDraw,
    line: OCRLine,
    scale: float,
    color: tuple[int, int, int],
    font: Font,
) -> None:
    points = [
        (round(point.x * scale), round(point.y * scale) + HEADER_HEIGHT) for point in line.box
    ]
    draw.line([*points, points[0]], fill=color, width=2, joint="curve")
    label = f"{line.text[:24]} {line.confidence:.2f}"
    x, y = points[0]
    text_box = draw.textbbox((x, y), label, font=font)
    label_height = text_box[3] - text_box[1] + 2
    label_width = min(text_box[2] - text_box[0] + 4, PANEL_MAX_WIDTH)
    label_y = max(HEADER_HEIGHT, y - label_height)
    draw.rectangle((x, label_y, x + label_width, label_y + label_height), fill=color)
    draw.text((x + 2, label_y), label, fill="white", font=font)


def _render_overview(comparison_paths: list[Path], output_path: Path) -> None:
    thumbnails: list[tuple[str, Image.Image]] = []
    for path in comparison_paths:
        with Image.open(path) as image:
            thumbnail = image.convert("RGB")
            thumbnail.thumbnail((700, 700), Image.Resampling.LANCZOS)
            thumbnails.append((path.stem, thumbnail.copy()))
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


def _load_font(size: int) -> Font:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _read_run(path: Path) -> OCREvaluationRun:
    return OCREvaluationRun.model_validate_json(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path, help="Saved OCR run for the left panel")
    parser.add_argument("right", type=Path, help="Saved OCR run for the right panel")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--output-root", type=Path, default=RUNS_ROOT)
    arguments = parser.parse_args()
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S%z")
    output_directory = arguments.output_root / timestamp
    paths = render_ocr_comparison(
        _read_run(arguments.left),
        _read_run(arguments.right),
        output_directory,
        arguments.manifest,
    )
    print(f"Rendered {len(paths)} OCR comparisons in {output_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
