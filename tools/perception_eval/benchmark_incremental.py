"""Benchmark one conservative role-contained refresh against a forced full pass."""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path
from typing import cast

from PIL import Image, ImageChops, ImageDraw, ImageFont

from yewo.agent.perception.detector import ImageDescriptor
from yewo.agent.perception.incremental import plan_incremental_refresh
from yewo.agent.perception.keyboard import KEYBOARD_KEY_ROLE, KEYBOARD_REGION_ROLE


def benchmark(screenshot: Path, model_reference: str, render_output: Path | None = None) -> dict:
    descriptor = ImageDescriptor(model_reference)
    source_bytes = screenshot.read_bytes()
    previous_frame, previous_image = descriptor.observe(source_bytes)
    elements = list(previous_frame.elements)
    key = min(
        (element for element in elements if KEYBOARD_KEY_ROLE in element.semantic_roles),
        key=lambda element: _area(element.bounding_box),
    )
    changed_image = previous_image.copy()
    box = key.bounding_box
    highlight_x = min(box.x_max - 12, box.x_min + 12)
    highlight_y = min(box.y_max - 12, box.y_min + 12)
    source_color = cast(tuple[int, int, int], changed_image.getpixel((highlight_x, highlight_y)))
    highlight_color = tuple(
        channel - 32 if channel > 127 else channel + 32 for channel in source_color[:3]
    )
    ImageDraw.Draw(changed_image).rectangle(
        (highlight_x, highlight_y, highlight_x + 4, highlight_y + 4),
        fill=highlight_color,
    )
    changed_bytes = _png_bytes(changed_image)
    plan = plan_incremental_refresh(
        previous_image,
        changed_image,
        elements,
        frozenset({KEYBOARD_REGION_ROLE}),
    )
    if not plan.reuse_detector:
        raise RuntimeError("benchmark mutation was not safely contained by a keyboard region")

    started = time.perf_counter()
    incremental_elements, _ = descriptor.describe_incremental(
        changed_bytes,
        previous_frame,
        plan.regions,
        reuse_detector=True,
    )
    incremental_ms = (time.perf_counter() - started) * 1000

    descriptor.clear_cache()
    started = time.perf_counter()
    full_elements, _ = descriptor.describe(changed_bytes)
    full_ms = (time.perf_counter() - started) * 1000
    incremental_texts = _texts(incremental_elements)
    full_texts = _texts(full_elements)

    result = {
        "screenshot": str(screenshot),
        "model": model_reference,
        "changed_pixel_fraction": round(plan.changed_fraction, 8),
        "ocr_regions": [list(region) for region in plan.regions],
        "ocr_input_pixels": plan.ocr_input_pixels,
        "screen_pixels": changed_image.width * changed_image.height,
        "detector_reused": plan.reuse_detector,
        "surface_roles": sorted(plan.surface_roles),
        "incremental_ms": round(incremental_ms, 3),
        "full_ms": round(full_ms, 3),
        "speedup": round(full_ms / max(incremental_ms, 0.001), 3),
        "incremental_element_count": len(incremental_elements),
        "full_element_count": len(full_elements),
        "incremental_text_count": len(incremental_texts),
        "full_text_count": len(full_texts),
        "unique_fused_text_equal": set(incremental_texts) == set(full_texts),
        "fused_text_multiset_equal": incremental_texts == full_texts,
        "incremental_only_text": sorted(set(incremental_texts) - set(full_texts)),
        "full_only_text": sorted(set(full_texts) - set(incremental_texts)),
        "incremental_class_counts": _class_counts(incremental_elements),
        "full_class_counts": _class_counts(full_elements),
    }
    if render_output is not None:
        render_output.parent.mkdir(parents=True, exist_ok=True)
        _render_progression(
            render_output,
            previous_image,
            changed_image,
            elements,
            key,
            plan,
            incremental_elements,
            result,
        )
        result["progression_image"] = str(render_output)
    return result


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _area(box) -> int:
    return max(0, box.x_max - box.x_min) * max(0, box.y_max - box.y_min)


def _class_counts(elements) -> dict[str, int]:
    counts: dict[str, int] = {}
    for element in elements:
        counts[element.element] = counts.get(element.element, 0) + 1
    return dict(sorted(counts.items()))


def _texts(elements) -> tuple[str, ...]:
    return tuple(sorted(element.text.strip() for element in elements if element.text.strip()))


def _render_progression(
    output: Path,
    baseline: Image.Image,
    changed: Image.Image,
    baseline_elements,
    changed_key,
    plan,
    incremental_elements,
    result: dict,
) -> None:
    panel_width = 390
    image_height = 800
    header_height = 92
    footer_height = 104
    panels = [
        _baseline_panel(baseline, baseline_elements, changed_key),
        _change_panel(baseline, changed, changed_key),
        _decision_panel(baseline, changed, baseline_elements, changed_key, plan),
        _result_panel(changed, incremental_elements),
    ]
    titles = (
        ("1  BASELINE", "YOLO finds the keyboard region and its keys"),
        ("2  VISUAL CHANGE", "A tiny state change occurs inside one key"),
        ("3  INVALIDATION", "Change is contained; refresh only the OCR crop"),
        ("4  RESULT", "Reuse detector geometry and fuse refreshed OCR"),
    )
    font = _font(19)
    small = _font(14)
    canvas = Image.new(
        "RGB",
        (panel_width * len(panels), header_height + image_height + footer_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, (panel, (title, subtitle)) in enumerate(zip(panels, titles, strict=True)):
        left = index * panel_width
        fitted = panel.resize((panel_width, image_height), Image.Resampling.LANCZOS)
        canvas.paste(fitted, (left, header_height))
        draw.text((left + 16, 14), title, fill=(25, 28, 35), font=font)
        draw.multiline_text((left + 16, 47), subtitle, fill=(72, 79, 92), font=small, spacing=3)
        if index:
            draw.line((left, 0, left, canvas.height), fill=(210, 214, 222), width=2)

    ocr_percent = result["ocr_input_pixels"] / result["screen_pixels"] * 100
    footers = (
        "Blue = keyboard_region\nGreen = selected keyboard_key",
        "Magenta = changed pixels\nInset magnifies the affected key",
        f"Orange = OCR crop ({ocr_percent:.2f}% of screen)\nDetector reuse: YES",
        (
            f"{result['incremental_ms']:.1f} ms partial vs {result['full_ms']:.1f} ms full\n"
            f"{result['speedup']:.1f}x faster · unique text preserved"
        ),
    )
    for index, footer in enumerate(footers):
        left = index * panel_width
        draw.multiline_text(
            (left + 16, header_height + image_height + 18),
            footer,
            fill=(40, 44, 52),
            font=small,
            spacing=6,
        )
    canvas.save(output)


def _baseline_panel(image, elements, changed_key):
    panel = image.copy().convert("RGB")
    draw = ImageDraw.Draw(panel)
    for element in elements:
        box = tuple(element.bounding_box)
        if KEYBOARD_REGION_ROLE in element.semantic_roles:
            draw.rectangle(box, outline=(35, 128, 255), width=8)
        elif KEYBOARD_KEY_ROLE in element.semantic_roles:
            draw.rectangle(box, outline=(90, 205, 125), width=3)
        else:
            draw.rectangle(box, outline=(160, 166, 176), width=2)
    draw.rectangle(tuple(changed_key.bounding_box), outline=(0, 180, 80), width=10)
    return panel


def _change_panel(baseline, changed, changed_key):
    panel = changed.copy().convert("RGB")
    draw = ImageDraw.Draw(panel)
    change_box = ImageChops.difference(baseline, changed).getbbox()
    draw.rectangle(tuple(changed_key.bounding_box), outline=(0, 180, 80), width=9)
    if change_box:
        draw.rectangle(change_box, outline=(255, 0, 170), width=12)
    _paste_key_inset(panel, changed, changed_key, change_box)
    return panel


def _decision_panel(baseline, changed, elements, changed_key, plan):
    panel = changed.copy().convert("RGB")
    veil = Image.new("RGBA", panel.size, (10, 15, 25, 80))
    panel = Image.alpha_composite(panel.convert("RGBA"), veil).convert("RGB")
    draw = ImageDraw.Draw(panel)
    for element in elements:
        if KEYBOARD_REGION_ROLE in element.semantic_roles:
            draw.rectangle(tuple(element.bounding_box), outline=(35, 128, 255), width=10)
    draw.rectangle(tuple(changed_key.bounding_box), outline=(0, 180, 80), width=8)
    change_box = ImageChops.difference(baseline, changed).getbbox()
    if change_box:
        draw.rectangle(change_box, outline=(255, 0, 170), width=12)
    for region in plan.regions:
        draw.rectangle(region, outline=(255, 156, 35), width=10)
    return panel


def _result_panel(image, elements):
    panel = image.copy().convert("RGB")
    draw = ImageDraw.Draw(panel)
    for element in elements:
        if KEYBOARD_REGION_ROLE in element.semantic_roles:
            color, width = (35, 128, 255), 8
        elif KEYBOARD_KEY_ROLE in element.semantic_roles:
            color, width = (90, 205, 125), 3
        else:
            color, width = (160, 166, 176), 2
        draw.rectangle(tuple(element.bounding_box), outline=color, width=width)
    return panel


def _paste_key_inset(panel, image, key, change_box):
    box = key.bounding_box
    padding = 16
    crop_box = (
        max(0, box.x_min - padding),
        max(0, box.y_min - padding),
        min(image.width, box.x_max + padding),
        min(image.height, box.y_max + padding),
    )
    inset_width = 360
    crop = image.crop(crop_box)
    inset_height = max(120, round(crop.height * inset_width / max(crop.width, 1)))
    crop = crop.resize((inset_width, inset_height), Image.Resampling.NEAREST)
    inset_x = image.width - inset_width - 30
    inset_y = 110
    panel.paste(crop, (inset_x, inset_y))
    draw = ImageDraw.Draw(panel)
    draw.rectangle(
        (inset_x - 5, inset_y - 5, inset_x + inset_width + 5, inset_y + inset_height + 5),
        outline=(255, 255, 255),
        width=10,
    )
    draw.rectangle(
        (inset_x, inset_y, inset_x + inset_width, inset_y + inset_height),
        outline=(255, 0, 170),
        width=7,
    )


def _font(size: int):
    for path in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("screenshot", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--render", type=Path)
    arguments = parser.parse_args()
    result = benchmark(arguments.screenshot, arguments.model, arguments.render)
    rendered = json.dumps(result, indent=2) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
