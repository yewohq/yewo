"""Capture the current detector and OCR output for the frozen corpus.

This command is an explicit integration check. Unit tests do not call it.
It loads local vision models but never opens a device or calls a paid service.
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

from yewo.agent.grounding import marks
from yewo.agent.perception import keyboard
from yewo.agent.perception.detector import ImageDescriptor

from tools.perception_eval.corpus import MANIFEST_PATH, read_corpus
from tools.perception_eval.metrics import Box, CaseObservation, EvaluationRun, ObservedElement


def capture(
    output: Path,
    manifest: Path = MANIFEST_PATH,
    model_reference: str | None = None,
    image_size: int | None = None,
) -> EvaluationRun:
    """Run the current production perception stack over every fixture once."""
    corpus = read_corpus(manifest)
    descriptor = ImageDescriptor(model_reference)
    if image_size is not None:
        descriptor.detection_image_size = image_size
    observations: list[CaseObservation] = []
    for case in corpus.cases:
        started = time.perf_counter()
        elements, _ = descriptor.describe(case.screenshot.read_bytes())
        latency_ms = (time.perf_counter() - started) * 1000
        width, height = _image_size(case.screenshot)
        keyboard_observation = keyboard.observe(elements, case.keyboard_visible)
        normal_projection = marks.project_targets(
            elements,
            width,
            height,
            case.keyboard_visible,
        )
        expanded_projection = marks.project_targets(
            elements,
            width,
            height,
            case.keyboard_visible,
            include_keyboard_targets=True,
        )
        frame = descriptor.last_frame
        observations.append(
            CaseObservation(
                case_id=case.id,
                fixture_digest=case.digest,
                visual_keyboard_visible=keyboard_observation.visual_visible,
                mark_count=len(normal_projection.marks),
                latency_ms=round(latency_ms, 3),
                peak_rss_mib=round(_peak_rss_mib(), 3),
                elements=[
                    ObservedElement(
                        element=element.element,
                        text=element.text,
                        confidence=element.confidence,
                        bounding_box=Box(**element.bounding_box.model_dump()),
                        semantic_roles=set(element.semantic_roles),
                    )
                    for element in elements
                ],
                association_count=len(frame.associations) if frame is not None else 0,
                out_of_bounds_count=sum(
                    not _is_inside_screen(element.bounding_box, width, height)
                    for element in elements
                ),
                normal_keyboard_target_count=sum(
                    bool(mark.element.semantic_roles & {"keyboard_region", "keyboard_key"})
                    for mark in normal_projection.marks
                ),
                expanded_keyboard_target_count=sum(
                    "keyboard_key" in mark.element.semantic_roles
                    for mark in expanded_projection.marks
                ),
            )
        )
    run = EvaluationRun(
        detector=descriptor.model_identity,
        inference_image_size=descriptor.detection_image_size,
        ocr=descriptor.ocr_reader.identity,
        observations=observations,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(run.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return run


def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        return image.size


def _peak_rss_mib() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return peak / divisor


def _is_inside_screen(box, width: int, height: int) -> bool:
    return (
        0 <= box.x_min <= box.x_max <= width
        and 0 <= box.y_min <= box.y_max <= height
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture the current perception baseline.")
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument(
        "--model",
        help="Pinned detector package: hf://owner/repository@commit",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        help="Benchmark override for the package's inference image size",
    )
    arguments = parser.parse_args()
    run = capture(
        arguments.output,
        arguments.manifest,
        arguments.model,
        arguments.image_size,
    )
    print(f"Captured {len(run.observations)} cases in {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
