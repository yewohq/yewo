"""Benchmark current perception and independent YOLO/RapidOCR fusion variants."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Callable

from PIL import Image

from tools.perception_eval.corpus import MANIFEST_PATH, PerceptionCase, read_corpus
from tools.perception_eval.ocr_evaluation import (
    OCRCaseObservation,
    OCREvaluationRun,
    score_ocr,
)
from yewo.agent.perception.detector import ImageDescriptor
from yewo.agent.perception.ocr import OCRResult
from yewo.agent.perception.rapidocr_adapter import RapidOCRReader
from yewo.defaults import DEFAULT_PERCEPTION_MODEL

DEFAULT_SIZES = (2000, 1600, 1280, 960)


def benchmark(
    output: Path,
    manifest_path: Path = MANIFEST_PATH,
    model_reference: str = DEFAULT_PERCEPTION_MODEL,
    image_size: int = 896,
    rapid_sizes: tuple[int, ...] = DEFAULT_SIZES,
    fusion_size: int = 1280,
) -> dict[str, object]:
    """Measure warmed corpus passes and write one machine-readable result."""
    corpus = read_corpus(manifest_path)
    images = [_load_image(case) for case in corpus.cases]
    png_bytes = [case.screenshot.read_bytes() for case in corpus.cases]
    descriptor = ImageDescriptor(model_reference)
    descriptor.detection_image_size = image_size

    descriptor.describe(png_bytes[0])
    descriptor.clear_cache()
    production_latencies = _measure_calls(
        [lambda data=data: descriptor.describe(data) for data in png_bytes]
    )
    cached_frame_latencies = _measure_calls(
        [lambda: descriptor.describe(png_bytes[-1]) for _ in range(5)]
    )
    _run_yolo(descriptor, images[0])
    yolo_latencies = _measure_calls(
        [lambda image=image: _run_yolo(descriptor, image) for image in images]
    )

    rapid_results: dict[str, object] = {}
    for size in rapid_sizes:
        reader = RapidOCRReader(max_side_len=size)
        reader.recognize(images[0])
        observations, latencies = _measure_ocr(reader, corpus.cases, images)
        run = OCREvaluationRun(ocr=reader.identity, observations=observations)
        quality = score_ocr(corpus, run).model_dump(exclude={"peak_rss_mib"})
        rapid_results[str(size)] = {
            "latency": _latency_summary(latencies),
            "quality": quality,
        }

    fusion_reader = RapidOCRReader(max_side_len=fusion_size)
    fusion_reader.recognize(images[0])
    _run_yolo(descriptor, images[0])
    sequential_latencies = _measure_calls(
        [lambda image=image: _run_sequential(descriptor, fusion_reader, image) for image in images]
    )
    concurrent_latencies = _measure_calls(
        [lambda image=image: _run_concurrent(descriptor, fusion_reader, image) for image in images]
    )

    result: dict[str, object] = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "manifest": str(manifest_path),
        "cases": len(corpus.cases),
        "detector": descriptor.model_identity,
        "detector_image_size": image_size,
        "production_fused": _latency_summary(production_latencies),
        "production_exact_frame_cache_hit": _latency_summary(cached_frame_latencies),
        "yolo_only": _latency_summary(yolo_latencies),
        "rapidocr_by_max_side": rapid_results,
        "fusion_max_side": fusion_size,
        "sequential_yolo_rapidocr": _latency_summary(sequential_latencies),
        "concurrent_yolo_rapidocr": _latency_summary(concurrent_latencies),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _load_image(case: PerceptionCase) -> Image.Image:
    with Image.open(case.screenshot) as image:
        return image.convert("RGB")


def _run_yolo(descriptor: ImageDescriptor, image: Image.Image) -> object:
    return descriptor.yolo(image, imgsz=descriptor.detection_image_size, verbose=False)[0]


def _run_sequential(
    descriptor: ImageDescriptor,
    reader: RapidOCRReader,
    image: Image.Image,
) -> None:
    _run_yolo(descriptor, image)
    reader.recognize(image)


def _run_concurrent(
    descriptor: ImageDescriptor,
    reader: RapidOCRReader,
    image: Image.Image,
) -> None:
    with ThreadPoolExecutor(max_workers=2) as executor:
        yolo_future = executor.submit(_run_yolo, descriptor, image)
        ocr_future = executor.submit(reader.recognize, image)
        yolo_future.result()
        ocr_future.result()


def _measure_ocr(
    reader: RapidOCRReader,
    cases: list[PerceptionCase],
    images: list[Image.Image],
) -> tuple[list[OCRCaseObservation], list[float]]:
    observations: list[OCRCaseObservation] = []
    latencies: list[float] = []
    for case, image in zip(cases, images, strict=True):
        started = time.perf_counter()
        result = reader.recognize(image)
        latencies.append((time.perf_counter() - started) * 1000)
        observations.append(_observation(case, result))
    return observations, latencies


def _observation(case: PerceptionCase, result: OCRResult) -> OCRCaseObservation:
    return OCRCaseObservation(
        case_id=case.id,
        fixture_digest=case.digest,
        result=result,
        peak_rss_mib=0,
    )


def _measure_calls(calls: list[Callable[[], object]]) -> list[float]:
    latencies: list[float] = []
    for call in calls:
        started = time.perf_counter()
        call()
        latencies.append((time.perf_counter() - started) * 1000)
    return latencies


def _latency_summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
    return {
        "mean_ms": round(statistics.fmean(values), 3),
        "median_ms": round(statistics.median(values), 3),
        "p95_ms": round(ordered[p95_index], 3),
        "min_ms": round(ordered[0], 3),
        "max_ms": round(ordered[-1], 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--model", default=DEFAULT_PERCEPTION_MODEL)
    parser.add_argument("--image-size", type=int, default=896)
    parser.add_argument("--rapid-sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--fusion-size", type=int, default=1280)
    arguments = parser.parse_args()
    result = benchmark(
        arguments.output,
        arguments.manifest,
        arguments.model,
        arguments.image_size,
        tuple(arguments.rapid_sizes),
        arguments.fusion_size,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
