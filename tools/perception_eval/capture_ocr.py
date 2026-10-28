"""Capture an independent whole-screen RapidOCR shadow baseline."""

from __future__ import annotations

import argparse
import resource
import sys
from pathlib import Path

from PIL import Image

from tools.perception_eval.corpus import MANIFEST_PATH, read_corpus
from tools.perception_eval.ocr_evaluation import OCRCaseObservation, OCREvaluationRun
from yewo.agent.perception.ocr import OCRReader
from yewo.agent.perception.rapidocr_adapter import RapidOCRReader


def capture_ocr(
    output: Path,
    manifest: Path = MANIFEST_PATH,
) -> OCREvaluationRun:
    """Run RapidOCR once over each complete fixture without invoking YOLO."""
    return capture_reader(RapidOCRReader(), output, manifest)


def capture_reader(
    reader: OCRReader,
    output: Path,
    manifest: Path = MANIFEST_PATH,
) -> OCREvaluationRun:
    """Run one OCR reader over the corpus and persist normalized observations."""
    corpus = read_corpus(manifest)
    observations: list[OCRCaseObservation] = []
    for case in corpus.cases:
        with Image.open(case.screenshot) as screenshot:
            result = reader.recognize(screenshot.convert("RGB"))
        observations.append(
            OCRCaseObservation(
                case_id=case.id,
                fixture_digest=case.digest,
                result=result,
                peak_rss_mib=round(_peak_rss_mib(), 3),
            )
        )
    run = OCREvaluationRun(
        ocr=reader.identity,
        observations=observations,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(run.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return run


def _peak_rss_mib() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return peak / divisor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    arguments = parser.parse_args()
    run = capture_ocr(arguments.output, arguments.manifest)
    print(f"Captured {len(run.observations)} OCR cases in {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
