"""Capture an independent whole-screen EasyOCR comparison baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from tools.perception_eval.capture_ocr import capture_reader
from tools.perception_eval.corpus import MANIFEST_PATH
from yewo.agent.perception.easyocr_adapter import EasyOCRReader


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    arguments = parser.parse_args()
    run = capture_reader(EasyOCRReader(), arguments.output, arguments.manifest)
    print(f"Captured {len(run.observations)} OCR cases in {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
