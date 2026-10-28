"""Report OCR-only metrics from a saved whole-screen shadow run."""

from __future__ import annotations

import argparse
from pathlib import Path

from tools.perception_eval.corpus import MANIFEST_PATH, read_corpus
from tools.perception_eval.ocr_evaluation import OCREvaluationRun, score_ocr


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observation", type=Path)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    arguments = parser.parse_args()
    corpus = read_corpus(arguments.manifest)
    run = OCREvaluationRun.model_validate_json(arguments.observation.read_text(encoding="utf-8"))
    print(score_ocr(corpus, run).model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
