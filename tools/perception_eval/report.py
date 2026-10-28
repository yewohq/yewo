"""Report comparable metrics from a saved perception observation."""

from __future__ import annotations

import argparse
from pathlib import Path

from tools.perception_eval.corpus import MANIFEST_PATH, read_corpus
from tools.perception_eval.metrics import EvaluationRun, score


def main() -> int:
    parser = argparse.ArgumentParser(description="Score a saved perception baseline.")
    parser.add_argument("observation", type=Path)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    arguments = parser.parse_args()
    corpus = read_corpus(arguments.manifest)
    run = EvaluationRun.model_validate_json(arguments.observation.read_text(encoding="utf-8"))
    print(score(corpus, run).model_dump_json(indent=2))
    if corpus.missing_platforms or corpus.missing_traits:
        print("Corpus coverage is incomplete.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
