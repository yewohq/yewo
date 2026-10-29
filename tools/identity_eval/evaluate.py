"""Evaluate and seal the held-out screen-family calibration gate."""

from __future__ import annotations

import argparse
from pathlib import Path

from yewo.discovery.identity_calibration import (
    build_calibration_result,
    seal_identity_calibration,
    serialize_identity_calibration,
)
from tools.identity_eval.corpus import MANIFEST_PATH, read_identity_corpus


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/identity/calibration_baseline.json"),
    )
    args = parser.parse_args()
    pairs, corpus_sha256, thresholds = read_identity_corpus(args.manifest)
    result = seal_identity_calibration(
        build_calibration_result(pairs, corpus_sha256, thresholds)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(serialize_identity_calibration(result))
    print(args.output)
    print(
        f"pairs={result.metrics.pair_count} "
        f"false_merges={result.metrics.false_merges} "
        f"false_splits={result.metrics.false_splits} "
        f"uncertain={result.metrics.uncertain} passed={result.passed}"
    )
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
