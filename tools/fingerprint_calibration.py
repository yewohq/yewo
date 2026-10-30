"""Measure screen-fingerprint distances on an explicitly labelled corpus."""

from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
from typing import Any

from yewo.observations import build_screen_fingerprint, fingerprint_distance


def calibrate(manifest_path: Path) -> dict[str, Any]:
    """Return pairwise measurements without selecting a production threshold."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = payload.get("samples")
    if not isinstance(samples, list) or len(samples) < 2:
        raise ValueError("The calibration manifest must contain at least two samples.")
    fingerprints = []
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("Each calibration sample must be an object.")
        relative_path = sample.get("path")
        logical_screen = sample.get("logical_screen")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError("Each calibration sample must have a path.")
        if not isinstance(logical_screen, str) or not logical_screen:
            raise ValueError("Each calibration sample must have a logical_screen label.")
        image_path = manifest_path.parent / relative_path
        content = image_path.read_bytes()
        artifact_id = f"cal_{sha256(content).hexdigest()[:32]}"
        fingerprint = build_screen_fingerprint(
            content,
            None,
            source_artifact_id=artifact_id,
        )
        fingerprints.append((relative_path, logical_screen, fingerprint))

    pairs: list[dict[str, Any]] = []
    distances: dict[str, list[int]] = defaultdict(list)
    for index, left in enumerate(fingerprints):
        for right in fingerprints[index + 1 :]:
            relationship = "same" if left[1] == right[1] else "different"
            distance = fingerprint_distance(left[2], right[2])
            distances[relationship].append(distance)
            pairs.append(
                {
                    "left": left[0],
                    "right": right[0],
                    "relationship": relationship,
                    "distance": distance,
                }
            )

    return {
        "algorithm": "application-phash-v1",
        "threshold_selected": False,
        "corpus_scope": payload.get("corpus_scope"),
        "sample_count": len(fingerprints),
        "logical_screen_count": len({item[1] for item in fingerprints}),
        "distributions": {
            relationship: _distribution(values)
            for relationship, values in sorted(distances.items())
        },
        "pairs": pairs,
        "limitations": payload.get("limitations", []),
    }


def _distribution(values: list[int]) -> dict[str, int | float]:
    return {
        "count": len(values),
        "minimum": min(values),
        "median": median(values),
        "maximum": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = calibrate(arguments.manifest)
    serialized = f"{json.dumps(result, indent=2, sort_keys=True)}\n"
    if arguments.output is None:
        print(serialized, end="")
        return
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(serialized, encoding="utf-8")
    print(
        f"Measured {result['sample_count']} samples and "
        f"{len(result['pairs'])} pairs. No threshold was selected."
    )


if __name__ == "__main__":
    main()
