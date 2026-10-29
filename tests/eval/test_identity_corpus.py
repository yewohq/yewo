import json

import pytest

from tools.identity_eval.corpus import (
    MANIFEST_PATH,
    IdentityCorpusError,
    read_identity_corpus,
)
from yewo.discovery.identity_calibration import (
    IdentityCalibrationResult,
    build_calibration_result,
    seal_identity_calibration,
    verify_identity_calibration,
)


BASELINE_PATH = MANIFEST_PATH.with_name("calibration_baseline.json")


def test_held_out_identity_baseline_matches_frozen_corpus():
    pairs, corpus_sha256, thresholds = read_identity_corpus()
    measured = seal_identity_calibration(
        build_calibration_result(pairs, corpus_sha256, thresholds)
    )
    baseline = IdentityCalibrationResult.model_validate_json(
        BASELINE_PATH.read_bytes()
    )

    assert measured == baseline
    assert verify_identity_calibration(baseline)
    assert baseline.passed
    assert baseline.metrics.pair_count == 12
    assert baseline.metrics.false_merges == 0
    assert baseline.metrics.false_splits == 0
    assert baseline.metrics.uncertain == 1


def test_identity_corpus_refuses_changed_screenshot_digest(tmp_path):
    payload = json.loads(MANIFEST_PATH.read_text())
    payload["samples"][0]["screenshot_sha256"] = "0" * 64
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload))

    with pytest.raises(IdentityCorpusError, match="digest mismatch"):
        read_identity_corpus(manifest)


def test_identity_corpus_refuses_missing_declared_trait(tmp_path):
    payload = json.loads(MANIFEST_PATH.read_text())
    payload["required_traits"].append("undeclared-trait")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload))

    with pytest.raises(IdentityCorpusError, match="missing a required trait"):
        read_identity_corpus(manifest)
