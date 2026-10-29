import pytest
from pydantic import ValidationError

from yewo.discovery.identity_calibration import (
    IdentityCalibrationThresholds,
    ScreenFamilyDecision,
    ScreenFamilyFingerprint,
    ScreenFamilyLabel,
    ScreenFamilyPair,
    build_calibration_result,
    classify_screen_family,
    require_operational_calibration,
    seal_identity_calibration,
    serialize_identity_calibration,
    verify_identity_calibration,
)


def _fingerprint(sample_id, phash, structure="a", traits=frozenset({"form"})):
    return ScreenFamilyFingerprint(
        sample_id=sample_id,
        app_scope="held-out-app",
        platform="android",
        viewport_class="phone-portrait",
        phash=phash,
        structure_sha256=structure * 64,
        traits=traits,
    )


def _thresholds(**updates):
    values = {
        "max_false_merge_rate": 0.0,
        "max_false_split_rate": 0.0,
        "max_uncertain_rate": 0.0,
        "min_pairs": 2,
        "min_same_family_pairs": 1,
        "min_different_family_pairs": 1,
        "required_platforms": frozenset({"android"}),
        "required_traits": frozenset({"form"}),
    }
    values.update(updates)
    return IdentityCalibrationThresholds(**values)


def _pairs():
    return (
        ScreenFamilyPair(
            pair_id="same-form",
            left=_fingerprint("form-keyboard", "0" * 64),
            right=_fingerprint("form-no-keyboard", "0" * 63 + "f"),
            label=ScreenFamilyLabel.SAME,
        ),
        ScreenFamilyPair(
            pair_id="different-screen",
            left=_fingerprint("form", "0" * 64),
            right=_fingerprint("drawer", "0" * 64, structure="b"),
            label=ScreenFamilyLabel.DIFFERENT,
        ),
    )


def test_classifier_requires_both_structure_and_visual_compatibility():
    same = classify_screen_family(_pairs()[0].left, _pairs()[0].right)
    different = classify_screen_family(_pairs()[1].left, _pairs()[1].right)
    uncertain = classify_screen_family(
        _fingerprint("left", "0" * 64),
        _fingerprint("right", "f" * 30 + "0" * 34),
    )

    assert same is ScreenFamilyDecision.SAME
    assert different is ScreenFamilyDecision.DIFFERENT
    assert uncertain is ScreenFamilyDecision.UNCERTAIN

    malformed = classify_screen_family(
        _fingerprint("left-invalid", "not-a-hash"),
        _fingerprint("right-invalid", "also-invalid"),
    )
    assert malformed is ScreenFamilyDecision.UNCERTAIN


def test_calibration_gate_measures_errors_abstention_and_coverage():
    result = build_calibration_result(_pairs(), "c" * 64, _thresholds())

    assert result.passed
    assert result.metrics.false_merge_rate == 0
    assert result.metrics.false_split_rate == 0
    assert result.metrics.uncertain_rate == 0

    insufficient = build_calibration_result(
        _pairs(),
        "c" * 64,
        _thresholds(required_traits=frozenset({"form", "overlay"})),
    )
    assert not insufficient.passed


def test_operational_calibration_requires_gate_and_integrity():
    draft = build_calibration_result(_pairs(), "c" * 64, _thresholds())

    with pytest.raises(ValueError, match="integrity"):
        require_operational_calibration(draft)
    with pytest.raises(ValueError, match="integrity"):
        serialize_identity_calibration(draft)

    sealed = seal_identity_calibration(draft)
    require_operational_calibration(sealed)
    assert verify_identity_calibration(sealed)
    assert b'"integrity_sha256":' in serialize_identity_calibration(sealed)

    tampered = sealed.model_copy(
        update={"corpus_sha256": "d" * 64},
    )
    assert not verify_identity_calibration(tampered)

    failed = seal_identity_calibration(
        build_calibration_result(
            _pairs(),
            "c" * 64,
            _thresholds(required_traits=frozenset({"missing"})),
        )
    )
    with pytest.raises(ValueError, match="has not passed"):
        require_operational_calibration(failed)


def test_calibration_rejects_duplicate_pair_identity():
    duplicate = _pairs()[0].model_copy()

    with pytest.raises(ValueError, match="identifiers must be unique"):
        build_calibration_result(
            (_pairs()[0], duplicate),
            "c" * 64,
            _thresholds(),
        )


def test_calibration_decision_cannot_disagree_with_metrics():
    result = build_calibration_result(_pairs(), "c" * 64, _thresholds())
    payload = result.model_dump(mode="json")
    payload["passed"] = False

    with pytest.raises(ValidationError, match="declared thresholds"):
        type(result).model_validate(payload)
