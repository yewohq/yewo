"""Measured screen-family classification and its operational gate."""

from __future__ import annotations

import hashlib
import hmac
import json
from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from yewo.agent.memory.screen_identity import phash_distance
from yewo.discovery.artifacts import DiscoveryContract, SHA256_PATTERN
from yewo.identity import derived_uuid, new_uuid

SCREEN_FAMILY_CLASSIFIER_VERSION = "screen-family-structural-phash-v1"
SAME_FAMILY_PHASH_DISTANCE = 112
DIFFERENT_FAMILY_PHASH_DISTANCE = 128


class ScreenFamilyDecision(str, Enum):
    SAME = "same_family"
    DIFFERENT = "different_family"
    UNCERTAIN = "uncertain"


class ScreenFamilyLabel(str, Enum):
    SAME = "same_family"
    DIFFERENT = "different_family"


class ScreenFamilyFingerprint(DiscoveryContract):
    """Language-independent visual and structural evidence for one state."""

    sample_id: str
    app_scope: str
    platform: str
    viewport_class: str
    phash: str
    structure_sha256: str = Field(pattern=SHA256_PATTERN)
    traits: frozenset[str] = Field(min_length=1)


class ScreenFamilyPair(DiscoveryContract):
    """One independently labelled classifier comparison."""

    pair_id: str
    left: ScreenFamilyFingerprint
    right: ScreenFamilyFingerprint
    label: ScreenFamilyLabel


class IdentityCalibrationThresholds(DiscoveryContract):
    max_false_merge_rate: float = Field(ge=0, le=1)
    max_false_split_rate: float = Field(ge=0, le=1)
    max_uncertain_rate: float = Field(ge=0, le=1)
    min_pairs: int = Field(gt=0)
    min_same_family_pairs: int = Field(gt=0)
    min_different_family_pairs: int = Field(gt=0)
    required_platforms: frozenset[str] = Field(min_length=1)
    required_traits: frozenset[str] = Field(min_length=1)


class IdentityCalibrationMetrics(DiscoveryContract):
    pair_count: int = Field(ge=0)
    same_family_pairs: int = Field(ge=0)
    different_family_pairs: int = Field(ge=0)
    false_merges: int = Field(ge=0)
    false_splits: int = Field(ge=0)
    uncertain: int = Field(ge=0)
    false_merge_rate: float = Field(ge=0, le=1)
    false_split_rate: float = Field(ge=0, le=1)
    uncertain_rate: float = Field(ge=0, le=1)
    platforms: frozenset[str]
    traits: frozenset[str]


class IdentityCalibrationResult(DiscoveryContract):
    """Integrity-protected evidence required before family grouping is enabled."""

    schema_version: Literal[1] = 1
    artifact_id: str = Field(default_factory=new_uuid)
    classifier_version: Literal["screen-family-structural-phash-v1"] = (
        SCREEN_FAMILY_CLASSIFIER_VERSION
    )
    corpus_sha256: str = Field(pattern=SHA256_PATTERN)
    thresholds: IdentityCalibrationThresholds
    metrics: IdentityCalibrationMetrics
    passed: bool
    integrity_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_gate_decision(self) -> "IdentityCalibrationResult":
        expected = calibration_passes(self.metrics, self.thresholds)
        if self.passed != expected:
            raise ValueError("The calibration decision must follow the declared thresholds.")
        return self


class ScreenFamilyClassificationReceipt(DiscoveryContract):
    """Local classifier evidence that may group one accepted observation."""

    receipt_id: str = Field(default_factory=new_uuid)
    evidence_owner: Literal["screen_family_classifier"] = "screen_family_classifier"
    observation_id: str
    screen_family_id: str
    classifier_version: Literal["screen-family-structural-phash-v1"] = (
        SCREEN_FAMILY_CLASSIFIER_VERSION
    )
    calibration_artifact_id: str
    fingerprint_sha256: str = Field(pattern=SHA256_PATTERN)


def classify_screen_family(
    left: ScreenFamilyFingerprint,
    right: ScreenFamilyFingerprint,
) -> ScreenFamilyDecision:
    """Classify conservatively; ambiguous visual distance remains uncertain."""
    if left.app_scope != right.app_scope:
        return ScreenFamilyDecision.DIFFERENT
    if left.structure_sha256 != right.structure_sha256:
        return ScreenFamilyDecision.DIFFERENT
    distance = phash_distance(left.phash, right.phash)
    if distance is None:
        return ScreenFamilyDecision.UNCERTAIN
    if distance <= SAME_FAMILY_PHASH_DISTANCE:
        return ScreenFamilyDecision.SAME
    if distance >= DIFFERENT_FAMILY_PHASH_DISTANCE:
        return ScreenFamilyDecision.DIFFERENT
    return ScreenFamilyDecision.UNCERTAIN


def evaluate_identity_pairs(
    pairs: tuple[ScreenFamilyPair, ...],
) -> IdentityCalibrationMetrics:
    """Measure merge, split, and abstention errors independently."""
    pair_ids = [pair.pair_id for pair in pairs]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("Identity calibration pair identifiers must be unique.")
    same_pairs = sum(pair.label is ScreenFamilyLabel.SAME for pair in pairs)
    different_pairs = len(pairs) - same_pairs
    false_merges = false_splits = uncertain = 0
    platforms: set[str] = set()
    traits: set[str] = set()
    for pair in pairs:
        decision = classify_screen_family(pair.left, pair.right)
        platforms.update((pair.left.platform, pair.right.platform))
        traits.update(pair.left.traits)
        traits.update(pair.right.traits)
        uncertain += decision is ScreenFamilyDecision.UNCERTAIN
        false_merges += (
            pair.label is ScreenFamilyLabel.DIFFERENT
            and decision is ScreenFamilyDecision.SAME
        )
        false_splits += (
            pair.label is ScreenFamilyLabel.SAME
            and decision is ScreenFamilyDecision.DIFFERENT
        )
    return IdentityCalibrationMetrics(
        pair_count=len(pairs),
        same_family_pairs=same_pairs,
        different_family_pairs=different_pairs,
        false_merges=false_merges,
        false_splits=false_splits,
        uncertain=uncertain,
        false_merge_rate=_ratio(false_merges, different_pairs),
        false_split_rate=_ratio(false_splits, same_pairs),
        uncertain_rate=_ratio(uncertain, len(pairs)),
        platforms=frozenset(platforms),
        traits=frozenset(traits),
    )


def calibration_passes(
    metrics: IdentityCalibrationMetrics,
    thresholds: IdentityCalibrationThresholds,
) -> bool:
    return (
        metrics.pair_count >= thresholds.min_pairs
        and metrics.same_family_pairs >= thresholds.min_same_family_pairs
        and metrics.different_family_pairs >= thresholds.min_different_family_pairs
        and metrics.false_merge_rate <= thresholds.max_false_merge_rate
        and metrics.false_split_rate <= thresholds.max_false_split_rate
        and metrics.uncertain_rate <= thresholds.max_uncertain_rate
        and thresholds.required_platforms.issubset(metrics.platforms)
        and thresholds.required_traits.issubset(metrics.traits)
    )


def build_calibration_result(
    pairs: tuple[ScreenFamilyPair, ...],
    corpus_sha256: str,
    thresholds: IdentityCalibrationThresholds,
) -> IdentityCalibrationResult:
    metrics = evaluate_identity_pairs(pairs)
    artifact_id = derived_uuid(
        "identity-calibration-v1",
        SCREEN_FAMILY_CLASSIFIER_VERSION,
        corpus_sha256,
    )
    return IdentityCalibrationResult(
        artifact_id=artifact_id,
        corpus_sha256=corpus_sha256,
        thresholds=thresholds,
        metrics=metrics,
        passed=calibration_passes(metrics, thresholds),
    )


def _normalized_calibration_dict(
    result: IdentityCalibrationResult,
    *,
    clear_integrity: bool,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": result.schema_version,
        "artifact_id": result.artifact_id,
        "classifier_version": result.classifier_version,
        "corpus_sha256": result.corpus_sha256,
        "thresholds": {
            **result.thresholds.model_dump(
                mode="json",
                exclude={"required_platforms", "required_traits"},
            ),
            "required_platforms": sorted(result.thresholds.required_platforms),
            "required_traits": sorted(result.thresholds.required_traits),
        },
        "metrics": {
            **result.metrics.model_dump(
                mode="json",
                exclude={"platforms", "traits"},
            ),
            "platforms": sorted(result.metrics.platforms),
            "traits": sorted(result.metrics.traits),
        },
        "passed": result.passed,
        "integrity_sha256": None if clear_integrity else result.integrity_sha256,
    }
    return payload


def _calibration_payload(result: IdentityCalibrationResult) -> bytes:
    payload = _normalized_calibration_dict(result, clear_integrity=True)
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def seal_identity_calibration(
    result: IdentityCalibrationResult,
) -> IdentityCalibrationResult:
    digest = hashlib.sha256(_calibration_payload(result)).hexdigest()
    return result.model_copy(update={"integrity_sha256": digest})


def verify_identity_calibration(result: IdentityCalibrationResult) -> bool:
    stored = result.integrity_sha256
    expected = hashlib.sha256(_calibration_payload(result)).hexdigest()
    return stored is not None and hmac.compare_digest(stored, expected)


def serialize_identity_calibration(result: IdentityCalibrationResult) -> bytes:
    if not verify_identity_calibration(result):
        raise ValueError("Identity calibration requires valid integrity evidence.")
    payload = _normalized_calibration_dict(result, clear_integrity=False)
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def require_operational_calibration(result: IdentityCalibrationResult) -> None:
    if not verify_identity_calibration(result):
        raise ValueError("Screen-family calibration requires valid integrity evidence.")
    if not result.passed:
        raise ValueError("Screen-family calibration has not passed its declared gate.")


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0
