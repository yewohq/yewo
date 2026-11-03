"""Tests for immutable observation evidence records."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from yewo.artifacts import ArtifactSensitivity
from yewo.devices import SystemRegionKind
from yewo.observations import (
    Observation,
    RedactionProvenance,
    ScreenFingerprint,
    ScreenRegion,
    ScreenshotRedactionState,
    ScreenshotReference,
)
from observations.factories import build_observation, build_screenshot


def test_observation_serialization_round_trip_preserves_evidence() -> None:
    observation = build_observation(predecessor_observation_id="obs_0")

    restored = Observation.model_validate_json(observation.model_dump_json())

    assert restored == observation


def test_screenshot_reference_rejects_unsafe_paths() -> None:
    with pytest.raises(ValidationError, match="safe relative path"):
        ScreenshotReference.model_validate(
            {
                **build_screenshot().model_dump(),
                "path": "../secret.png",
            }
        )

    with pytest.raises(ValidationError, match="safe relative path"):
        ScreenshotReference.model_validate(
            {
                **build_screenshot().model_dump(),
                "path": "/tmp/screenshot.png",
            }
        )


def test_observation_requires_truthful_redaction_labels() -> None:
    with pytest.raises(ValidationError, match="raw screenshot reference"):
        build_observation(
            raw_screenshot=build_screenshot(redaction_state=ScreenshotRedactionState.REDACTED)
        )


def test_redacted_reference_retains_its_input_sensitivity() -> None:
    payload = build_screenshot().model_dump()
    payload["redaction"] = RedactionProvenance(
        policy_id="region-mask",
        region_count=1,
        region_set_sha256="d" * 64,
        input_sensitivity=ArtifactSensitivity.DECLARED_SENSITIVE,
    )

    with pytest.raises(ValidationError, match="input sensitivity"):
        ScreenshotReference.model_validate(payload)


def test_observation_distinguishes_unknown_and_empty_system_regions() -> None:
    unknown = build_observation(system_regions=None)
    observed_empty = build_observation(system_regions=())

    assert unknown.system_regions is None
    assert observed_empty.system_regions == ()


def test_observation_rejects_a_fingerprint_with_different_mask_evidence() -> None:
    fingerprint = ScreenFingerprint(
        value="0123456789abcdef",
        masked_pixels_sha256="b" * 64,
        system_region_mask_sha256="d" * 64,
        source_artifact_id="shot_1",
    )

    with pytest.raises(ValidationError, match="system-region mask"):
        build_observation(screen_fingerprint=fingerprint, system_regions=None)


def test_observation_rejects_region_outside_screenshot() -> None:
    region = ScreenRegion(
        kind=SystemRegionKind.STATUS_BAR,
        left=0,
        top=0,
        right=201,
        bottom=40,
    )

    with pytest.raises(ValidationError, match="outside the screenshot"):
        build_observation(system_regions=(region,))


def test_observation_rejects_naive_time_and_self_predecessor() -> None:
    with pytest.raises(ValidationError, match="include a timezone"):
        build_observation(captured_at=datetime(2026, 10, 30))
    with pytest.raises(ValidationError, match="cannot name itself"):
        build_observation(predecessor_observation_id="obs_1")


def test_observation_rejects_an_incompatible_schema_version() -> None:
    payload = build_observation().model_dump()
    payload["schema_version"] = "unsupported"

    with pytest.raises(ValidationError, match="schema_version"):
        Observation.model_validate(payload)
