"""Tests for grounding request and result invariants."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from yewo.artifacts import ArtifactRetention, ArtifactSensitivity
from yewo.grounding import (
    GroundingDisposition,
    GroundingOutputShape,
    GroundingTargetOperation,
)
from yewo.models import ModelArtifactKind, ModelArtifactReference, ModelCallDisposition
from yewo.observations import ScreenshotRedactionState
from grounding.support import build_grounding_request
from observations.factories import build_screenshot


def test_grounding_request_rejects_raw_pixels_and_incompatible_region_output() -> None:
    with pytest.raises(ValidationError, match="redacted screenshot"):
        build_grounding_request(
            screenshot=build_screenshot(redaction_state=ScreenshotRedactionState.RAW)
        )

    with pytest.raises(ValidationError, match="scroll-region"):
        build_grounding_request(output_shape=GroundingOutputShape.REGION)


def test_rejected_result_cannot_contain_a_location() -> None:
    from yewo.grounding import GroundedPoint, GroundingResult, NormalizedPoint, PixelPoint
    from yewo.models import ModelCallUsage

    with pytest.raises(ValidationError, match="reason and no location"):
        GroundingResult(
            grounding_id="grounding-1",
            profile_id="visual-grounder-primary",
            model_profile_id="grounder-primary",
            source_screenshot_sha256="a" * 64,
            output_shape=GroundingOutputShape.POINT,
            disposition=GroundingDisposition.MALFORMED,
            location=GroundedPoint(
                normalized=NormalizedPoint(x=0.5, y=0.5),
                pixel=PixelPoint(x=100, y=200),
            ),
            reason="Invalid response.",
            requested_provider="openrouter",
            requested_model="openrouter/example/grounder",
            transport_disposition=ModelCallDisposition.MALFORMED,
            raw_response_artifact=ModelArtifactReference(
                artifact_id="call-1",
                kind=ModelArtifactKind.CALL,
                sha256="b" * 64,
                body_sha256="c" * 64,
                content_sha256="d" * 64,
                path=Path("assets/models/calls.jsonl"),
                body_retained=True,
                sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
                retention=ArtifactRetention.RUN,
            ),
            usage=ModelCallUsage(),
        )


def test_region_output_is_available_only_for_scroll_region_requests() -> None:
    request = build_grounding_request(
        operation=GroundingTargetOperation.SCROLL_REGION,
        output_shape=GroundingOutputShape.REGION,
    )

    assert request.output_shape is GroundingOutputShape.REGION
