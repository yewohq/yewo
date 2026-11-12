"""Tests for grounding request and result invariants."""

import pytest
from pydantic import ValidationError

from yewo.grounding import (
    GroundingDisposition,
    GroundingCorrection,
    GroundingEvidence,
    GroundingOutputShape,
    GroundingTargetOperation,
    ModelGroundingResolution,
    PixelPoint,
)
from grounding.support import DeterministicGrounder, build_grounding_request


def test_grounding_request_rejects_incompatible_region_output() -> None:
    with pytest.raises(ValidationError, match="scroll-region"):
        build_grounding_request(output_shape=GroundingOutputShape.REGION)

    with pytest.raises(ValidationError, match="point output"):
        build_grounding_request(
            operation=GroundingTargetOperation.SCROLL_REGION,
            output_shape=GroundingOutputShape.REGION,
            correction=GroundingCorrection(
                previous_point=PixelPoint(x=100, y=200),
                minimum_distance_px=8,
            ),
        )


def test_rejected_result_cannot_contain_a_location() -> None:
    from yewo.grounding import GroundedPoint, GroundingResult, NormalizedPoint, PixelPoint

    with pytest.raises(ValidationError, match="reason and no location"):
        GroundingResult(
            grounding_id="grounding-1",
            source_screenshot_sha256="a" * 64,
            output_shape=GroundingOutputShape.POINT,
            disposition=GroundingDisposition.MALFORMED,
            resolution=ModelGroundingResolution(
                grounding_profile_id="visual-grounder-primary",
                model_request_ids=("request-1",),
            ),
            location=GroundedPoint(
                normalized=NormalizedPoint(x=0.5, y=0.5),
                pixel=PixelPoint(x=100, y=200),
            ),
            reason="Invalid response.",
        )


def test_region_output_is_available_only_for_scroll_region_requests() -> None:
    request = build_grounding_request(
        operation=GroundingTargetOperation.SCROLL_REGION,
        output_shape=GroundingOutputShape.REGION,
    )

    assert request.output_shape is GroundingOutputShape.REGION


@pytest.mark.parametrize(
    "updates",
    [
        {"grounding_id": "grounding-other"},
        {"source_screenshot_sha256": "f" * 64},
        {"output_shape": GroundingOutputShape.REGION},
    ],
)
def test_grounding_evidence_rejects_a_mismatched_result(updates: dict[str, object]) -> None:
    request = build_grounding_request()
    result = DeterministicGrounder(100, 200).ground(request).model_copy(update=updates)

    with pytest.raises(ValidationError):
        GroundingEvidence(request=request, result=result)
