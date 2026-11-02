"""Tests for model-backed grounding orchestration and validation."""

import pytest

from yewo.devices.screen import SystemRegionKind
from yewo.grounding import (
    GroundedPoint,
    GroundedRegion,
    GroundingDisposition,
    GroundingCorrection,
    GroundingInputError,
    GroundingOutputShape,
    GroundingService,
    GroundingTargetOperation,
    VisualGrounder,
    PixelPoint,
)
from yewo.models import ImageModelContent, ModelCallDisposition, TextModelContent
from yewo.observations import ScreenRegion
from grounding.support import (
    DeterministicGrounder,
    StubModelClient,
    build_grounding_profile,
    build_grounding_request,
)


def test_service_sends_only_target_context_and_preserves_call_evidence() -> None:
    client = StubModelClient("(150,300)")
    service = GroundingService(
        client,
        build_grounding_profile(),
        identifier_factory=lambda: "model-request-1",
    )

    result = service.ground(build_grounding_request())

    assert result.disposition is GroundingDisposition.ACCEPTED
    assert isinstance(result.location, GroundedPoint)
    assert result.location.pixel.model_dump() == {"x": 150, "y": 300}
    assert result.source_screenshot_sha256 == "a" * 64
    assert result.served_model == "example/grounder"
    assert result.upstream_provider == "example-provider"
    assert result.raw_response_artifact is not None
    assert result.raw_response_artifact.path.as_posix() == "assets/models/calls.jsonl"
    assert result.latency_ms == 25
    assert result.usage.total_tokens == 105

    request = client.requests[0]
    assert request.profile_id == "grounder-primary"
    assert request.role.value == "grounder"
    assert request.response_contract is None
    text_content, image_content = request.messages[0].content
    assert isinstance(text_content, TextModelContent)
    assert "pink add button" in text_content.text
    assert "Screenshot size: 200 by 400 pixels" in text_content.text
    assert "Locate the named target itself" in text_content.text
    assert "Do not explain the answer" in text_content.text
    assert "goal" not in text_content.text.lower()
    assert isinstance(image_content, ImageModelContent)
    assert image_content.screenshot.sha256 == "a" * 64


def test_service_rejects_platform_region_without_changing_transport_disposition() -> None:
    client = StubModelClient("(10,20)")
    service = GroundingService(client, build_grounding_profile())
    request = build_grounding_request(
        excluded_regions=(
            ScreenRegion(
                kind=SystemRegionKind.STATUS_BAR,
                left=0,
                top=0,
                right=200,
                bottom=40,
            ),
        )
    )

    result = service.ground(request)

    assert result.disposition is GroundingDisposition.OUT_OF_BOUNDS
    assert result.transport_disposition is ModelCallDisposition.COMPLETED
    assert result.location is None
    assert result.reason == "The model target is inside an excluded platform region."


def test_service_tells_grounder_about_failed_point_and_rejects_an_equivalent_retry() -> None:
    client = StubModelClient("(100,199)")
    service = GroundingService(client, build_grounding_profile())
    request = build_grounding_request(
        correction=GroundingCorrection(
            previous_point=PixelPoint(x=100, y=200),
            minimum_distance_px=8,
        )
    )

    result = service.ground(request)

    assert result.disposition is GroundingDisposition.REPEATED_POINT
    assert result.location is None
    assert result.reason == "The correction repeated the previous ineffective target point."
    prompt = client.requests[0].messages[0].content[0]
    assert isinstance(prompt, TextModelContent)
    assert "previous dispatched point (100,200)" in prompt.text
    assert "at least 8 source screenshot pixels away" in prompt.text


@pytest.mark.parametrize(
    ("transport", "grounding"),
    [
        (ModelCallDisposition.REFUSED, GroundingDisposition.REFUSED),
        (ModelCallDisposition.EMPTY, GroundingDisposition.MALFORMED),
        (ModelCallDisposition.TRUNCATED, GroundingDisposition.MALFORMED),
        (ModelCallDisposition.MALFORMED, GroundingDisposition.MALFORMED),
        (ModelCallDisposition.PROVIDER_ERROR, GroundingDisposition.MALFORMED),
    ],
)
def test_service_preserves_transport_failure_while_refusing_a_location(
    transport: ModelCallDisposition,
    grounding: GroundingDisposition,
) -> None:
    client = StubModelClient(None, transport, "The model call did not complete.")

    result = GroundingService(client, build_grounding_profile()).ground(build_grounding_request())

    assert result.disposition is grounding
    assert result.transport_disposition is transport
    assert result.location is None


def test_service_rejects_a_shape_not_supported_by_the_profile_before_calling_model() -> None:
    client = StubModelClient("(10,20)")
    request = build_grounding_request(
        operation=GroundingTargetOperation.SCROLL_REGION,
        output_shape=GroundingOutputShape.REGION,
    )

    with pytest.raises(GroundingInputError, match="region output"):
        GroundingService(client, build_grounding_profile()).ground(request)

    assert client.requests == []


def test_second_model_profile_uses_the_same_service_with_its_coordinate_convention() -> None:
    client = StubModelClient("(0.5,0.25)")
    profile = build_grounding_profile(
        profile_id="visual-grounder-secondary",
        model_profile_id="grounder-secondary",
        coordinate_convention="normalized",
    )

    result = GroundingService(client, profile).ground(build_grounding_request())

    assert isinstance(result.location, GroundedPoint)
    assert result.model_profile_id == "grounder-secondary"
    assert result.location.pixel.model_dump() == {"x": 100, "y": 100}
    prompt = client.requests[0].messages[0].content[0]
    assert isinstance(prompt, TextModelContent)
    assert "normalized numbers from 0 to 1" in prompt.text
    assert "refused: target not found" in prompt.text.lower()
    assert "ambiguous: target is not unique" in prompt.text.lower()


def test_service_resolves_a_scroll_region_without_another_codec() -> None:
    client = StubModelClient("(20,40,180,360)")
    profile = build_grounding_profile(
        supported_shapes=frozenset(GroundingOutputShape),
    )
    request = build_grounding_request(
        operation=GroundingTargetOperation.SCROLL_REGION,
        output_shape=GroundingOutputShape.REGION,
    )

    result = GroundingService(client, profile).ground(request)

    assert result.disposition is GroundingDisposition.ACCEPTED
    assert isinstance(result.location, GroundedRegion)
    assert result.location.pixel.model_dump() == {
        "left": 20,
        "top": 40,
        "right": 180,
        "bottom": 360,
    }


def test_deterministic_grounder_implements_the_public_contract_offline() -> None:
    grounder: VisualGrounder = DeterministicGrounder(50, 100)

    result = grounder.ground(build_grounding_request())

    assert result.disposition is GroundingDisposition.ACCEPTED
    assert isinstance(result.location, GroundedPoint)
    assert result.location.pixel.model_dump() == {"x": 50, "y": 100}
