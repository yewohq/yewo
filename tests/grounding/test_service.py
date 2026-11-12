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
    GroundingResolutionSource,
    GroundingTargetOperation,
    ModelGroundingResolution,
    NormalizedRegion,
    ScreenAnalysisGroundingCandidate,
    ScreenAnalysisGroundingResolution,
    VisualGrounder,
    PixelPoint,
)
from yewo.grounding.records import GroundingCoordinateConvention
from yewo.grounding.service import (
    _bounds_instruction,
    _correction_instruction,
    _excluded_region_instruction,
    _operation_instruction,
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
    assert result.resolution == ModelGroundingResolution(
        grounding_profile_id="visual-grounder-primary",
        model_request_ids=("model-request-1",),
    )

    request = client.requests[0]
    assert request.profile_id == "grounder-primary"
    assert request.role.value == "grounder"
    assert request.response_contract is None
    text_content, image_content = request.messages[0].content
    assert isinstance(text_content, TextModelContent)
    assert "visible create control" in text_content.text
    assert "Screenshot size: 200 by 400 pixels" in text_content.text
    assert "Locate the named target itself" in text_content.text
    assert "Return the coordinates alone." in text_content.text
    assert "A valid point satisfies 0 <= x < 200 and 0 <= y < 400." in text_content.text
    assert "The target is the activating surface of the named control." in text_content.text
    assert "region" not in text_content.text.lower()
    assert "goal" not in text_content.text.lower()
    assert isinstance(image_content, ImageModelContent)
    assert image_content.screenshot.sha256 == "a" * 64


def test_service_uses_high_confidence_screen_analysis_without_a_model_call() -> None:
    client = StubModelClient("(10,20)")
    service = GroundingService(client, build_grounding_profile())
    request = build_grounding_request(
        screen_analysis_candidate=ScreenAnalysisGroundingCandidate(
            analysis_id="analysis-1",
            element_ref="element-1",
            screenshot_sha256="a" * 64,
            region=NormalizedRegion(left=0.25, top=0.25, right=0.75, bottom=0.75),
        )
    )

    assert service.required_model_requests(request) == 0
    result = service.ground(request)

    assert result.resolution_source is GroundingResolutionSource.SCREEN_ANALYSIS
    assert result.resolution == ScreenAnalysisGroundingResolution(
        analysis_id="analysis-1",
        element_ref="element-1",
    )
    assert isinstance(result.location, GroundedPoint)
    assert result.location.pixel == PixelPoint(x=100, y=200)
    assert result.model_request_count == 0
    assert client.requests == []


def test_service_falls_back_when_analysis_region_points_into_system_ui() -> None:
    client = StubModelClient("(100,200)")
    service = GroundingService(client, build_grounding_profile())
    request = build_grounding_request(
        excluded_regions=(
            ScreenRegion(
                kind=SystemRegionKind.STATUS_BAR,
                left=0,
                top=0,
                right=200,
                bottom=80,
            ),
        ),
        screen_analysis_candidate=ScreenAnalysisGroundingCandidate(
            analysis_id="analysis-1",
            element_ref="element-1",
            screenshot_sha256="a" * 64,
            region=NormalizedRegion(left=0.25, top=0, right=0.75, bottom=0.1),
        ),
    )

    assert service.required_model_requests(request) == 1
    result = service.ground(request)

    assert result.resolution_source is GroundingResolutionSource.MODEL
    assert len(client.requests) == 1


@pytest.mark.parametrize(
    "region",
    [
        NormalizedRegion(left=0, top=0, right=1, bottom=1),
        NormalizedRegion(left=0.49, top=0, right=0.495, bottom=1),
    ],
)
def test_service_falls_back_for_an_unlocalized_analysis_region(
    region: NormalizedRegion,
) -> None:
    client = StubModelClient("(100,200)")
    service = GroundingService(client, build_grounding_profile())
    request = build_grounding_request(
        screen_analysis_candidate=ScreenAnalysisGroundingCandidate(
            analysis_id="analysis-1",
            element_ref="element-1",
            screenshot_sha256="a" * 64,
            region=region,
        )
    )

    assert service.required_model_requests(request) == 1
    result = service.ground(request)

    assert result.resolution_source is GroundingResolutionSource.MODEL
    assert len(client.requests) == 1


def test_service_rejects_a_model_point_inside_a_platform_region() -> None:
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
    assert isinstance(result.resolution, ModelGroundingResolution)
    assert result.resolution.grounding_profile_id == "visual-grounder-primary"
    assert len(result.resolution.model_request_ids) == 1
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
    assert "A tap at (100,200) produced no observable change" in prompt.text
    assert "at least 8 source screenshot pixels from the previous point" in prompt.text


@pytest.mark.parametrize(
    ("transport", "grounding"),
    [
        (ModelCallDisposition.REFUSED, GroundingDisposition.REFUSED),
        (ModelCallDisposition.EMPTY, GroundingDisposition.MALFORMED),
        (ModelCallDisposition.TRUNCATED, GroundingDisposition.MALFORMED),
        (ModelCallDisposition.MALFORMED, GroundingDisposition.MALFORMED),
        (
            ModelCallDisposition.PROVIDER_ERROR,
            GroundingDisposition.PROVIDER_UNAVAILABLE,
        ),
    ],
)
def test_service_maps_incomplete_model_calls_to_grounding_dispositions(
    transport: ModelCallDisposition,
    grounding: GroundingDisposition,
) -> None:
    client = StubModelClient(None, transport, "The model call did not complete.")

    result = GroundingService(client, build_grounding_profile()).ground(build_grounding_request())

    assert result.disposition is grounding
    assert result.resolution_source is GroundingResolutionSource.MODEL
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
    assert isinstance(result.resolution, ModelGroundingResolution)
    assert client.requests[0].profile_id == "grounder-secondary"
    assert result.location.pixel.model_dump() == {"x": 100, "y": 100}
    prompt = client.requests[0].messages[0].content[0]
    assert isinstance(prompt, TextModelContent)
    assert "normalized numbers from 0 to 1" in prompt.text
    assert "refused: target not found" in prompt.text.lower()
    assert "ambiguous: target is not unique" in prompt.text.lower()


@pytest.mark.parametrize(
    ("convention", "number_format", "answer", "pixel"),
    [
        (
            GroundingCoordinateConvention.SOURCE_PIXELS,
            "Use whole numbers.",
            "(150,300)",
            {"x": 150, "y": 300},
        ),
        (
            GroundingCoordinateConvention.NORMALIZED_1000,
            "Use whole numbers.",
            "(500,250)",
            {"x": 100, "y": 100},
        ),
        (
            GroundingCoordinateConvention.NORMALIZED,
            "Use decimal numbers.",
            "(0.5,0.25)",
            {"x": 100, "y": 100},
        ),
    ],
)
def test_each_convention_instructs_the_number_format_its_own_space_admits(
    convention: GroundingCoordinateConvention,
    number_format: str,
    answer: str,
    pixel: dict[str, int],
) -> None:
    # One shared whole-number instruction contradicted the normalized space, where 0 and 1 were
    # then the only compliant answers and the whole screen was the only compliant region.
    client = StubModelClient(answer)
    profile = build_grounding_profile(coordinate_convention=convention)

    result = GroundingService(client, profile).ground(build_grounding_request())

    prompt = client.requests[0].messages[0].content[0]
    assert isinstance(prompt, TextModelContent)
    assert number_format in prompt.text
    assert ("Use decimal numbers." in prompt.text) is (
        convention is GroundingCoordinateConvention.NORMALIZED
    )
    assert isinstance(result.location, GroundedPoint)
    assert result.location.pixel.model_dump() == pixel


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


def test_platform_region_instruction_states_bounds_and_its_coordinate_space() -> None:
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

    pixels = _excluded_region_instruction(request, GroundingCoordinateConvention.SOURCE_PIXELS)
    normalized = _excluded_region_instruction(
        request, GroundingCoordinateConvention.NORMALIZED_1000
    )

    assert "(0,0,200,40)" in pixels
    assert "Convert" not in pixels
    assert "(0,0,200,40)" in normalized
    assert "Convert those source pixels" in normalized


def test_no_platform_region_instruction_without_excluded_regions() -> None:
    instruction = _excluded_region_instruction(
        build_grounding_request(), GroundingCoordinateConvention.NORMALIZED_1000
    )

    assert instruction == ""


def test_correction_instruction_states_its_coordinate_space() -> None:
    request = build_grounding_request(
        correction=GroundingCorrection(
            previous_point=PixelPoint(x=100, y=200),
            minimum_distance_px=8,
        )
    )

    pixels = _correction_instruction(request, GroundingCoordinateConvention.SOURCE_PIXELS)
    normalized = _correction_instruction(request, GroundingCoordinateConvention.NORMALIZED)

    assert "at least 8 source screenshot pixels from the previous point" in pixels
    assert "Convert" not in pixels
    assert "Convert those source pixels" in normalized


@pytest.mark.parametrize(
    ("shape", "convention", "expected"),
    [
        (
            GroundingOutputShape.POINT,
            GroundingCoordinateConvention.SOURCE_PIXELS,
            "A valid point satisfies 0 <= x < 200 and 0 <= y < 400.",
        ),
        (
            GroundingOutputShape.POINT,
            GroundingCoordinateConvention.NORMALIZED_1000,
            "A valid point satisfies 0 <= x <= 1000 and 0 <= y <= 1000.",
        ),
        (
            GroundingOutputShape.POINT,
            GroundingCoordinateConvention.NORMALIZED,
            "A valid point satisfies 0 <= x <= 1 and 0 <= y <= 1.",
        ),
        (
            GroundingOutputShape.REGION,
            GroundingCoordinateConvention.SOURCE_PIXELS,
            "A valid region satisfies 0 <= left < right <= 200 and 0 <= top < bottom <= 400, "
            "so equal opposite edges are invalid.",
        ),
        (
            GroundingOutputShape.REGION,
            GroundingCoordinateConvention.NORMALIZED_1000,
            "A valid region satisfies 0 <= left < right <= 1000 and 0 <= top < bottom <= 1000, "
            "so equal opposite edges are invalid. The region must still cover at least one whole "
            "pixel after scaling.",
        ),
    ],
)
def test_bounds_instruction_states_the_codec_rule_for_the_requested_shape(
    shape: GroundingOutputShape,
    convention: GroundingCoordinateConvention,
    expected: str,
) -> None:
    request = build_grounding_request(
        output_shape=shape,
        operation=(
            GroundingTargetOperation.SCROLL_REGION
            if shape is GroundingOutputShape.REGION
            else GroundingTargetOperation.TAP
        ),
    )

    assert _bounds_instruction(request, convention) == expected


def test_service_tells_the_grounder_which_platform_regions_to_avoid() -> None:
    client = StubModelClient("(150,300)")
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

    assert result.disposition is GroundingDisposition.ACCEPTED
    prompt = client.requests[0].messages[0].content[0]
    assert isinstance(prompt, TextModelContent)
    assert "(0,0,200,40)" in prompt.text
    assert "choose a target outside them" in prompt.text


def test_every_target_operation_has_a_placement_instruction() -> None:
    for operation in GroundingTargetOperation:
        shape = (
            GroundingOutputShape.REGION
            if operation is GroundingTargetOperation.SCROLL_REGION
            else GroundingOutputShape.POINT
        )
        request = build_grounding_request(operation=operation, output_shape=shape)

        assert _operation_instruction(request)


def test_placement_instruction_names_the_surface_the_operation_needs() -> None:
    field = build_grounding_request(operation=GroundingTargetOperation.TEXT_FIELD)
    destination = build_grounding_request(operation=GroundingTargetOperation.DRAG_DESTINATION)

    assert _operation_instruction(field) == (
        "The target is the editable area of the named field, not its label."
    )
    assert _operation_instruction(destination) == (
        "The target is the drop position, which can be empty space rather than a control."
    )


def test_a_correction_asks_for_the_target_not_merely_a_different_point() -> None:
    # Told only to move a minimum distance, a grounder satisfies the instruction by stepping
    # sideways off the control it already missed.
    instruction = _correction_instruction(
        build_grounding_request(
            correction=GroundingCorrection(
                previous_point=PixelPoint(x=100, y=200),
                minimum_distance_px=8,
            )
        ),
        GroundingCoordinateConvention.SOURCE_PIXELS,
    )

    assert "was not on the target" in instruction
    assert "centre of the surface it names" in instruction
    assert "Do not offset from the previous point" in instruction
