"""Model-backed visual grounding and deterministic validation."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from yewo.artifacts import new_record_identifier
from yewo.grounding.codecs import CoordinateTextCodec, DecodedGrounding
from yewo.grounding.records import (
    GroundingCandidate,
    GroundedPoint,
    GroundingCoordinateConvention,
    GroundingDisposition,
    GroundingOutputShape,
    GroundingProfile,
    GroundingRequest,
    GroundingResult,
    GroundingTargetOperation,
    ModelGroundingResolution,
    NormalizedPoint,
    PixelPoint,
    ScreenAnalysisGroundingResolution,
    validate_grounding_request,
)
from yewo.models import (
    ImageModelContent,
    ModelCallDisposition,
    ModelClient,
    ModelMessage,
    ModelMessageRole,
    ModelRequest,
    ModelResponse,
    ModelRole,
    TextModelContent,
)
from yewo.grounding.service_prompt import GROUNDING_PROMPT
from yewo.observations import PixelRegion

MINIMUM_DIRECT_ANALYSIS_REGION_EDGE_PX = 8
MAXIMUM_DIRECT_ANALYSIS_REGION_AREA_RATIO = 0.5


def _coordinate_instruction(convention: GroundingCoordinateConvention) -> str:
    if convention is GroundingCoordinateConvention.SOURCE_PIXELS:
        return "source screenshot pixels"
    if convention is GroundingCoordinateConvention.NORMALIZED_1000:
        return "integers from 0 to 1000 relative to the screenshot width and height"
    return "normalized numbers from 0 to 1"


class GroundingEvidenceError(RuntimeError):
    """Raised when a model response lacks its required call evidence."""


class GroundingService:
    """Resolve visual targets through one configured model profile."""

    def __init__(
        self,
        model_client: ModelClient,
        profile: GroundingProfile,
        *,
        identifier_factory: Callable[[], str] = new_record_identifier,
    ) -> None:
        self._model_client = model_client
        self._profile = profile
        self._codec = CoordinateTextCodec()
        self._identifier_factory = identifier_factory

    def required_model_requests(self, request: GroundingRequest) -> int:
        """Return the maximum logical model calls needed to resolve one request."""
        validate_grounding_request(self._profile, request)
        if self._screen_analysis_result(request) is not None:
            return 0
        return len(self._profile.candidates)

    def ground(self, request: GroundingRequest) -> GroundingResult:
        """Resolve one target with bounded, observation-corroborated recovery."""
        validate_grounding_request(self._profile, request)
        analysis_result = self._screen_analysis_result(request)
        if analysis_result is not None:
            return analysis_result
        attempts: list[GroundingResult] = []
        for candidate in self._profile.candidates:
            result = self._ground_with_candidate(request, candidate)
            attempts.append(result)
            if result.disposition is not GroundingDisposition.PROVIDER_UNAVAILABLE:
                return _aggregate_results(attempts)
        if not attempts:
            raise GroundingEvidenceError("A grounding profile declares no candidate grounder.")
        return _aggregate_results(attempts)

    def _screen_analysis_result(self, request: GroundingRequest) -> GroundingResult | None:
        candidate = request.screen_analysis_candidate
        if candidate is None or not _is_direct_analysis_region_eligible(request):
            return None
        normalized_point = NormalizedPoint(
            x=(candidate.region.left + candidate.region.right) / 2,
            y=(candidate.region.top + candidate.region.bottom) / 2,
        )
        location = GroundedPoint(
            normalized=normalized_point,
            pixel=PixelPoint(
                x=round(normalized_point.x * (request.screenshot.width_px - 1)),
                y=round(normalized_point.y * (request.screenshot.height_px - 1)),
            ),
        )
        validated_target = self._validate_excluded_regions(
            DecodedGrounding(GroundingDisposition.ACCEPTED, location=location),
            request,
        )
        if validated_target.disposition is not GroundingDisposition.ACCEPTED:
            return None
        return GroundingResult(
            grounding_id=request.grounding_id,
            source_screenshot_sha256=request.screenshot.sha256,
            output_shape=request.output_shape,
            disposition=GroundingDisposition.ACCEPTED,
            resolution=ScreenAnalysisGroundingResolution(
                analysis_id=candidate.analysis_id,
                element_ref=candidate.element_ref,
            ),
            location=location,
        )

    def _ground_with_candidate(
        self,
        request: GroundingRequest,
        candidate: GroundingCandidate,
    ) -> GroundingResult:
        model_request = self._build_model_request(request, candidate)
        response = self._model_client.complete(model_request)
        decoded = self._decode_response(response, request, candidate)
        validated_target = self._validate_correction_distance(decoded, request)
        validated_target = self._validate_excluded_regions(validated_target, request)
        return self._build_result(request, response, validated_target)

    def _build_model_request(
        self,
        request: GroundingRequest,
        candidate: GroundingCandidate,
    ) -> ModelRequest:
        coordinate_instruction = _coordinate_instruction(candidate.coordinate_convention)
        response_instruction = (
            "one coordinate as (x,y)"
            if request.output_shape is GroundingOutputShape.POINT
            else "one region as (left,top,right,bottom)"
        )
        location_instruction = (
            "Choose a stable point inside the named target, near the center of that target."
            if request.output_shape is GroundingOutputShape.POINT
            else "Bound the named visible region without including surrounding controls."
        )
        prompt = GROUNDING_PROMPT.format(
            operation=request.operation.value,
            expression=request.expression,
            width_px=request.screenshot.width_px,
            height_px=request.screenshot.height_px,
            location_instruction=location_instruction,
            correction_instruction=_correction_instruction(
                request, candidate.coordinate_convention
            ),
            excluded_region_instruction=_excluded_region_instruction(
                request, candidate.coordinate_convention
            ),
            operation_instruction=_operation_instruction(request),
            bounds_instruction=_bounds_instruction(request, candidate.coordinate_convention),
            response_shape=response_instruction,
            coordinate_space=coordinate_instruction,
        )
        return ModelRequest(
            request_id=self._identifier_factory(),
            profile_id=candidate.model_profile_id,
            role=ModelRole.GROUNDER,
            messages=(
                ModelMessage(
                    role=ModelMessageRole.USER,
                    content=(
                        TextModelContent(text=prompt),
                        ImageModelContent(screenshot=request.screenshot),
                    ),
                ),
            ),
            sensitivity=request.screenshot.sensitivity,
            retention=request.screenshot.retention,
        )

    def _decode_response(
        self,
        response: ModelResponse,
        request: GroundingRequest,
        candidate: GroundingCandidate,
    ) -> DecodedGrounding:
        if response.disposition is ModelCallDisposition.COMPLETED:
            if response.text is None:
                raise GroundingEvidenceError("A completed model response contains no text.")
            return self._codec.decode(
                response.text,
                candidate.coordinate_convention,
                request.screenshot.width_px,
                request.screenshot.height_px,
                request.output_shape,
            )
        if response.disposition is ModelCallDisposition.REFUSED:
            disposition = GroundingDisposition.REFUSED
        elif response.disposition is ModelCallDisposition.PROVIDER_ERROR:
            disposition = GroundingDisposition.PROVIDER_UNAVAILABLE
        else:
            disposition = GroundingDisposition.MALFORMED
        return DecodedGrounding(
            disposition,
            reason=response.reason or "The model did not return a usable grounding response.",
        )

    @staticmethod
    def _validate_correction_distance(
        decoded: DecodedGrounding,
        request: GroundingRequest,
    ) -> DecodedGrounding:
        if not isinstance(decoded.location, GroundedPoint) or request.correction is None:
            return decoded
        point = decoded.location.pixel
        previous = request.correction.previous_point
        distance_squared = (point.x - previous.x) ** 2 + (point.y - previous.y) ** 2
        if distance_squared >= request.correction.minimum_distance_px**2:
            return decoded
        return DecodedGrounding(
            GroundingDisposition.REPEATED_POINT,
            reason="The correction repeated the previous ineffective target point.",
        )

    @staticmethod
    def _validate_excluded_regions(
        decoded: DecodedGrounding,
        request: GroundingRequest,
    ) -> DecodedGrounding:
        if decoded.location is None or request.excluded_regions is None:
            return decoded
        if isinstance(decoded.location, GroundedPoint):
            x_px = decoded.location.pixel.x
            y_px = decoded.location.pixel.y
            excluded = any(
                region.left <= x_px < region.right and region.top <= y_px < region.bottom
                for region in request.excluded_regions
            )
        else:
            excluded = any(
                _regions_overlap(decoded.location.pixel, region)
                for region in request.excluded_regions
            )
        if not excluded:
            return decoded
        return DecodedGrounding(
            GroundingDisposition.OUT_OF_BOUNDS,
            reason="The model target is inside an excluded platform region.",
        )

    def _build_result(
        self,
        request: GroundingRequest,
        response: ModelResponse,
        decoded: DecodedGrounding,
    ) -> GroundingResult:
        return GroundingResult(
            grounding_id=request.grounding_id,
            source_screenshot_sha256=request.screenshot.sha256,
            output_shape=request.output_shape,
            disposition=decoded.disposition,
            resolution=ModelGroundingResolution(
                grounding_profile_id=self._profile.profile_id,
                model_request_ids=(response.request_id,),
            ),
            location=decoded.location,
            reason=decoded.reason,
        )


_OPERATION_INSTRUCTIONS = {
    GroundingTargetOperation.TAP: "The target is the activating surface of the named control.",
    GroundingTargetOperation.TEXT_FIELD: (
        "The target is the editable area of the named field, not its label."
    ),
    GroundingTargetOperation.DRAG_SOURCE: (
        "The target is the grab position on the item being moved."
    ),
    GroundingTargetOperation.DRAG_DESTINATION: (
        "The target is the drop position, which can be empty space rather than a control."
    ),
    GroundingTargetOperation.SCROLL_REGION: (
        "The target is the scrollable area itself, not one row inside it."
    ),
}


def _operation_instruction(request: GroundingRequest) -> str:
    return _OPERATION_INSTRUCTIONS[request.operation]


def _bounds_instruction(
    request: GroundingRequest,
    convention: GroundingCoordinateConvention,
) -> str:
    source_pixels = convention is GroundingCoordinateConvention.SOURCE_PIXELS
    if source_pixels:
        width = str(request.screenshot.width_px)
        height = str(request.screenshot.height_px)
    else:
        limit = "1000" if convention is GroundingCoordinateConvention.NORMALIZED_1000 else "1"
        width, height = limit, limit
    if request.output_shape is GroundingOutputShape.POINT:
        # A source pixel coordinate indexes the screenshot, so its last valid value is one
        # less than the dimension. A normalized coordinate includes its upper limit.
        if source_pixels:
            return f"A valid point satisfies 0 <= x < {width} and 0 <= y < {height}."
        return f"A valid point satisfies 0 <= x <= {width} and 0 <= y <= {height}."
    rule = (
        f"A valid region satisfies 0 <= left < right <= {width} and "
        f"0 <= top < bottom <= {height}, so equal opposite edges are invalid."
    )
    if source_pixels:
        return rule
    return f"{rule} The region must still cover at least one whole pixel after scaling."


def _pixel_conversion_note(convention: GroundingCoordinateConvention) -> str:
    if convention is GroundingCoordinateConvention.SOURCE_PIXELS:
        return ""
    return " Convert those source pixels into the required output coordinate space."


def _excluded_region_instruction(
    request: GroundingRequest,
    convention: GroundingCoordinateConvention,
) -> str:
    regions = request.excluded_regions or ()
    if not regions:
        return ""
    bounds = ", ".join(
        f"({region.left},{region.top},{region.right},{region.bottom})" for region in regions
    )
    return (
        "The platform owns these source-pixel regions as (left,top,right,bottom): "
        f"{bounds}. A target inside one of them is rejected, so choose a target outside them."
        f"{_pixel_conversion_note(convention)}"
    )


def _correction_instruction(
    request: GroundingRequest,
    convention: GroundingCoordinateConvention,
) -> str:
    if request.correction is None:
        return ""
    point = request.correction.previous_point
    return (
        f"The previous dispatched point ({point.x},{point.y}) produced no observable change. "
        "Use the refined target description. "
        f"Choose a point at least {request.correction.minimum_distance_px} source screenshot "
        f"pixels away from the previous point.{_pixel_conversion_note(convention)}"
    )


def _aggregate_results(results: list[GroundingResult]) -> GroundingResult:
    final = results[-1]
    request_ids = tuple(
        request_id
        for result in results
        if isinstance(result.resolution, ModelGroundingResolution)
        for request_id in result.resolution.model_request_ids
    )
    final_resolution = cast(ModelGroundingResolution, final.resolution)
    return final.model_copy(
        update={
            "resolution": final_resolution.model_copy(update={"model_request_ids": request_ids}),
        }
    )


def _is_direct_analysis_region_eligible(request: GroundingRequest) -> bool:
    candidate = request.screen_analysis_candidate
    if candidate is None:
        return False
    region = candidate.region
    width_px = (region.right - region.left) * request.screenshot.width_px
    height_px = (region.bottom - region.top) * request.screenshot.height_px
    area_ratio = (region.right - region.left) * (region.bottom - region.top)
    return (
        width_px >= MINIMUM_DIRECT_ANALYSIS_REGION_EDGE_PX
        and height_px >= MINIMUM_DIRECT_ANALYSIS_REGION_EDGE_PX
        and area_ratio <= MAXIMUM_DIRECT_ANALYSIS_REGION_AREA_RATIO
    )


def _regions_overlap(first: PixelRegion, second: PixelRegion) -> bool:
    return (
        first.left < second.right
        and second.left < first.right
        and first.top < second.bottom
        and second.top < first.bottom
    )
