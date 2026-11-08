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
from yewo.observations import PixelRegion


GROUNDING_PROMPT = """You are a GUI visual grounder. Locate one visible target in the screenshot.

Target operation: {operation}
Referring expression: {expression}
Screenshot size: {width_px} by {height_px} pixels.

Localization rules:
- Locate the named target itself, not a nearby item, blank space, or parent container.
- Use all stated visual attributes, relationships, and scope to reject similar targets.
- A text label and the control or row that contains it are different targets.
- {location_instruction}
{correction_instruction}

Output only {response_shape} in {coordinate_space}. Do not explain the answer.
If no visible target matches, return: refused: target not found
If more than one visible target matches, return: ambiguous: target is not unique"""


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
            correction_instruction=_correction_instruction(request),
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


def _correction_instruction(request: GroundingRequest) -> str:
    if request.correction is None:
        return ""
    point = request.correction.previous_point
    return (
        f"The previous dispatched point ({point.x},{point.y}) produced no observable change. "
        "Use the refined target description. "
        f"Choose a point at least {request.correction.minimum_distance_px} source screenshot "
        "pixels away from the previous point."
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
