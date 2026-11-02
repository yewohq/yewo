"""Model-backed visual grounding and deterministic validation."""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal

from yewo.artifacts import new_record_identifier
from yewo.grounding.codecs import CoordinateTextCodec, DecodedGrounding
from yewo.grounding.records import (
    GroundedPoint,
    GroundingCoordinateConvention,
    GroundingDisposition,
    GroundingOutputShape,
    GroundingProfile,
    GroundingRequest,
    GroundingResult,
    validate_grounding_request,
)
from yewo.models import (
    ImageModelContent,
    ModelArtifactReference,
    ModelCallDisposition,
    ModelCallRecord,
    ModelCallUsage,
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
        """Return the provider calls required to resolve one request."""
        validate_grounding_request(self._profile, request)
        return 1

    def ground(self, request: GroundingRequest) -> GroundingResult:
        """Resolve and validate one referring expression."""
        validate_grounding_request(self._profile, request)
        model_request = self._build_model_request(request)
        response = self._model_client.complete(model_request)
        decoded = self._decode_response(response, request)
        validated = self._validate_correction_distance(decoded, request)
        validated = self._validate_excluded_regions(validated, request)
        return self._build_result(request, response, validated)

    def _build_model_request(self, request: GroundingRequest) -> ModelRequest:
        coordinate_instruction = (
            "source screenshot pixels"
            if self._profile.coordinate_convention is GroundingCoordinateConvention.SOURCE_PIXELS
            else "normalized numbers from 0 to 1"
        )
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
            profile_id=self._profile.model_profile_id,
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
    ) -> DecodedGrounding:
        if response.disposition is ModelCallDisposition.COMPLETED:
            if response.text is None:
                raise GroundingEvidenceError("A completed model response contains no text.")
            return self._codec.decode(
                response.text,
                self._profile.coordinate_convention,
                request.screenshot.width_px,
                request.screenshot.height_px,
                request.output_shape,
            )
        if response.disposition is ModelCallDisposition.REFUSED:
            disposition = GroundingDisposition.REFUSED
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
        final_call = response.calls[-1]
        return GroundingResult(
            grounding_id=request.grounding_id,
            profile_id=self._profile.profile_id,
            model_profile_id=self._profile.model_profile_id,
            source_screenshot_sha256=request.screenshot.sha256,
            output_shape=request.output_shape,
            disposition=decoded.disposition,
            location=decoded.location,
            reason=decoded.reason,
            requested_provider=final_call.requested_provider,
            requested_model=final_call.requested_model,
            requested_revision=final_call.requested_revision,
            served_model=final_call.served_model,
            upstream_provider=final_call.upstream_provider,
            transport_disposition=response.disposition,
            raw_response_artifact=_require_call_artifact(final_call),
            latency_ms=_total_latency(response.calls),
            usage=_total_usage(response.calls),
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


def _regions_overlap(first: PixelRegion, second: PixelRegion) -> bool:
    return (
        first.left < second.right
        and second.left < first.right
        and first.top < second.bottom
        and second.top < first.bottom
    )


def _require_call_artifact(call: ModelCallRecord) -> ModelArtifactReference:
    if call.call_artifact is None:
        raise GroundingEvidenceError("The model call has no persisted response artifact.")
    return call.call_artifact


def _total_latency(calls: tuple[ModelCallRecord, ...]) -> float | None:
    latency_values = tuple(call.latency_ms for call in calls)
    if any(value is None for value in latency_values):
        return None
    return sum(value for value in latency_values if value is not None)


def _total_usage(calls: tuple[ModelCallRecord, ...]) -> ModelCallUsage:
    usage_records = tuple(call.usage for call in calls)
    return ModelCallUsage(
        prompt_tokens=_sum_optional_values(tuple(usage.prompt_tokens for usage in usage_records)),
        completion_tokens=_sum_optional_values(
            tuple(usage.completion_tokens for usage in usage_records)
        ),
        total_tokens=_sum_optional_values(tuple(usage.total_tokens for usage in usage_records)),
        cached_tokens=_sum_optional_values(tuple(usage.cached_tokens for usage in usage_records)),
        cache_write_tokens=_sum_optional_values(
            tuple(usage.cache_write_tokens for usage in usage_records)
        ),
        reasoning_tokens=_sum_optional_values(
            tuple(usage.reasoning_tokens for usage in usage_records)
        ),
        prompt_image_tokens=_sum_optional_values(
            tuple(usage.prompt_image_tokens for usage in usage_records)
        ),
        prompt_text_tokens=_sum_optional_values(
            tuple(usage.prompt_text_tokens for usage in usage_records)
        ),
        cost_usd=_sum_cost(usage_records),
    )


def _sum_optional_values(values: tuple[int | None, ...]) -> int | None:
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def _sum_cost(usage_records: tuple[ModelCallUsage, ...]) -> Decimal | None:
    costs = tuple(usage.cost_usd for usage in usage_records)
    if any(cost is None for cost in costs):
        return None
    return sum((cost for cost in costs if cost is not None), start=Decimal())
