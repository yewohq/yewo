"""Deterministic builders for grounding tests."""

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel

from yewo.artifacts import ArtifactRetention
from yewo.grounding import (
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
)
from yewo.models import (
    ModelArtifactKind,
    ModelArtifactReference,
    ModelCallDisposition,
    ModelCallRecord,
    ModelCallUsage,
    ModelRequest,
    ModelResponse,
    ModelRole,
)
from observations.factories import build_screenshot


def build_grounding_profile(**updates: object) -> GroundingProfile:
    values: dict[str, object] = {
        "profile_id": "visual-grounder-primary",
        "model_profile_id": "grounder-primary",
        "coordinate_convention": GroundingCoordinateConvention.SOURCE_PIXELS,
        "supported_operations": frozenset(GroundingTargetOperation),
        "supported_shapes": frozenset({GroundingOutputShape.POINT}),
    }
    values.update(updates)
    return GroundingProfile.model_validate(values)


def build_grounding_request(**updates: object) -> GroundingRequest:
    values: dict[str, object] = {
        "grounding_id": "grounding-1",
        "screenshot": build_screenshot(),
        "expression": "the visible create control in the lower-right corner",
        "operation": GroundingTargetOperation.TAP,
        "output_shape": GroundingOutputShape.POINT,
    }
    values.update(updates)
    return GroundingRequest.model_validate(values)


def build_call_artifact(artifact_id: str = "call-artifact-1") -> ModelArtifactReference:
    return ModelArtifactReference(
        artifact_id=artifact_id,
        kind=ModelArtifactKind.CALL,
        sha256="b" * 64,
        body_sha256="c" * 64,
        content_sha256="d" * 64,
        path=Path("models/calls.jsonl"),
        body_retained=True,
        retention=ArtifactRetention.RUN,
    )


def build_call(
    request_id: str,
    *,
    profile_id: str = "grounder-primary",
    attempt_number: int = 1,
    disposition: ModelCallDisposition = ModelCallDisposition.COMPLETED,
    latency_ms: float | None = 25,
    cost_usd: Decimal | None = Decimal("0.001"),
) -> ModelCallRecord:
    return ModelCallRecord(
        request_id=request_id,
        profile_id=profile_id,
        role=ModelRole.GROUNDER,
        requested_provider="openrouter",
        requested_model="openrouter/example/grounder",
        requested_revision="revision-1",
        served_model="example/grounder",
        upstream_provider="example-provider",
        attempt_number=attempt_number,
        provider_attempt_number=attempt_number,
        started_at=datetime(2026, 10, 30, tzinfo=timezone.utc),
        latency_ms=latency_ms,
        disposition=disposition,
        usage=ModelCallUsage(
            prompt_tokens=100,
            completion_tokens=5,
            total_tokens=105,
            cached_tokens=20,
            cost_usd=cost_usd,
        ),
        call_artifact=build_call_artifact(f"call-artifact-{attempt_number}"),
    )


class StubModelClient:
    """Return configured model content and retain the received request."""

    def __init__(
        self,
        text: str | None,
        disposition: ModelCallDisposition = ModelCallDisposition.COMPLETED,
        reason: str | None = None,
    ) -> None:
        self.text = text
        self.disposition = disposition
        self.reason = reason
        self.requests: list[ModelRequest] = []

    def complete(
        self,
        request: ModelRequest,
        response_model: type[BaseModel] | None = None,
        validation_context: dict[str, object] | None = None,
    ) -> ModelResponse:
        del response_model, validation_context
        self.requests.append(request)
        return ModelResponse(
            request_id=request.request_id,
            disposition=self.disposition,
            text=self.text,
            reason=self.reason,
            calls=(
                build_call(
                    request.request_id,
                    profile_id=request.profile_id,
                    disposition=self.disposition,
                ),
            ),
            retention=request.retention,
        )


class DeterministicGrounder:
    """Implement the visual-grounder contract without a model or network."""

    def __init__(self, x_px: int, y_px: int) -> None:
        self._x_px = x_px
        self._y_px = y_px

    def required_model_requests(self, request: GroundingRequest) -> int:
        del request
        return 1

    def ground(self, request: GroundingRequest) -> GroundingResult:
        width_px = request.screenshot.width_px
        height_px = request.screenshot.height_px
        return GroundingResult(
            grounding_id=request.grounding_id,
            source_screenshot_sha256=request.screenshot.sha256,
            output_shape=request.output_shape,
            disposition=GroundingDisposition.ACCEPTED,
            resolution=ModelGroundingResolution(
                grounding_profile_id="deterministic-grounder",
                model_request_ids=(request.grounding_id,),
            ),
            location=GroundedPoint(
                normalized=NormalizedPoint(
                    x=self._x_px / (width_px - 1),
                    y=self._y_px / (height_px - 1),
                ),
                pixel=PixelPoint(x=self._x_px, y=self._y_px),
            ),
        )


class ExpressionGrounder:
    """Choose distinct deterministic points from the referring expression."""

    def required_model_requests(self, request: GroundingRequest) -> int:
        del request
        return 1

    def ground(self, request: GroundingRequest) -> GroundingResult:
        x_px = 20 if "first" in request.expression else 150
        return DeterministicGrounder(x_px, 200).ground(request)


class FailingGrounder:
    def required_model_requests(self, request: GroundingRequest) -> int:
        del request
        return 1

    def ground(self, request: GroundingRequest) -> GroundingResult:
        del request
        raise RuntimeError("grounding failed")
