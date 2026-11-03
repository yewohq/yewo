"""Exact procedure grounding with memory and model fallback."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Protocol

from yewo.artifacts import RecordIdentifier
from yewo.grounding import (
    GroundedLocation,
    GroundedPoint,
    GroundedRegion,
    GroundingDisposition,
    GroundingEvidence,
    GroundingRequest,
    GroundingResolutionSource,
    GroundingResult,
    NormalizedPoint,
    PixelPoint,
    VisualGrounder,
)
from yewo.models import ModelCallUsage
from yewo.observations import DeviceOrientation, ScreenFingerprint
from yewo.observations import ObservationCaptureError, ScreenshotReference
from yewo.agent.replay.procedure_records import (
    ProcedureGroundingDecision,
    ProcedureGroundingDecisionCode,
    ProcedureRepairTarget,
)
from yewo.run_narration import current_narrator
from yewo.agent.replay.reconciliation_records import (
    EvidenceDocument,
    ReplayProcedure,
    ReplayProcedureStep,
    TargetVisualDescriptor,
)
from yewo.agent.replay.visual_matching import TargetMatch, match_target_neighborhood
from yewo.observations import Observation


@dataclass(frozen=True)
class ProcedureGroundingAuthorization:
    """Short-lived permission to ground one step on one validated observation."""

    source_step_id: RecordIdentifier
    procedure_index: int
    expected_state_id: RecordIdentifier
    observation_id: RecordIdentifier
    screenshot_artifact_id: RecordIdentifier
    screenshot_sha256: str
    source_fingerprint: ScreenFingerprint
    source_orientation: DeviceOrientation | None
    source_keyboard_visible: bool | None
    width_px: int
    height_px: int
    expected_phash_distance: int | None
    competing_phash_distance: int | None
    source_visual_match_ratio: float | None
    requires_target_visual_match: bool = False


@dataclass(frozen=True)
class ProcedureGroundingMatch:
    """One stored grounding tied to its compiled source step."""

    source_step_id: RecordIdentifier
    evidence: GroundingEvidence
    authorization: ProcedureGroundingAuthorization
    target_visual_descriptor: TargetVisualDescriptor | None


@dataclass(frozen=True)
class _ProcedureGroundingResolution:
    decision: ProcedureGroundingDecisionCode
    location: GroundedLocation | None
    target_match: TargetMatch | None = None


class ProcedureGroundingSource(Protocol):
    """Expose grounding for the action that is pending dispatch."""

    def grounding_match(self, request: GroundingRequest) -> ProcedureGroundingMatch | None: ...


class ProcedureGrounder:
    """Reuse exact procedure grounding and delegate every non-exact request."""

    def __init__(
        self,
        delegate: VisualGrounder,
        source: ProcedureGroundingSource,
        screenshot_loader: Callable[[ScreenshotReference], bytes] | None = None,
    ) -> None:
        self._delegate = delegate
        self._source = source
        self._screenshot_loader = screenshot_loader
        self._decisions: list[ProcedureGroundingDecision] = []
        self._resolutions: dict[RecordIdentifier, _ProcedureGroundingResolution] = {}

    @property
    def decisions(self) -> tuple[ProcedureGroundingDecision, ...]:
        """Return every completed grounding decision in request order."""
        return tuple(self._decisions)

    def required_model_requests(self, request: GroundingRequest) -> int:
        """Return zero only when exact procedure evidence can ground the request."""
        match = self._source.grounding_match(request)
        resolution = self._resolve(match, request)
        return (
            0
            if resolution.location is not None
            else self._delegate.required_model_requests(request)
        )

    def ground(self, request: GroundingRequest) -> GroundingResult:
        """Resolve exact procedure grounding or use the configured fallback chain."""
        match = self._source.grounding_match(request)
        resolution = self._resolve(match, request)
        if resolution.location is None:
            _narrate_lookup(request, reused=False)
            result = self._delegate.ground(request)
            self._record_decision(request, match, resolution, result)
            return result
        if match is None:
            raise AssertionError("Reusable procedure grounding requires one location.")
        stored = match.evidence.result
        _narrate_lookup(request, reused=True)
        result = GroundingResult(
            grounding_id=request.grounding_id,
            profile_id=stored.profile_id,
            model_profile_id=stored.model_profile_id,
            source_screenshot_sha256=request.screenshot.sha256,
            output_shape=request.output_shape,
            disposition=GroundingDisposition.ACCEPTED,
            resolution_source=GroundingResolutionSource.PROCEDURE,
            location=resolution.location,
            requested_provider="procedure",
            requested_model=stored.requested_model,
            requested_revision=stored.requested_revision,
            served_model=stored.served_model,
            upstream_provider=stored.upstream_provider,
            cached_procedure_step_id=match.source_step_id,
            usage=ModelCallUsage(),
            confidence=stored.confidence,
        )
        self._record_decision(request, match, resolution, result)
        return result

    def _resolve(
        self,
        match: ProcedureGroundingMatch | None,
        request: GroundingRequest,
    ) -> _ProcedureGroundingResolution:
        cached = self._resolutions.get(request.grounding_id)
        if cached is not None:
            return cached
        rejection = _reuse_rejection(match, request)
        if rejection is not None:
            resolution = _ProcedureGroundingResolution(rejection, None)
        else:
            resolution = self._resolve_target(match, request)
        self._resolutions[request.grounding_id] = resolution
        return resolution

    def _resolve_target(
        self,
        match: ProcedureGroundingMatch | None,
        request: GroundingRequest,
    ) -> _ProcedureGroundingResolution:
        if match is None or match.evidence.result.location is None:
            raise AssertionError("Authorized procedure grounding requires a location.")
        stored_location = match.evidence.result.location
        descriptor = match.target_visual_descriptor
        if match.authorization.requires_target_visual_match and (
            descriptor is None
            or not isinstance(stored_location, GroundedPoint)
            or self._screenshot_loader is None
        ):
            return _ProcedureGroundingResolution(
                ProcedureGroundingDecisionCode.TARGET_VISUAL_CHANGED,
                None,
            )
        if (
            descriptor is None
            or not isinstance(stored_location, GroundedPoint)
            or self._screenshot_loader is None
        ):
            return _ProcedureGroundingResolution(
                ProcedureGroundingDecisionCode.AUTHORIZED_COORDINATE_REUSED,
                stored_location,
            )
        try:
            png = self._screenshot_loader(request.screenshot)
        except ObservationCaptureError:
            return _ProcedureGroundingResolution(
                ProcedureGroundingDecisionCode.TARGET_VISUAL_CHANGED,
                None,
            )
        target_match = match_target_neighborhood(
            png,
            descriptor,
            stored_location,
            request.screenshot.width_px,
            request.screenshot.height_px,
        )
        if target_match is None:
            return _ProcedureGroundingResolution(
                ProcedureGroundingDecisionCode.TARGET_VISUAL_CHANGED,
                None,
            )
        decision = (
            ProcedureGroundingDecisionCode.AUTHORIZED_COORDINATE_RELOCATED
            if target_match.relocated
            else ProcedureGroundingDecisionCode.AUTHORIZED_COORDINATE_REUSED
        )
        return _ProcedureGroundingResolution(decision, target_match.point, target_match)

    def _record_decision(
        self,
        request: GroundingRequest,
        match: ProcedureGroundingMatch | None,
        resolution: _ProcedureGroundingResolution,
        result: GroundingResult,
    ) -> None:
        authorization = match.authorization if match is not None else None
        stored_location = match.evidence.result.location if match is not None else None
        self._decisions.append(
            ProcedureGroundingDecision(
                grounding_id=request.grounding_id,
                source_step_id=(match.source_step_id if match is not None else None),
                procedure_index=(
                    authorization.procedure_index if authorization is not None else None
                ),
                expected_state_id=(
                    authorization.expected_state_id if authorization is not None else None
                ),
                observation_id=(
                    authorization.observation_id if authorization is not None else None
                ),
                screenshot_sha256=request.screenshot.sha256,
                decision=resolution.decision,
                expected_phash_distance=(
                    authorization.expected_phash_distance
                    if authorization is not None
                    else None
                ),
                competing_phash_distance=(
                    authorization.competing_phash_distance
                    if authorization is not None
                    else None
                ),
                source_visual_match_ratio=(
                    authorization.source_visual_match_ratio
                    if authorization is not None
                    else None
                ),
                target_hash_distance=(
                    resolution.target_match.hash_distance
                    if resolution.target_match is not None
                    else None
                ),
                target_runner_up_distance=(
                    resolution.target_match.runner_up_distance
                    if resolution.target_match is not None
                    else None
                ),
                stored_location=stored_location,
                resolved_location=result.location,
                resolution_source=result.resolution_source,
            )
        )


def build_keyboard_repair_grounding_match(
    proposal_expression: str | None,
    observation: Observation,
    target: ProcedureRepairTarget | None,
    step: ReplayProcedureStep | None,
    evidence: EvidenceDocument,
) -> ProcedureGroundingMatch | None:
    """Authorize one locally verifiable compiled field on a repair observation."""
    if (
        target is None
        or proposal_expression is None
        or target.keyboard_visible is not True
        or observation.keyboard_visible is not False
        or observation.foreground_application_id != target.foreground_application_id
        or observation.platform_context != target.platform_context
        or observation.orientation is not target.orientation
        or step is None
        or step.proposal.target_expression != proposal_expression
        or step.target_grounding is None
        or step.target_visual_descriptor is None
    ):
        return None
    expected_state = evidence.states[target.expected_state_id]
    screenshot = observation.redacted_screenshot
    if (
        (screenshot.width_px, screenshot.height_px)
        != (expected_state.width_px, expected_state.height_px)
        or (evidence.run.device is not None and observation.device != evidence.run.device)
    ):
        return None
    authorization = ProcedureGroundingAuthorization(
        source_step_id=step.source_step_id,
        procedure_index=step.procedure_index,
        expected_state_id=target.expected_state_id,
        observation_id=observation.observation_id,
        screenshot_artifact_id=screenshot.artifact_id,
        screenshot_sha256=screenshot.sha256,
        source_fingerprint=observation.screen_fingerprint,
        source_orientation=observation.orientation,
        source_keyboard_visible=observation.keyboard_visible,
        width_px=screenshot.width_px,
        height_px=screenshot.height_px,
        expected_phash_distance=None,
        competing_phash_distance=None,
        source_visual_match_ratio=None,
        requires_target_visual_match=True,
    )
    return ProcedureGroundingMatch(
        source_step_id=step.source_step_id,
        evidence=step.target_grounding,
        authorization=authorization,
        target_visual_descriptor=step.target_visual_descriptor,
    )


def build_keyboard_reflow_step(
    required_step: ReplayProcedureStep | None,
    reset_step: ReplayProcedureStep | None,
    procedure: ReplayProcedure,
    evidence: EvidenceDocument,
) -> ReplayProcedureStep | None:
    """Move a required field by a same-control keyboard reflow vector."""
    if required_step is None or reset_step is None:
        return None
    required_point = _target_point(required_step)
    reset_hidden_point = _target_point(reset_step)
    evidence_steps = {step.source_step_id: step for step in evidence.steps}
    reset_evidence = evidence_steps.get(reset_step.source_step_id)
    if (
        required_point is None
        or reset_hidden_point is None
        or reset_evidence is None
        or reset_evidence.control.identity_key is None
    ):
        return None
    reset_visible_step = next(
        (
            step
            for step in procedure.steps
            if step.source_step_id in evidence_steps
            and evidence_steps[step.source_step_id].control.identity_key
            == reset_evidence.control.identity_key
            and evidence.states[step.source_state_id].keyboard_visible is True
            and _target_point(step) is not None
        ),
        None,
    )
    reset_visible_point = _target_point(reset_visible_step)
    grounding = required_step.target_grounding
    if reset_visible_point is None or grounding is None:
        return None
    screenshot = grounding.request.screenshot
    translated_x = (
        required_point.pixel.x + reset_hidden_point.pixel.x - reset_visible_point.pixel.x
    )
    translated_y = (
        required_point.pixel.y + reset_hidden_point.pixel.y - reset_visible_point.pixel.y
    )
    if not (0 <= translated_x < screenshot.width_px and 0 <= translated_y < screenshot.height_px):
        return None
    translated = PixelPoint(x=translated_x, y=translated_y)
    location = GroundedPoint(
        normalized=NormalizedPoint(
            x=translated.x / max(1, screenshot.width_px - 1),
            y=translated.y / max(1, screenshot.height_px - 1),
        ),
        pixel=translated,
    )
    translated_grounding = grounding.model_copy(
        update={"result": grounding.result.model_copy(update={"location": location})}
    )
    return required_step.model_copy(update={"target_grounding": translated_grounding})


def _target_point(step: ReplayProcedureStep | None) -> GroundedPoint | None:
    if step is None or step.target_grounding is None:
        return None
    location = step.target_grounding.result.location
    return location if isinstance(location, GroundedPoint) else None


def _reuse_rejection(
    match: ProcedureGroundingMatch | None,
    request: GroundingRequest,
) -> ProcedureGroundingDecisionCode | None:
    if match is None:
        return ProcedureGroundingDecisionCode.NO_COMPILED_GROUNDING
    if request.correction is not None:
        return ProcedureGroundingDecisionCode.CORRECTION_REQUESTED
    stored_request = match.evidence.request
    stored_result = match.evidence.result
    current_fingerprint = request.source_fingerprint
    authorization = match.authorization
    if stored_result.disposition is not GroundingDisposition.ACCEPTED or stored_result.location is None:
        return ProcedureGroundingDecisionCode.STORED_GROUNDING_INVALID
    if (
        stored_request.expression != request.expression
        or stored_request.operation is not request.operation
        or stored_request.output_shape is not request.output_shape
        or stored_result.location.kind != request.output_shape.value
    ):
        return ProcedureGroundingDecisionCode.TARGET_CONTRACT_CHANGED
    if (
        current_fingerprint is None
        or request.screenshot.artifact_id != authorization.screenshot_artifact_id
        or request.screenshot.sha256 != authorization.screenshot_sha256
        or current_fingerprint != authorization.source_fingerprint
        or authorization.source_orientation is not request.source_orientation
        or authorization.source_keyboard_visible is not request.source_keyboard_visible
    ):
        return ProcedureGroundingDecisionCode.AUTHORIZED_OBSERVATION_CHANGED
    if (
        (
            request.screenshot.width_px,
            request.screenshot.height_px,
        )
        != (authorization.width_px, authorization.height_px)
        or (
            stored_request.screenshot.width_px,
            stored_request.screenshot.height_px,
        )
        != (authorization.width_px, authorization.height_px)
    ):
        return ProcedureGroundingDecisionCode.GEOMETRY_CHANGED
    if _location_is_excluded(stored_result.location, request):
        return ProcedureGroundingDecisionCode.LOCATION_EXCLUDED
    return None


def _location_is_excluded(
    location: GroundedLocation,
    request: GroundingRequest,
) -> bool:
    regions = request.excluded_regions or ()
    if isinstance(location, GroundedPoint):
        point = location.pixel
        return any(
            region.left <= point.x < region.right and region.top <= point.y < region.bottom
            for region in regions
        )
    if isinstance(location, GroundedRegion):
        grounded = location.pixel
        return any(
            grounded.left < region.right
            and grounded.right > region.left
            and grounded.top < region.bottom
            and grounded.bottom > region.top
            for region in regions
        )
    return True


def _narrate_lookup(request: GroundingRequest, *, reused: bool) -> None:
    narrator = current_narrator()
    if narrator is None:
        return
    message = (
        "Reusing exact grounding from the compiled procedure."
        if reused
        else "Procedure grounding did not match exact pixels. Using the fallback grounder."
    )
    narrator.record("procedure-grounding", message)
    narrator.trace(
        "procedure-grounding",
        f"request={request.grounding_id} reused={str(reused).lower()}",
    )
