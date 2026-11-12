"""Exact-screen grounding reuse from accepted app-memory transitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from yewo.agent.actuation import ACTION_DEFINITIONS
from yewo.artifacts import RecordIdentifier, Sha256Digest
from yewo.grounding import (
    GroundedLocation,
    GroundedPoint,
    GroundedRegion,
    GroundingDisposition,
    GroundingProfile,
    GroundingRequest,
    GroundingResolution,
    GroundingResult,
    MemoryGroundingResolution,
    VisualGrounder,
    validate_grounding_request,
)
from yewo.run_narration import current_narrator
from yewo.agent.memory.knowledge import (
    GroundingAssessment,
    GroundingAssessmentStatus,
    GroundingRole,
)

if TYPE_CHECKING:
    from yewo.agent.memory.records import (
        AcceptedTransition,
        AppMemory,
        LogicalScreen,
        ObservationVariant,
    )


class GroundingMemoryDisposition(StrEnum):
    """The result of one Yewo grounding-memory lookup."""

    HIT = "hit"
    MISS = "miss"
    INVALIDATED = "invalidated"
    BYPASSED = "bypassed"


@dataclass(frozen=True)
class GroundingMemoryLookup:
    """One grounding-memory lookup and its reusable evidence, when present."""

    disposition: GroundingMemoryDisposition
    reason: str
    match: tuple[AcceptedTransition, AcceptedGrounding] | None = None


class AcceptedGrounding(BaseModel):
    """One accepted location tied to exact application-owned source pixels."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    location: GroundedLocation
    source_observation_id: RecordIdentifier
    source_masked_pixels_sha256: Sha256Digest
    source_system_region_mask_sha256: Sha256Digest | None = None
    resolution: GroundingResolution


class AppMemoryGrounder:
    """Reuse exact accepted locations and delegate every non-exact request."""

    def __init__(
        self,
        delegate: VisualGrounder,
        memory: AppMemory,
        profile: GroundingProfile,
    ) -> None:
        self._delegate = delegate
        self._memory = memory
        self._profile = profile
        self._variants_by_id = {
            variant.observation_id: variant
            for screen in memory.screens
            for variant in screen.variants
        }
        self._assessments_by_key = {
            (assessment.transition_id, assessment.role, assessment.source_observation_id): (
                assessment
            )
            for assessment in memory.app_knowledge.grounding_assessments
        }

    def required_model_requests(self, request: GroundingRequest) -> int:
        """Return zero only when exact memory can resolve the request."""
        validate_grounding_request(self._profile, request)
        lookup = self._lookup(request)
        return 0 if lookup.match is not None else self._delegate.required_model_requests(request)

    def ground(self, request: GroundingRequest) -> GroundingResult:
        """Resolve from exact memory or use the configured visual grounder."""
        validate_grounding_request(self._profile, request)
        lookup = self._lookup(request)
        _narrate_grounding_lookup(request, lookup)
        if lookup.match is None:
            return self._delegate.ground(request)
        transition, grounding = lookup.match
        return GroundingResult(
            grounding_id=request.grounding_id,
            source_screenshot_sha256=request.screenshot.sha256,
            output_shape=request.output_shape,
            disposition=GroundingDisposition.ACCEPTED,
            resolution=MemoryGroundingResolution(transition_id=transition.transition_id),
            location=grounding.location,
        )

    def _lookup(
        self,
        request: GroundingRequest,
    ) -> GroundingMemoryLookup:
        precondition = _lookup_precondition(request)
        if precondition is not None:
            return precondition
        matches, invalidation_reasons, found_target_evidence = self._candidate_groundings(request)
        return _resolve_grounding_lookup(matches, invalidation_reasons, found_target_evidence)

    def _candidate_groundings(
        self,
        request: GroundingRequest,
    ) -> tuple[
        list[tuple[AcceptedTransition, AcceptedGrounding]],
        set[str],
        bool,
    ]:
        matches: list[tuple[AcceptedTransition, AcceptedGrounding]] = []
        invalidation_reasons: set[str] = set()
        found_target_evidence = False
        for transition in reversed(self._memory.transitions):
            definition = ACTION_DEFINITIONS[transition.action.action]
            candidates = (
                (
                    "target",
                    transition.action.target_description,
                    definition.target_operation,
                    transition.target_grounding,
                ),
                (
                    "destination",
                    transition.action.destination_description,
                    definition.destination_operation,
                    transition.destination_grounding,
                ),
            )
            for role, expression, operation, grounding in candidates:
                if (
                    grounding is None
                    or expression != request.expression
                    or operation is not request.operation
                ):
                    continue
                found_target_evidence = True
                source_observation_id = grounding.source_observation_id
                source_variant = self._variants_by_id[source_observation_id]
                assessment = self._assessments_by_key.get(
                    (transition.transition_id, GroundingRole(role), source_observation_id)
                )
                reason = _grounding_invalidation_reason(
                    grounding,
                    source_variant,
                    request,
                    self._profile,
                    assessment,
                )
                if reason is None:
                    matches.append((transition, grounding))
                else:
                    invalidation_reasons.add(reason)
        return matches, invalidation_reasons, found_target_evidence


def _lookup_precondition(request: GroundingRequest) -> GroundingMemoryLookup | None:
    if request.source_presentation is None:
        return GroundingMemoryLookup(
            GroundingMemoryDisposition.MISS,
            "source_fingerprint_unavailable",
        )
    if request.correction is not None:
        return GroundingMemoryLookup(
            GroundingMemoryDisposition.BYPASSED,
            "correction_requires_fresh_grounding",
        )
    return None


def _resolve_grounding_lookup(
    matches: list[tuple[AcceptedTransition, AcceptedGrounding]],
    invalidation_reasons: set[str],
    found_target_evidence: bool,
) -> GroundingMemoryLookup:
    if not matches:
        if not found_target_evidence:
            return GroundingMemoryLookup(
                GroundingMemoryDisposition.MISS,
                "no_compatible_target_evidence",
            )
        return GroundingMemoryLookup(
            GroundingMemoryDisposition.INVALIDATED,
            _primary_invalidation_reason(invalidation_reasons),
        )
    locations = {item[1].location.model_dump_json() for item in matches}
    if len(locations) != 1:
        return GroundingMemoryLookup(
            GroundingMemoryDisposition.INVALIDATED,
            "conflicting_cached_locations",
        )
    return GroundingMemoryLookup(
        GroundingMemoryDisposition.HIT,
        "exact_source_match",
        matches[0],
    )


def _grounding_invalidation_reason(
    grounding: AcceptedGrounding,
    source_variant: ObservationVariant,
    request: GroundingRequest,
    profile: GroundingProfile,
    assessment: GroundingAssessment | None,
) -> str | None:
    contract_reason = _grounding_contract_invalidation_reason(
        grounding,
        request,
        profile,
        assessment,
    )
    if contract_reason is not None:
        return contract_reason
    return _grounding_source_invalidation_reason(grounding, source_variant, request)


def _grounding_contract_invalidation_reason(
    grounding: AcceptedGrounding,
    request: GroundingRequest,
    profile: GroundingProfile,
    assessment: GroundingAssessment | None,
) -> str | None:
    if assessment is not None and assessment.status is GroundingAssessmentStatus.INVALIDATED:
        return "writeback_invalidated"
    del profile
    if grounding.location.kind != request.output_shape.value:
        return "output_shape_changed"
    return None


def _grounding_source_invalidation_reason(
    grounding: AcceptedGrounding,
    source_variant: ObservationVariant,
    request: GroundingRequest,
) -> str | None:
    presentation = request.source_presentation
    if presentation is None:
        return "source_fingerprint_unavailable"
    source_presentation = source_variant.presentation
    if (source_presentation.width_px, source_presentation.height_px) != (
        request.screenshot.width_px,
        request.screenshot.height_px,
    ):
        return "screenshot_dimensions_changed"
    if source_presentation.orientation is not presentation.orientation:
        return "orientation_changed"
    if source_presentation.keyboard_visible is not presentation.keyboard_visible:
        return "keyboard_state_changed"
    if grounding.source_masked_pixels_sha256 != presentation.fingerprint.masked_pixels_sha256:
        return "source_pixels_changed"
    if (
        grounding.source_system_region_mask_sha256
        != presentation.fingerprint.system_region_mask_sha256
    ):
        return "system_region_mask_changed"
    if _location_is_excluded(grounding.location, request):
        return "cached_location_excluded"
    return None


_INVALIDATION_REASON_PRIORITY = (
    "writeback_invalidated",
    "cached_location_excluded",
    "system_region_mask_changed",
    "source_pixels_changed",
    "keyboard_state_changed",
    "orientation_changed",
    "screenshot_dimensions_changed",
    "output_shape_changed",
)


def _primary_invalidation_reason(reasons: set[str]) -> str:
    return next(
        (reason for reason in _INVALIDATION_REASON_PRIORITY if reason in reasons), "unknown"
    )


def _narrate_grounding_lookup(
    request: GroundingRequest,
    lookup: GroundingMemoryLookup,
) -> None:
    narrator = current_narrator()
    if narrator is None:
        return
    transition_id = lookup.match[0].transition_id if lookup.match is not None else "none"
    if lookup.disposition is GroundingMemoryDisposition.HIT:
        message = "Reusing an exact grounding location from app memory."
    elif lookup.disposition is GroundingMemoryDisposition.INVALIDATED:
        message = "Grounding memory was invalidated. Requesting fresh grounding."
    elif lookup.disposition is GroundingMemoryDisposition.BYPASSED:
        message = "Grounding memory was bypassed. Requesting corrective grounding."
    else:
        message = "No reusable grounding location was found. Requesting fresh grounding."
    narrator.record("grounding-memory", message)
    narrator.trace(
        "grounding-memory",
        f"request={request.grounding_id} lookup={lookup.disposition.value} "
        f"reason={lookup.reason} transition={transition_id}",
    )


def _location_is_excluded(location: GroundedLocation, request: GroundingRequest) -> bool:
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


def validate_transition_groundings(
    screens: tuple[LogicalScreen, ...],
    transitions: tuple[AcceptedTransition, ...],
) -> None:
    """Require every cached location to identify its exact source pixels."""
    variants_by_screen_id = {
        screen.screen_id: {variant.observation_id: variant for variant in screen.variants}
        for screen in screens
    }
    for transition in transitions:
        source_variants = variants_by_screen_id.get(transition.source_screen_id, {})
        destination_variants = variants_by_screen_id.get(transition.destination_screen_id, {})
        if transition.source_observation_id not in source_variants:
            raise ValueError("A transition must reference its source observation.")
        if transition.destination_observation_id not in destination_variants:
            raise ValueError("A transition must reference its destination observation.")
        for grounding in (transition.target_grounding, transition.destination_grounding):
            if grounding is None:
                continue
            grounding_observation_id = grounding.source_observation_id
            grounding_variant = source_variants.get(grounding_observation_id)
            if grounding_variant is None:
                raise ValueError("Cached grounding must reference a source-screen observation.")
            if (
                grounding.source_masked_pixels_sha256
                != grounding_variant.presentation.fingerprint.masked_pixels_sha256
                or grounding.source_system_region_mask_sha256
                != grounding_variant.presentation.fingerprint.system_region_mask_sha256
            ):
                raise ValueError("Cached grounding must match its exact source observation.")
