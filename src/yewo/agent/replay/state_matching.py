"""Deterministic matching between live observations and compiled evidence states."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

from yewo.agent.replay.reconciliation.records import EvidenceDocument, EvidenceState
from yewo.agent.replay.visual_matching import (
    state_visual_match_is_exact,
    state_visual_match_ratio,
)
from yewo.observations import (
    Observation,
    ObservationCaptureError,
    ScreenshotReference,
    fingerprint_distance,
)
from yewo.run_narration import current_narrator


@dataclass(frozen=True)
class EvidenceStateMatch:
    """One explainable source-state comparison."""

    accepted: bool
    expected_distance: int | None
    competing_distance: int | None
    visual_match_ratio: float | None
    reason: str


AMBIGUOUS_SOURCE_STATE = "source_state_ambiguous"


def trace_repair_target_match(accepted: bool, reason: str) -> bool:
    """Record why one repair-target comparison accepted or rejected a screen."""
    narrator = current_narrator()
    if narrator is not None:
        narrator.trace(
            "procedure-repair",
            f"repair_target_match accepted={str(accepted).lower()} reason={reason}",
        )
    return accepted


def match_evidence_state(
    observation: Observation,
    expected_state_id: str,
    evidence: EvidenceDocument,
    screenshot_loader: Callable[[ScreenshotReference], bytes] | None = None,
) -> EvidenceStateMatch:
    """Match one live observation and reject an ambiguous known-state result."""
    expected = evidence.states[expected_state_id]
    if evidence.run.device is not None and observation.device != evidence.run.device:
        return EvidenceStateMatch(False, None, None, None, "source_device_changed")
    if not _metadata_matches(observation, expected):
        return EvidenceStateMatch(False, None, None, None, "source_metadata_changed")
    expected_distance = _minimum_distance(observation, expected)
    if expected_distance is None:
        return EvidenceStateMatch(False, None, None, None, "source_fingerprint_incompatible")
    if (
        expected.visual_signature is None
        and expected_distance > evidence.state_grouping_maximum_phash_distance
    ):
        return EvidenceStateMatch(
            False,
            expected_distance,
            None,
            None,
            "expected_state_too_distant",
        )
    competing_distance = _closest_competing_distance(
        observation,
        expected_state_id,
        evidence,
    )
    if (
        competing_distance is not None
        and competing_distance - expected_distance < evidence.state_matching_minimum_phash_margin
    ):
        return EvidenceStateMatch(
            False,
            expected_distance,
            competing_distance,
            None,
            AMBIGUOUS_SOURCE_STATE,
        )
    visual_match_ratio = _visual_match_ratio(observation, expected, screenshot_loader)
    if expected.visual_signature is not None and (
        visual_match_ratio is None
        or visual_match_ratio < expected.visual_signature.minimum_match_ratio
    ):
        return EvidenceStateMatch(
            False,
            expected_distance,
            competing_distance,
            visual_match_ratio,
            "source_stable_regions_changed",
        )
    return EvidenceStateMatch(
        True,
        expected_distance,
        competing_distance,
        visual_match_ratio,
        "expected_state_matched",
    )


def state_matches_observation(
    observation: Observation,
    state: EvidenceState,
    maximum_phash_distance: int,
    screenshot_loader: Callable[[ScreenshotReference], bytes] | None = None,
) -> bool:
    """Return whether one observation matches any recorded state variant."""
    if not _metadata_matches(observation, state):
        return False
    distance = _minimum_distance(observation, state)
    if distance is None:
        return False
    visual_match_ratio = _visual_match_ratio(observation, state, screenshot_loader)
    if state.visual_signature is None:
        return distance <= maximum_phash_distance
    return (
        visual_match_ratio is not None
        and visual_match_ratio >= state.visual_signature.minimum_match_ratio
    )


def repair_state_matches_observation(
    observation: Observation,
    state: EvidenceState,
    maximum_phash_distance: int,
    screenshot_loader: Callable[[ScreenshotReference], bytes] | None = None,
) -> bool:
    """Accept repair rejoin only when all compiler-stable content remains intact."""
    if not _metadata_matches(observation, state):
        return False
    distance = _minimum_distance(observation, state)
    if distance is None:
        return False
    if state.visual_signature is None:
        return distance <= maximum_phash_distance
    if screenshot_loader is None:
        return False
    try:
        png = screenshot_loader(observation.redacted_screenshot)
    except ObservationCaptureError:
        return False
    return state_visual_match_is_exact(png, state.visual_signature)


def _visual_match_ratio(
    observation: Observation,
    state: EvidenceState,
    screenshot_loader: Callable[[ScreenshotReference], bytes] | None,
) -> float | None:
    if state.visual_signature is None:
        return None
    if screenshot_loader is None:
        return None
    try:
        png = screenshot_loader(observation.redacted_screenshot)
    except ObservationCaptureError:
        return None
    return state_visual_match_ratio(png, state.visual_signature)


def _closest_competing_distance(
    observation: Observation,
    expected_state_id: str,
    evidence: EvidenceDocument,
) -> int | None:
    distances = tuple(
        distance
        for state_id, state in evidence.states.items()
        if state_id != expected_state_id and _metadata_matches(observation, state)
        for distance in (_minimum_distance(observation, state),)
        if distance is not None
    )
    return min(distances, default=None)


def _minimum_distance(
    observation: Observation,
    state: EvidenceState,
) -> int | None:
    distances: list[int] = []
    for fingerprint in state.comparison_fingerprints():
        try:
            distances.append(fingerprint_distance(observation.screen_fingerprint, fingerprint))
        except ValueError:
            continue
    return min(distances, default=None)


def _metadata_matches(observation: Observation, state: EvidenceState) -> bool:
    screenshot = observation.redacted_screenshot
    return (
        observation.foreground_application_id == state.foreground_application_id
        and observation.platform_context == state.platform_context
        and observation.keyboard_visible is state.keyboard_visible
        and observation.orientation is state.orientation
        and (screenshot.width_px, screenshot.height_px) == (state.width_px, state.height_px)
    )
