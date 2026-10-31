"""Logical-screen matching and retained observation variants."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from yewo.artifacts import RecordIdentifier
from yewo.grounding import GroundingProfile
from yewo.observations import Observation, fingerprint_distance

from yewo.agent.app_memory import (
    AppMemory,
    LogicalScreen,
    MemoryCompatibility,
    ObservationVariant,
    ScreenComparisonPolicy,
    UnsuccessfulAttempt,
    derived_memory_identifier,
)
from yewo.agent.actuation import ActionKind, ActionProposal
from yewo.agent.planner import ActionProposalSource, PlanningRequest


MEMORY_FACT_LIMIT = 20


class AppMemoryActionSource:
    """Add current-screen memory evidence before requesting the next action."""

    def __init__(
        self,
        source: ActionProposalSource,
        memory: AppMemory,
        grounding_profile: GroundingProfile,
    ) -> None:
        self._source = source
        self._memory = memory
        self._grounding_profile = grounding_profile

    def plan(self, request: PlanningRequest) -> ActionProposal:
        """Delegate with bounded facts from matching compatible screen evidence."""
        memory_facts = memory_planning_facts(
            self._memory,
            request.observation,
            self._grounding_profile,
        )
        known_facts = tuple(dict.fromkeys((*request.known_facts, *memory_facts)))[:100]
        return self._source.plan(request.model_copy(update={"known_facts": known_facts}))


def memory_planning_facts(
    memory: AppMemory,
    observation: Observation,
    grounding_profile: GroundingProfile,
) -> tuple[str, ...]:
    """Describe prior attempts and proven expressions for the current screen."""
    screen_ids = {
        screen.screen_id
        for screen in memory.screens
        if observation_matches_screen(observation, screen, memory.compatibility.screen_comparison)
    }
    unsuccessful_facts = tuple(
        _unsuccessful_attempt_fact(attempt)
        for attempt in memory.unsuccessful_attempts
        if attempt.screen_id in screen_ids
        and (
            attempt.grounding_profile_id is None
            or (
                attempt.grounding_profile_id == grounding_profile.profile_id
                and attempt.grounder_model_profile_id == grounding_profile.model_profile_id
            )
        )
    )
    proven_expression_facts = tuple(
        _proven_expression_fact(transition.action.action, transition.action.target_description)
        for transition in memory.transitions
        if transition.source_screen_id in screen_ids
        and transition.action.target_description is not None
        and any(
            profile_id == grounding_profile.profile_id
            and model_profile_id == grounding_profile.model_profile_id
            for profile_id, model_profile_id in zip(
                transition.grounding_profile_ids,
                transition.grounder_model_profile_ids,
                strict=True,
            )
        )
    )
    return tuple(
        dict.fromkeys((*unsuccessful_facts[-MEMORY_FACT_LIMIT:], *proven_expression_facts))
    )[:MEMORY_FACT_LIMIT]


def _unsuccessful_attempt_fact(attempt: UnsuccessfulAttempt) -> str:
    target = (
        f' target "{attempt.target_description}"' if attempt.target_description is not None else ""
    )
    point = (
        f" at ({attempt.grounded_point.x}, {attempt.grounded_point.y})"
        if attempt.grounded_point is not None
        else ""
    )
    return (
        f"Previous unsuccessful attempt on this screen: {attempt.action.value}{target}{point} "
        f"ended as {attempt.disposition.value}: {attempt.reason} "
        "This is attempt history, not proof that the application lacks the target or behavior."
    )


def _proven_expression_fact(action: ActionKind, target_description: str) -> str:
    return (
        f'Grounder-proven expression on this screen: {action.value} target "{target_description}" '
        "produced an accepted outcome. Reuse it only when that target is visible."
    )


def observation_matches_screen(
    observation: Observation,
    screen: LogicalScreen,
    policy: ScreenComparisonPolicy,
) -> bool:
    """Return whether current evidence matches one stored logical screen."""
    return any(
        _observation_matches_variant(
            observation,
            variant,
            policy.maximum_phash_distance,
        )
        for variant in screen.variants
    )


def observations_match(
    first: Observation,
    second: Observation,
    policy: ScreenComparisonPolicy,
) -> bool:
    """Return whether two current observations represent one logical screen."""
    first_screenshot = first.redacted_screenshot
    second_screenshot = second.redacted_screenshot
    if (
        first.foreground_application_id != second.foreground_application_id
        or first.platform_context != second.platform_context
        or first.orientation is not second.orientation
        or (first_screenshot.width_px, first_screenshot.height_px)
        != (second_screenshot.width_px, second_screenshot.height_px)
    ):
        return False
    try:
        return (
            fingerprint_distance(first.screen_fingerprint, second.screen_fingerprint)
            <= policy.maximum_phash_distance
        )
    except ValueError:
        return False


def validate_memory_observations(
    observations: Mapping[str, Observation],
    compatibility: MemoryCompatibility,
) -> None:
    """Validate observations before they can contribute to app memory."""
    if any(key != value.observation_id for key, value in observations.items()):
        raise ValueError("Observation lookup keys must match their record identifiers.")
    if any(
        observation.application != compatibility.application
        for observation in observations.values()
    ):
        raise ValueError("Discovery observations do not match the memory application.")
    if any(
        observation.screen_fingerprint.algorithm != compatibility.fingerprint_algorithm
        for observation in observations.values()
    ):
        raise ValueError("Discovery observations use an incompatible fingerprint algorithm.")


def add_observation_screen(
    observation: Observation,
    screens: list[LogicalScreen],
    source_run_id: str,
    compatibility: MemoryCompatibility,
) -> LogicalScreen:
    """Merge one observation into its retained logical screen or create one."""
    for index, screen in enumerate(screens):
        if not observation_matches_screen(
            observation,
            screen,
            compatibility.screen_comparison,
        ):
            continue
        if observation.observation_id not in {
            variant.observation_id for variant in screen.variants
        }:
            maximum = compatibility.retention.maximum_variants_per_screen
            variants = (*screen.variants, _observation_variant(observation, source_run_id))
            screen = screen.model_copy(update={"variants": variants[-maximum:]})
            screens[index] = screen
        return screen
    screen = LogicalScreen(
        screen_id=derived_memory_identifier(
            "screen",
            observation.application.application_id,
            observation.screen_fingerprint.value,
            observation.screen_fingerprint.system_region_mask_sha256 or "unmasked",
            observation.platform_context or "",
            str(observation.redacted_screenshot.width_px),
            str(observation.redacted_screenshot.height_px),
        ),
        variants=(_observation_variant(observation, source_run_id),),
    )
    screens.append(screen)
    return screen


def merge_replay_variants(
    memory: AppMemory,
    observations: Mapping[str, Observation],
    *,
    source_run_id: RecordIdentifier,
    updated_at: datetime,
) -> AppMemory:
    """Learn fresh variants without changing reviewed route semantics."""
    validate_memory_observations(observations, memory.compatibility)
    screens = list(memory.screens)
    changed = False
    for observation in observations.values():
        before = tuple(screens)
        if not any(
            observation_matches_screen(
                observation,
                screen,
                memory.compatibility.screen_comparison,
            )
            for screen in screens
        ):
            continue
        add_observation_screen(
            observation,
            screens,
            source_run_id,
            memory.compatibility,
        )
        changed = changed or before != tuple(screens)
    if not changed:
        return memory
    return memory.model_copy(
        update={
            "screens": tuple(screens),
            "source_run_ids": tuple(dict.fromkeys((*memory.source_run_ids, source_run_id))),
            "updated_at": updated_at,
        }
    )


def _observation_matches_variant(
    observation: Observation,
    variant: ObservationVariant,
    maximum_distance: int,
) -> bool:
    screenshot = observation.redacted_screenshot
    if (
        observation.foreground_application_id != variant.foreground_application_id
        or observation.platform_context != variant.platform_context
        or observation.orientation is not variant.orientation
        or (screenshot.width_px, screenshot.height_px) != (variant.width_px, variant.height_px)
    ):
        return False
    try:
        return (
            fingerprint_distance(observation.screen_fingerprint, variant.fingerprint)
            <= maximum_distance
        )
    except ValueError:
        return False


def _observation_variant(
    observation: Observation,
    source_run_id: str,
) -> ObservationVariant:
    screenshot = observation.redacted_screenshot
    return ObservationVariant(
        observation_id=observation.observation_id,
        source_run_id=source_run_id,
        foreground_application_id=observation.foreground_application_id,
        platform_context=observation.platform_context,
        keyboard_visible=observation.keyboard_visible,
        orientation=observation.orientation,
        width_px=screenshot.width_px,
        height_px=screenshot.height_px,
        fingerprint=observation.screen_fingerprint,
        screenshot=screenshot,
    )
