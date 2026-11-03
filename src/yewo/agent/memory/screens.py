"""Logical-screen matching and retained observation variants."""

from __future__ import annotations

from collections.abc import Mapping

from yewo.artifacts import RecordIdentifier
from yewo.grounding import GroundingProfile
from yewo.observations import Observation, fingerprint_distance
from yewo.run_narration import current_narrator

from yewo.agent.memory.records import (
    AcceptedTransition,
    AppMemory,
    LogicalScreen,
    MemoryCompatibility,
    ObservationVariant,
    ScreenComparisonPolicy,
    UnsuccessfulAttempt,
    derived_memory_identifier,
)
from yewo.agent.actuation import ActionKind, ActionProposal
from yewo.agent.narration import narrate_memory_eviction
from yewo.agent.memory.references import referenced_memory_observation_ids
from yewo.agent.planning.planner import ActionProposalSource, PlanningRequest, TaskProgressContext
from yewo.agent.planning.evidence import (
    PlannerDecisionMetadata,
    resolve_planner_decision_metadata,
)


MEMORY_FACT_LIMIT = 20
ExactObservationStateKey = tuple[
    str | None,
    str | None,
    bool | None,
    str | None,
    int,
    int,
    str,
    str,
    str | None,
]


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
        """Delegate with bounded historical hints for the matching screen."""
        progress = request.progress_context or TaskProgressContext(
            remaining_work=request.open_work
        )
        memory_facts = memory_planning_facts(
            self._memory,
            request.observation,
            self._grounding_profile,
        )
        _narrate_planner_memory(request.observation, len(memory_facts))
        historical_facts = tuple(
            dict.fromkeys((*progress.historical_memory_facts, *memory_facts))
        )[:20]
        return self._source.plan(
            request.model_copy(
                update={
                    "progress_context": progress.model_copy(
                        update={"historical_memory_facts": historical_facts}
                    )
                }
            )
        )

    def required_model_requests(self, request: PlanningRequest) -> int:
        """Delegate model-request accounting without adding a call."""
        return self._source.required_model_requests(request)

    def decision_metadata(self) -> PlannerDecisionMetadata:
        """Preserve the evidence provenance selected by the delegated source."""
        return resolve_planner_decision_metadata(self._source)


def _narrate_planner_memory(observation: Observation, fact_count: int) -> None:
    narrator = current_narrator()
    if narrator is None:
        return
    if fact_count:
        narrator.record(
            "planner-memory",
            f"Reusing {fact_count} app-memory fact(s) for the planner.",
        )
        disposition = "hit"
        reason = "compatible_screen_evidence"
    else:
        narrator.record(
            "planner-memory",
            "No compatible app-memory facts were found for the planner.",
        )
        disposition = "miss"
        reason = "no_compatible_screen_evidence"
    narrator.trace(
        "planner-memory",
        f"observation={observation.observation_id} lookup={disposition} "
        f"reason={reason} facts={fact_count}",
    )


def memory_planning_facts(
    memory: AppMemory,
    observation: Observation,
    grounding_profile: GroundingProfile,
) -> tuple[str, ...]:
    """Describe prior attempts and proven expressions for the current screen."""
    screen_ids = {
        screen.screen_id
        for screen in memory.screens
        if any(
            observation_matches_replay_variant(
                observation,
                variant,
                memory.compatibility.screen_comparison,
            )
            for variant in screen.variants
        )
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
    input_facts = tuple(
        _input_contract_fact(transition)
        for transition in memory.transitions
        if transition.source_screen_id in screen_ids
        and transition.action.input_contract is not None
    )
    return tuple(
        dict.fromkeys(
            (*unsuccessful_facts[-MEMORY_FACT_LIMIT:], *proven_expression_facts, *input_facts)
        )
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
        f'Previously grounded target on this screen: {action.value} target "{target_description}". '
        "This proves only that the grounder located the visible target. It does not prove that "
        "the action is useful, reaches the expected destination, or advances the current goal."
    )


def _input_contract_fact(transition: AcceptedTransition) -> str:
    action = transition.action
    contract = action.input_contract
    if contract is None or action.target_description is None:
        raise ValueError("A reusable input fact requires a targeted input contract.")
    format_detail = (
        f" with format {contract.format_pattern}" if contract.format_pattern is not None else ""
    )
    return (
        f'Reusable input contract on this screen: "{action.target_description}" accepts '
        f"{contract.value_shape.value}{format_detail} input. Use targeted type directly. "
        "Do not focus this field first."
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


def observation_matches_exact_variant(
    observation: Observation,
    variant: ObservationVariant,
) -> bool:
    """Require identical application pixels and observation geometry."""
    return exact_observation_state_key(observation) == exact_variant_state_key(variant)


def observation_matches_replay_variant(
    observation: Observation,
    variant: ObservationVariant,
    policy: ScreenComparisonPolicy,
) -> bool:
    """Match stable replay state while tolerating small transient pixels."""
    screenshot = observation.redacted_screenshot
    if (
        observation.foreground_application_id != variant.foreground_application_id
        or observation.platform_context != variant.platform_context
        or observation.keyboard_visible is not variant.keyboard_visible
        or observation.orientation is not variant.orientation
        or (screenshot.width_px, screenshot.height_px) != (variant.width_px, variant.height_px)
    ):
        return False
    try:
        return (
            fingerprint_distance(observation.screen_fingerprint, variant.fingerprint)
            <= policy.maximum_replay_phash_distance
        )
    except ValueError:
        return False


def exact_observation_state_key(observation: Observation) -> ExactObservationStateKey:
    """Build the cross-run identity of one exact current application state."""
    screenshot = observation.redacted_screenshot
    fingerprint = observation.screen_fingerprint
    return (
        observation.foreground_application_id,
        observation.platform_context,
        observation.keyboard_visible,
        observation.orientation,
        screenshot.width_px,
        screenshot.height_px,
        fingerprint.algorithm,
        fingerprint.masked_pixels_sha256,
        fingerprint.system_region_mask_sha256,
    )


def exact_variant_state_key(variant: ObservationVariant) -> ExactObservationStateKey:
    """Build the cross-run identity of one exact retained application state."""
    return (
        variant.foreground_application_id,
        variant.platform_context,
        variant.keyboard_visible,
        variant.orientation,
        variant.width_px,
        variant.height_px,
        variant.fingerprint.algorithm,
        variant.fingerprint.masked_pixels_sha256,
        variant.fingerprint.system_region_mask_sha256,
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
            variants = (*screen.variants, _observation_variant(observation, source_run_id))
            screen = screen.model_copy(update={"variants": variants})
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


def retain_observation_variants(
    screens: list[LogicalScreen],
    transitions: tuple[AcceptedTransition, ...],
    unsuccessful_attempts: tuple[UnsuccessfulAttempt, ...],
    maximum_per_screen: int,
) -> tuple[LogicalScreen, ...]:
    """Retain referenced variants and the newest optional variants."""
    referenced_observation_ids = referenced_memory_observation_ids(
        transitions,
        unsuccessful_attempts,
    )
    retained_screens: list[LogicalScreen] = []
    evicted_observation_ids: list[RecordIdentifier] = []
    for screen in screens:
        required = [
            variant
            for variant in screen.variants
            if variant.observation_id in referenced_observation_ids
        ]
        optional = [
            variant
            for variant in screen.variants
            if variant.observation_id not in referenced_observation_ids
        ]
        retained_optional = optional[-maximum_per_screen:]
        selected_ids = {variant.observation_id for variant in (*required, *retained_optional)}
        evicted_observation_ids.extend(
            variant.observation_id
            for variant in optional
            if variant.observation_id not in selected_ids
        )
        retained_screens.append(
            screen.model_copy(
                update={
                    "variants": tuple(
                        variant
                        for variant in screen.variants
                        if variant.observation_id in selected_ids
                    )
                }
            )
        )
    narrate_memory_eviction("optional_variant", tuple(evicted_observation_ids))
    return tuple(retained_screens)


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
