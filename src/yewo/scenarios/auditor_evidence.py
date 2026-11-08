"""Current-run evidence selection for independent scenario audit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from yewo.agent.planning.goal_evidence import (
    AdaptiveGoalEvidencePacker,
    GoalEvidenceImage,
    GoalEvidenceSource,
)
from yewo.agent.runtime.records import AgentResult, AgentStep
from yewo.observations import Observation, ObservationCaptureError, ObservationStore
from yewo.models import ModelImageConstraints, ModelInputError
from yewo.scenarios.coordinator import ScenarioCoordinationResult
from yewo.scenarios.records import (
    DiscoveredTestSurface,
    ScenarioCoverageObligation,
    ScenarioExpectedOutcome,
    ScenarioSuite,
)
from yewo.scenarios.taxonomy import RequiredEvidenceOwner


class ScenarioEvidence(BaseModel):
    """Bounded current-run evidence inventory for one scenario audit."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    images: tuple[GoalEvidenceSource, ...]
    owner_counts: dict[RequiredEvidenceOwner, int]
    action_facts: tuple[str, ...]
    capability_facts: tuple[str, ...]
    visual_episodes: tuple["VisualEvidenceEpisode", ...] = ()
    required_surface_ids: tuple[str, ...] = ()
    covered_surface_ids: tuple[str, ...] = ()
    missing_surface_ids: tuple[str, ...] = ()


class VisualEvidenceEpisode(BaseModel):
    """One chronological application-screen episode for visual audit."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    episode_id: str
    label: str
    images: tuple[GoalEvidenceSource, ...]
    action_facts: tuple[str, ...] = ()


@dataclass(frozen=True)
class _VisualFrame:
    """One current-run observation with its execution provenance."""

    observation_id: str
    step: AgentStep | None
    phase: str
    episode_key: str
    episode_label: str


def pack_readable_evidence_batches(
    root: Path,
    constraints: ModelImageConstraints,
    sources: tuple[GoalEvidenceSource, ...],
) -> tuple[tuple[GoalEvidenceImage, ...], ...]:
    """Pack every source into the fewest readable model requests."""
    if not sources:
        return ((),)
    packer = AdaptiveGoalEvidencePacker(root, constraints)
    batches: list[tuple[GoalEvidenceImage, ...]] = []
    start = 0
    while start < len(sources):
        end = len(sources)
        while end > start:
            candidate = sources[start:end]
            if not packer.fits(candidate):
                end -= 1
                continue
            packed = packer.pack(candidate)
            batches.append(packed)
            start = end
            break
        else:
            raise ModelInputError(
                "A scenario audit image cannot fit the configured image capacity."
            )
    return tuple(batches)


def collect_scenario_evidence(
    child_root: Path,
    suite: ScenarioSuite,
    obligation: ScenarioCoverageObligation,
    result: ScenarioCoordinationResult,
) -> ScenarioEvidence:
    """Select one current-run observation for every required test surface."""
    return _collect_evidence(
        child_root,
        suite,
        obligation.required_surface_ids,
        result,
    )


def collect_outcome_evidence(
    child_root: Path,
    suite: ScenarioSuite,
    outcome: ScenarioExpectedOutcome,
    result: ScenarioCoordinationResult,
    *,
    include_following_observation: bool = False,
) -> ScenarioEvidence:
    """Select current-run evidence for one expected outcome."""
    return _collect_evidence(
        child_root,
        suite,
        outcome.required_surface_ids,
        result,
        include_following_observation=include_following_observation,
    )


def collect_visual_outcome_evidence(
    child_root: Path,
    suite: ScenarioSuite,
    outcome: ScenarioExpectedOutcome,
    result: ScenarioCoordinationResult,
) -> ScenarioEvidence:
    """Collect the complete current-run observation sequence for UI audit."""
    agents = _result_agents(result)
    surfaces = tuple(
        surface for surface in suite.surfaces if surface.surface_id in outcome.required_surface_ids
    )
    frames, covered_surface_ids = _visual_frames(agents, surfaces)
    observation_ids = {frame.observation_id for frame in frames}
    try:
        observations = ObservationStore(child_root).load_selected(observation_ids)
    except ObservationCaptureError:
        observations = ()
    observation_by_id = {item.observation_id: item for item in observations}
    episodes = _build_visual_episodes(frames, observation_by_id)
    images = tuple(image for episode in episodes for image in episode.images)
    steps = tuple(step for agent in agents for step in agent.steps)
    required_surface_ids = tuple(surface.surface_id for surface in surfaces)
    covered = tuple(
        surface_id for surface_id in required_surface_ids if surface_id in covered_surface_ids
    )
    missing_surface_ids = tuple(
        surface_id for surface_id in required_surface_ids if surface_id not in covered_surface_ids
    )
    return ScenarioEvidence(
        images=images,
        owner_counts={
            RequiredEvidenceOwner.APPLICATION_OBSERVATION: len(images),
            RequiredEvidenceOwner.ACTION_OUTCOME: sum(step.outcome is not None for step in steps),
            RequiredEvidenceOwner.DEVICE_RECEIPT: len(result.capability_receipts),
            RequiredEvidenceOwner.APPLICATION_GROUND_TRUTH: 0,
            RequiredEvidenceOwner.MEASUREMENT: 0,
            RequiredEvidenceOwner.ACCESSIBILITY_INSPECTION: 0,
        },
        action_facts=tuple(_action_fact(step) for step in steps if step.outcome is not None),
        capability_facts=_capability_facts(result),
        visual_episodes=episodes,
        required_surface_ids=required_surface_ids,
        covered_surface_ids=covered,
        missing_surface_ids=missing_surface_ids,
    )


def _collect_evidence(
    child_root: Path,
    suite: ScenarioSuite,
    required_surface_ids: tuple[str, ...],
    result: ScenarioCoordinationResult,
    *,
    include_following_observation: bool = False,
) -> ScenarioEvidence:
    agents = _result_agents(result)
    steps = tuple(step for agent in agents for step in agent.steps)
    surfaces = tuple(
        surface for surface in suite.surfaces if surface.surface_id in required_surface_ids
    )
    relevant_indexes = {
        index for surface in surfaces for index in surface.provenance.procedure_step_indexes
    }
    relevant_steps = (
        tuple(step for step in steps if step.decision.reused_segment_index in relevant_indexes)
        or steps
    )
    observation_labels = _required_observation_labels(
        surfaces,
        agents,
        steps,
        include_following_observation=include_following_observation,
    )
    images = _load_evidence_images(child_root, observation_labels)
    action_facts = tuple(_action_fact(step) for step in relevant_steps if step.outcome is not None)
    capability_facts = _capability_facts(result)
    return ScenarioEvidence(
        images=images,
        owner_counts={
            RequiredEvidenceOwner.APPLICATION_OBSERVATION: len(images),
            RequiredEvidenceOwner.ACTION_OUTCOME: len(action_facts),
            RequiredEvidenceOwner.DEVICE_RECEIPT: len(result.capability_receipts),
            RequiredEvidenceOwner.APPLICATION_GROUND_TRUTH: 0,
            RequiredEvidenceOwner.MEASUREMENT: 0,
            RequiredEvidenceOwner.ACCESSIBILITY_INSPECTION: 0,
        },
        action_facts=action_facts,
        capability_facts=capability_facts,
    )


def _result_agents(result: ScenarioCoordinationResult) -> tuple[AgentResult, ...]:
    return tuple(
        agent
        for agent in (
            result.route.agent_result if result.route is not None else None,
            result.agent,
        )
        if agent is not None
    )


def _capability_facts(result: ScenarioCoordinationResult) -> tuple[str, ...]:
    return tuple(
        f"Capability {receipt.request.kind.value}: application verified="
        f"{receipt.application.verified}. Restoration verified="
        f"{receipt.restoration.verified}."
        for receipt in result.capability_receipts
    )


def _visual_frames(
    agents: tuple[AgentResult, ...],
    surfaces: tuple[DiscoveredTestSurface, ...],
) -> tuple[tuple[_VisualFrame, ...], frozenset[str]]:
    surface_ids_by_index: dict[int, tuple[str, ...]] = {}
    for surface in surfaces:
        for procedure_index in surface.provenance.procedure_step_indexes:
            existing_surface_ids = surface_ids_by_index.get(procedure_index, ())
            surface_ids_by_index[procedure_index] = tuple(
                dict.fromkeys((*existing_surface_ids, surface.surface_id))
            )
    frames: list[_VisualFrame] = []
    covered_surface_ids: set[str] = set()
    seen_observation_ids: set[str] = set()
    unmapped_episode_count = 0
    active_unmapped_key: str | None = None
    for agent_index, agent in enumerate(agents, start=1):
        for step in agent.steps:
            segment_index = step.decision.reused_segment_index
            surface_ids = (
                surface_ids_by_index.get(segment_index, ()) if segment_index is not None else ()
            )
            if surface_ids:
                covered_surface_ids.update(surface_ids)
                episode_key = "surface:" + ",".join(surface_ids)
                episode_label = "Required surfaces " + ", ".join(surface_ids)
                active_unmapped_key = None
            else:
                if active_unmapped_key is None:
                    unmapped_episode_count += 1
                    active_unmapped_key = f"unmapped:{agent_index}:{unmapped_episode_count}"
                episode_key = active_unmapped_key
                episode_label = "Chronological application states without segment ownership"
            candidates = [
                (step.source_observation_id, "source"),
                *((item, "settlement") for item in step.settlement_observation_ids),
            ]
            if step.recovery_observation_id is not None:
                candidates.append((step.recovery_observation_id, "recovery"))
            for observation_id, phase in candidates:
                if observation_id in seen_observation_ids:
                    continue
                seen_observation_ids.add(observation_id)
                frames.append(
                    _VisualFrame(
                        observation_id=observation_id,
                        step=step,
                        phase=phase,
                        episode_key=episode_key,
                        episode_label=episode_label,
                    )
                )
        final_id = agent.final_observation_id
        if final_id not in seen_observation_ids:
            if active_unmapped_key is None:
                unmapped_episode_count += 1
                active_unmapped_key = f"unmapped:{agent_index}:{unmapped_episode_count}"
            frames.append(
                _VisualFrame(
                    observation_id=final_id,
                    step=None,
                    phase="final",
                    episode_key=active_unmapped_key,
                    episode_label="Final current-run application state",
                )
            )
            seen_observation_ids.add(final_id)
    return tuple(frames), frozenset(covered_surface_ids)


def _build_visual_episodes(
    frames: tuple[_VisualFrame, ...],
    observation_by_id: dict[str, Observation],
) -> tuple[VisualEvidenceEpisode, ...]:
    grouped: list[tuple[str, str, list[GoalEvidenceSource]]] = []
    screenshot_source_by_sha: dict[str, GoalEvidenceSource] = {}
    for sequence_number, frame in enumerate(frames, start=1):
        observation = observation_by_id.get(frame.observation_id)
        if observation is None:
            continue
        labels = (_visual_frame_label(sequence_number, frame, observation.keyboard_visible),)
        existing_source = screenshot_source_by_sha.get(observation.redacted_screenshot.sha256)
        if existing_source is not None:
            replacement = GoalEvidenceSource(
                observation_id=existing_source.observation_id,
                labels=tuple(dict.fromkeys((*existing_source.labels, *labels))),
                screenshot=existing_source.screenshot,
                related_observation_ids=tuple(
                    dict.fromkeys((*existing_source.related_observation_ids, frame.observation_id))
                ),
            )
            screenshot_source_by_sha[observation.redacted_screenshot.sha256] = replacement
            for _unused_key, _unused_label, sources in grouped:
                for index, source in enumerate(sources):
                    if source.observation_id == existing_source.observation_id:
                        sources[index] = replacement
            continue
        source = GoalEvidenceSource(
            observation_id=frame.observation_id,
            labels=labels,
            screenshot=observation.redacted_screenshot,
        )
        screenshot_source_by_sha[observation.redacted_screenshot.sha256] = source
        if not grouped or grouped[-1][0] != frame.episode_key:
            grouped.append((frame.episode_key, frame.episode_label, []))
        grouped[-1][2].append(source)
    frame_by_observation_id = {frame.observation_id: frame for frame in frames}
    episodes: list[VisualEvidenceEpisode] = []
    for index, (_unused_key, label, images) in enumerate(grouped, start=1):
        if not images:
            continue
        episode_frames = tuple(
            frame_by_observation_id[observation_id]
            for image in images
            for observation_id in image.observation_ids
            if observation_id in frame_by_observation_id
        )
        episodes.append(
            VisualEvidenceEpisode(
                episode_id=f"visual_episode_{index:03d}",
                label=label,
                images=tuple(images),
                action_facts=tuple(
                    dict.fromkeys(
                        _action_fact(frame.step)
                        for frame in episode_frames
                        if frame.step is not None and frame.step.outcome is not None
                    )
                ),
            )
        )
    return tuple(episodes)


def _visual_frame_label(
    sequence_number: int,
    frame: _VisualFrame,
    keyboard_visible: bool | None,
) -> str:
    step = frame.step
    if step is None:
        return f"F{sequence_number:03d}. Final observation. Keyboard={keyboard_visible}."
    return (
        f"F{sequence_number:03d}. Step={step.step_index}. "
        f"Action={step.proposal.action.value}. Phase={frame.phase}. "
        f"Source={step.decision.source.value}. Keyboard={keyboard_visible}."
    )


def _required_observation_labels(
    surfaces: tuple[DiscoveredTestSurface, ...],
    agents: tuple[AgentResult, ...],
    steps: tuple[AgentStep, ...],
    *,
    include_following_observation: bool = False,
) -> dict[str, tuple[str, ...]]:
    labels_by_observation: dict[str, tuple[str, ...]] = {}
    relevant_indexes = {
        index for surface in surfaces for index in surface.provenance.procedure_step_indexes
    }
    for surface in surfaces:
        indexes = set(surface.provenance.procedure_step_indexes)
        candidates = tuple(step for step in steps if step.decision.reused_segment_index in indexes)
        if not candidates:
            continue
        latest_step = max(candidates, key=lambda item: item.step_index)
        observation_id = latest_step.settled_observation_id or latest_step.source_observation_id
        existing_labels = labels_by_observation.get(observation_id, ())
        labels_by_observation[observation_id] = tuple(dict.fromkeys((*existing_labels, surface.summary)))
    if include_following_observation and relevant_indexes:
        following = min(
            (
                step
                for step in steps
                if step.decision.reused_segment_index is not None
                and step.decision.reused_segment_index > max(relevant_indexes)
            ),
            key=lambda step: step.decision.reused_segment_index or 0,
            default=None,
        )
        if following is not None:
            observation_id = following.settled_observation_id or following.source_observation_id
            labels_by_observation[observation_id] = (
                "First settled observation after the required test surfaces.",
            )
    if not labels_by_observation and agents:
        labels_by_observation[agents[-1].final_observation_id] = (
            "Final current-run scenario observation.",
        )
    return labels_by_observation


def _load_evidence_images(
    root: Path,
    labels_by_observation: dict[str, tuple[str, ...]],
) -> tuple[GoalEvidenceSource, ...]:
    try:
        observations = ObservationStore(root).load_selected(set(labels_by_observation))
    except ObservationCaptureError:
        return ()
    by_id = {item.observation_id: item for item in observations}
    return tuple(
        GoalEvidenceSource(
            observation_id=observation_id,
            labels=labels,
            screenshot=by_id[observation_id].redacted_screenshot,
        )
        for observation_id, labels in labels_by_observation.items()
        if observation_id in by_id
    )


def _action_fact(step: AgentStep) -> str:
    outcome = step.outcome
    if outcome is None:
        raise AssertionError("An action fact requires an outcome.")
    target = step.proposal.target_expression or "no named target"
    signals = ", ".join(sorted(signal.value for signal in outcome.signals)) or "none"
    return (
        f"Step {step.step_index}: action={step.proposal.action.value}. Target={target}. "
        f"Disposition={outcome.disposition.value}. Signals={signals}."
    )
