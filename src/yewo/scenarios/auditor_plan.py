"""Deterministic request planning for independent scenario audit."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from yewo.agent.planning.goal_evidence import (
    AdaptiveGoalEvidencePacker,
    GoalEvidenceImage,
)
from yewo.models import ModelImageConstraints, ModelInputError
from yewo.project_config import CoverageCategory
from yewo.scenarios.auditor_evidence import (
    ScenarioEvidence,
    pack_readable_evidence_batches,
)
from yewo.scenarios.records import ScenarioExpectedOutcome


UNIVERSAL_BATCH_CATEGORIES = frozenset(
    {
        CoverageCategory.DATA_CONSISTENCY,
        CoverageCategory.STATE,
        CoverageCategory.NAVIGATION,
        CoverageCategory.UI_VISUAL,
        CoverageCategory.SECURITY,
    }
)


@dataclass(frozen=True)
class AuditTarget:
    """One expected outcome and its exact current-run evidence."""

    category: CoverageCategory
    outcome: ScenarioExpectedOutcome
    evidence: ScenarioEvidence


@dataclass(frozen=True)
class AuditRequestBatch:
    """One bounded model request with explicit aggregation semantics."""

    targets: tuple[AuditTarget, ...]
    evidence: ScenarioEvidence
    images: tuple[GoalEvidenceImage, ...]
    batch_index: int = 0
    batch_count: int = 1
    universal: bool = False
    recovery_depth: int = 0
    visual_episode_id: str | None = None
    visual_episode_label: str | None = None


@dataclass(frozen=True)
class AuditRequestPlan:
    """Executable audit requests and outcomes that exceeded capacity."""

    requests: tuple[AuditRequestBatch, ...]
    capacity_outcome_ids: frozenset[str]


def build_audit_request_plan(
    root: Path,
    constraints: ModelImageConstraints,
    targets: tuple[AuditTarget, ...],
) -> AuditRequestPlan:
    """Group compatible checks and split universal visual coverage."""
    packer = AdaptiveGoalEvidencePacker(root, constraints)
    screen_visual = tuple(
        target
        for target in targets
        if target.category is CoverageCategory.UI_VISUAL and target.evidence.visual_episodes
    )
    visual = tuple(
        target
        for target in targets
        if target not in screen_visual
        if target.category in UNIVERSAL_BATCH_CATEGORIES and not packer.fits(target.evidence.images)
    )
    regular = tuple(
        target for target in targets if target not in visual and target not in screen_visual
    )
    requests: list[AuditRequestBatch] = []
    capacity: set[str] = set()
    for target in screen_visual:
        episode_requests, exceeded_capacity = _visual_episode_requests(
            root,
            constraints,
            target,
        )
        requests.extend(episode_requests)
        if exceeded_capacity:
            capacity.add(target.outcome.outcome_id)
    for target in visual:
        try:
            batches = pack_readable_evidence_batches(
                root,
                constraints,
                target.evidence.images,
            )
        except ModelInputError:
            capacity.add(target.outcome.outcome_id)
            continue
        for index, images in enumerate(batches):
            evidence = _select_image_evidence(target.evidence, images)
            requests.append(
                AuditRequestBatch(
                    targets=(
                        AuditTarget(
                            category=target.category,
                            outcome=target.outcome,
                            evidence=evidence,
                        ),
                    ),
                    evidence=evidence,
                    images=images,
                    batch_index=index,
                    batch_count=len(batches),
                    universal=True,
                )
            )
    regular_requests, regular_capacity = _group_regular_targets(
        root,
        constraints,
        regular,
    )
    requests.extend(regular_requests)
    capacity.update(regular_capacity)
    return AuditRequestPlan(
        requests=tuple(requests),
        capacity_outcome_ids=frozenset(capacity),
    )


def split_audit_request_batch(
    root: Path,
    constraints: ModelImageConstraints,
    batch: AuditRequestBatch,
    *,
    maximum_recovery_depth: int,
) -> tuple[AuditRequestBatch, ...]:
    """Split one exhausted universal batch at source-observation boundaries."""
    sources = batch.evidence.images
    if not batch.universal or batch.recovery_depth >= maximum_recovery_depth or len(sources) < 2:
        return ()
    midpoint = len(sources) // 2
    partitions = (sources[:midpoint], sources[midpoint:])
    packer = AdaptiveGoalEvidencePacker(root, constraints)
    split_batches: list[AuditRequestBatch] = []
    for index, partition in enumerate(partitions):
        evidence = batch.evidence.model_copy(update={"images": partition})
        targets = tuple(
            AuditTarget(
                category=target.category,
                outcome=target.outcome,
                evidence=evidence,
            )
            for target in batch.targets
        )
        split_batches.append(
            AuditRequestBatch(
                targets=targets,
                evidence=evidence,
                images=packer.pack(partition),
                batch_index=index,
                batch_count=len(partitions),
                universal=True,
                recovery_depth=batch.recovery_depth + 1,
                visual_episode_id=batch.visual_episode_id,
                visual_episode_label=batch.visual_episode_label,
            )
        )
    return tuple(split_batches)


def _visual_episode_requests(
    root: Path,
    constraints: ModelImageConstraints,
    target: AuditTarget,
) -> tuple[tuple[AuditRequestBatch, ...], bool]:
    """Build bounded chronological requests for every visual screen episode."""
    requests: list[AuditRequestBatch] = []
    for episode in target.evidence.visual_episodes:
        for start in range(0, len(episode.images), 4):
            sources = episode.images[start : start + 4]
            try:
                batches = pack_readable_evidence_batches(root, constraints, sources)
            except ModelInputError:
                return tuple(requests), True
            for images in batches:
                episode_evidence = target.evidence.model_copy(
                    update={"action_facts": episode.action_facts}
                )
                evidence = _select_image_evidence(episode_evidence, images)
                requests.append(
                    AuditRequestBatch(
                        targets=(
                            AuditTarget(
                                category=target.category,
                                outcome=target.outcome,
                                evidence=evidence,
                            ),
                        ),
                        evidence=evidence,
                        images=images,
                        batch_index=len(requests),
                        batch_count=0,
                        universal=True,
                        visual_episode_id=episode.episode_id,
                        visual_episode_label=episode.label,
                    )
                )
    request_count = len(requests)
    return tuple(replace(request, batch_count=request_count) for request in requests), False


def _group_regular_targets(
    root: Path,
    constraints: ModelImageConstraints,
    targets: tuple[AuditTarget, ...],
) -> tuple[tuple[AuditRequestBatch, ...], frozenset[str]]:
    packer = AdaptiveGoalEvidencePacker(root, constraints)
    requests: list[AuditRequestBatch] = []
    capacity: set[str] = set()
    current: tuple[AuditTarget, ...] = ()
    current_evidence: ScenarioEvidence | None = None
    for target in targets:
        candidate = (*current, target)
        evidence = merge_target_evidence(candidate)
        if not packer.fits(evidence.images):
            if current and current_evidence is not None:
                requests.append(
                    AuditRequestBatch(
                        current,
                        current_evidence,
                        packer.pack(current_evidence.images),
                    )
                )
                current = ()
                current_evidence = None
            if not packer.fits(target.evidence.images):
                capacity.add(target.outcome.outcome_id)
                continue
            current = (target,)
            current_evidence = target.evidence
            continue
        current = candidate
        current_evidence = evidence
    if current and current_evidence is not None:
        requests.append(
            AuditRequestBatch(
                current,
                current_evidence,
                packer.pack(current_evidence.images),
            )
        )
    return tuple(requests), frozenset(capacity)


def merge_target_evidence(targets: tuple[AuditTarget, ...]) -> ScenarioEvidence:
    """Merge exact evidence without duplicating observations or facts."""
    images = {image.observation_id: image for target in targets for image in target.evidence.images}
    owner_counts = {
        owner: max(target.evidence.owner_counts.get(owner, 0) for target in targets)
        for target in targets
        for owner in target.evidence.owner_counts
    }
    return ScenarioEvidence(
        images=tuple(images.values()),
        owner_counts=owner_counts,
        action_facts=tuple(
            dict.fromkeys(fact for target in targets for fact in target.evidence.action_facts)
        ),
        capability_facts=tuple(
            dict.fromkeys(fact for target in targets for fact in target.evidence.capability_facts)
        ),
        required_surface_ids=tuple(
            dict.fromkeys(
                surface_id
                for target in targets
                for surface_id in target.evidence.required_surface_ids
            )
        ),
        covered_surface_ids=tuple(
            dict.fromkeys(
                surface_id
                for target in targets
                for surface_id in target.evidence.covered_surface_ids
            )
        ),
        missing_surface_ids=tuple(
            dict.fromkeys(
                surface_id
                for target in targets
                for surface_id in target.evidence.missing_surface_ids
            )
        ),
    )


def _select_image_evidence(
    evidence: ScenarioEvidence,
    images: tuple[GoalEvidenceImage, ...],
) -> ScenarioEvidence:
    observation_ids = {
        observation_id for image in images for observation_id in image.observation_ids
    }
    return evidence.model_copy(
        update={
            "images": tuple(
                source
                for source in evidence.images
                if any(item in observation_ids for item in source.observation_ids)
            ),
            "visual_episodes": (),
        }
    )
