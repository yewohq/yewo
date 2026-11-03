"""Deterministic compiler gates for platform-observed application hazards."""

from __future__ import annotations

from yewo.agent.memory.knowledge import HazardKnowledge
from yewo.agent.memory.records import AppMemory, LogicalScreen, ObservationVariant
from yewo.agent.replay.reconciliation.records import (
    CompilerCheck,
    CompilerReason,
    EvidenceDocument,
    EvidenceState,
    EvidenceStep,
    ReplayProcedure,
)
from yewo.observations import fingerprint_distance


def known_hazard_reasons(
    evidence: EvidenceDocument,
    selected_steps: tuple[EvidenceStep, ...],
    memory: AppMemory | None,
) -> tuple[CompilerReason, ...]:
    """Reject selected actions that match compatible platform-observed hazards."""
    if memory is None or not memory.app_knowledge.hazards:
        return ()
    if memory.compatibility.application != evidence.run.application:
        return (
            CompilerReason(
                check=CompilerCheck.AVOIDS_KNOWN_HAZARDS,
                reason="Known-hazard checks require memory for the same application build.",
            ),
        )
    hazards_by_screen = _hazards_by_screen(memory)
    reasons: list[CompilerReason] = []
    for step in selected_steps:
        state = evidence.states[step.source_state_id]
        for screen in memory.screens:
            if not _state_matches_screen(state, screen, memory):
                continue
            reasons.extend(
                _step_hazard_reasons(step, hazards_by_screen.get(screen.screen_id, ()))
            )
    return tuple(reasons)


def compiled_procedure_hazard_reasons(
    evidence: EvidenceDocument,
    procedure: ReplayProcedure,
    memory: AppMemory | None,
) -> tuple[CompilerReason, ...]:
    """Recheck a stored procedure against hazards learned after compilation."""
    evidence_by_step = {step.step: step for step in evidence.steps}
    selected_steps = tuple(
        evidence_by_step[procedure_step.source_step] for procedure_step in procedure.steps
    )
    return known_hazard_reasons(evidence, selected_steps, memory)


def compiled_procedure_has_known_hazard(
    evidence: EvidenceDocument,
    procedure: ReplayProcedure,
    memory: AppMemory | None,
) -> bool:
    """Return whether current memory blocks a stored procedure."""
    return bool(compiled_procedure_hazard_reasons(evidence, procedure, memory))


def _hazards_by_screen(memory: AppMemory) -> dict[str, tuple[HazardKnowledge, ...]]:
    grouped: dict[str, list[HazardKnowledge]] = {}
    for hazard in memory.app_knowledge.hazards:
        grouped.setdefault(hazard.screen_id, []).append(hazard)
    return {
        screen_id: tuple(sorted(hazards, key=lambda item: item.hazard_id))
        for screen_id, hazards in grouped.items()
    }


def _step_hazard_reasons(
    step: EvidenceStep,
    hazards: tuple[HazardKnowledge, ...],
) -> tuple[CompilerReason, ...]:
    reasons: list[CompilerReason] = []
    for hazard in hazards:
        if hazard.action is not step.action or not _target_matches(hazard, step):
            continue
        reasons.append(
            CompilerReason(
                check=CompilerCheck.AVOIDS_KNOWN_HAZARDS,
                source_step=step.step,
                reason=(
                    f"Step {step.step} matches platform-observed {hazard.kind.value} hazard "
                    f"'{hazard.hazard_id}'. No deterministic safe precondition is recorded."
                ),
                hazard_id=hazard.hazard_id,
                hazard_sources=hazard.sources,
            )
        )
    return tuple(reasons)


def _target_matches(hazard: HazardKnowledge, step: EvidenceStep) -> bool:
    if hazard.target_description is None:
        return True
    target = step.proposal.target_expression
    return target is not None and _normalize_target(target) == _normalize_target(
        hazard.target_description
    )


def _normalize_target(value: str) -> str:
    return " ".join(value.casefold().split())


def _state_matches_screen(
    state: EvidenceState,
    screen: LogicalScreen,
    memory: AppMemory,
) -> bool:
    if set(state.observation_ids) & {variant.observation_id for variant in screen.variants}:
        return True
    maximum_distance = memory.compatibility.screen_comparison.maximum_phash_distance
    return any(
        _state_matches_variant(state, variant, maximum_distance)
        for variant in screen.variants
    )


def _state_matches_variant(
    state: EvidenceState,
    variant: ObservationVariant,
    maximum_distance: int,
) -> bool:
    if (
        state.foreground_application_id != variant.foreground_application_id
        or state.platform_context != variant.platform_context
        or state.keyboard_visible is not variant.keyboard_visible
        or state.orientation is not variant.orientation
        or (state.width_px, state.height_px) != (variant.width_px, variant.height_px)
    ):
        return False
    for fingerprint in state.comparison_fingerprints():
        try:
            if fingerprint_distance(fingerprint, variant.fingerprint) <= maximum_distance:
                return True
        except ValueError:
            continue
    return False
