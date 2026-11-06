"""Capture current-screen analyses into AppMemory control knowledge."""

from collections.abc import Mapping

from yewo.agent.memory.knowledge import ControlKnowledge, KnowledgeEvidenceReference
from yewo.agent.memory.records import AppMemory, derived_memory_identifier
from yewo.agent.memory.screens import observation_matches_screen
from yewo.agent.memory.writeback import MemoryWritebackSkip
from yewo.agent.runtime.records import AgentResult
from yewo.agent.screen_analysis import (
    CurrentScreenAnalysis,
    ScreenElementAffordance,
    screen_role_affordance,
)
from yewo.artifacts import RecordIdentifier
from yewo.observations import Observation


def capture_controls(
    memory: AppMemory,
    source_run_id: RecordIdentifier,
    result: AgentResult,
    screen_analyses: tuple[CurrentScreenAnalysis, ...],
    observations: Mapping[RecordIdentifier, Observation],
    procedure_attempt_id: RecordIdentifier | None,
    skipped_records: list[MemoryWritebackSkip],
) -> tuple[ControlKnowledge, ...]:
    """Return screen-scoped visible controls with deterministic exercise evidence."""
    if not screen_analyses:
        skipped_records.append(
            MemoryWritebackSkip(
                record_kind="control",
                reason="No screen-analysis evidence was available for control capture.",
            )
        )
        return ()
    captured: dict[RecordIdentifier, ControlKnowledge] = {}
    for analysis in screen_analyses:
        observation = observations.get(analysis.observation_id)
        screen_id = _screen_id(memory, observation) if observation is not None else None
        if screen_id is None:
            skipped_records.append(
                MemoryWritebackSkip(
                    record_kind="control",
                    reason="No unique retained screen matched screen-analysis evidence.",
                )
            )
            continue
        for element in analysis.elements:
            if (
                screen_role_affordance(element.kind.semantic)
                is not ScreenElementAffordance.INTERACTIVE
            ):
                continue
            control_id = derived_memory_identifier(
                "visible-control",
                screen_id,
                element.kind.semantic.value,
                element.kind.visual.value,
                element.name.casefold(),
            )
            exercised = any(
                step.receipt is not None
                and step.proposal.source_observation_id == analysis.observation_id
                and step.proposal.target_element_ref == element.element_ref
                for step in result.steps
            )
            update = ControlKnowledge(
                control_id=control_id,
                screen_id=screen_id,
                name=element.name,
                kind=element.kind,
                state=element.state,
                options=element.options,
                exercised=exercised,
                sources=(
                    KnowledgeEvidenceReference(
                        run_id=source_run_id,
                        observation_id=analysis.observation_id,
                        procedure_attempt_id=procedure_attempt_id,
                    ),
                ),
            )
            captured[control_id] = _merge_control(captured.get(control_id), update)
    return tuple(captured.values())


def _merge_control(
    current: ControlKnowledge | None,
    update: ControlKnowledge,
) -> ControlKnowledge:
    if current is None:
        return update
    return update.model_copy(
        update={
            "options": tuple(dict.fromkeys((*current.options, *update.options))),
            "exercised": current.exercised or update.exercised,
            "sources": tuple(dict.fromkeys((*current.sources, *update.sources))),
        }
    )


def _screen_id(memory: AppMemory, observation: Observation) -> RecordIdentifier | None:
    matches = tuple(
        screen.screen_id
        for screen in memory.screens
        if observation_matches_screen(observation, screen, memory.compatibility.screen_comparison)
    )
    return matches[0] if len(matches) == 1 else None
