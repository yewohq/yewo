"""Capture action-verifier control observations into AppMemory knowledge."""

from collections.abc import Mapping

from yewo.agent.memory.knowledge import ControlKnowledge, KnowledgeEvidenceReference
from yewo.agent.memory.records import AppMemory, derived_memory_identifier
from yewo.agent.memory.screens import observation_matches_screen
from yewo.agent.memory.writeback import MemoryWritebackSkip
from yewo.agent.runtime.records import AgentResult
from yewo.artifacts import RecordIdentifier
from yewo.observations import Observation


def capture_controls(
    memory: AppMemory,
    source_run_id: RecordIdentifier,
    result: AgentResult,
    observations: Mapping[RecordIdentifier, Observation],
    procedure_attempt_id: RecordIdentifier | None,
    skipped_records: list[MemoryWritebackSkip],
) -> tuple[ControlKnowledge, ...]:
    """Return screen-scoped visible controls with deterministic exercise evidence."""
    captured: dict[RecordIdentifier, ControlKnowledge] = {}
    for visible in result.visible_control_observations:
        observation = observations.get(visible.observation_id)
        screen_id = _screen_id(memory, observation) if observation is not None else None
        if screen_id is None:
            skipped_records.append(
                MemoryWritebackSkip(
                    record_kind="control",
                    reason="No unique retained screen matched visible-control evidence.",
                )
            )
            continue
        dispatched = any(
            step.proposal.proposal_id == visible.proposal_id and step.receipt is not None
            for step in result.steps
        )
        for control in visible.controls:
            control_id = derived_memory_identifier(
                "visible-control", screen_id, control.kind.value, control.name.casefold()
            )
            update = ControlKnowledge(
                control_id=control_id,
                screen_id=screen_id,
                name=control.name,
                kind=control.kind,
                state=control.state,
                options=control.options,
                exercised=dispatched and control.matches_proposal_target,
                sources=(
                    KnowledgeEvidenceReference(
                        run_id=source_run_id,
                        observation_id=visible.observation_id,
                        procedure_attempt_id=procedure_attempt_id,
                    ),
                ),
            )
            current = captured.get(control_id)
            captured[control_id] = (
                update
                if current is None
                else update.model_copy(
                    update={
                        "options": tuple(dict.fromkeys((*current.options, *update.options))),
                        "exercised": current.exercised or update.exercised,
                        "sources": tuple(dict.fromkeys((*current.sources, *update.sources))),
                    }
                )
            )
    return tuple(captured.values())


def _screen_id(memory: AppMemory, observation: Observation) -> RecordIdentifier | None:
    matches = tuple(
        screen.screen_id
        for screen in memory.screens
        if observation_matches_screen(observation, screen, memory.compatibility.screen_comparison)
    )
    return matches[0] if len(matches) == 1 else None
