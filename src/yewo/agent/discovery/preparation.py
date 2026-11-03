"""Prepare discovery memory before the automation session starts."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from yewo.agent.memory.records import AppMemory, build_memory_compatibility
from yewo.agent.composition import build_model_client
from yewo.agent.discovery.records import DiscoveryRequest
from yewo.agent.memory.store import AppMemoryStore
from yewo.agent.planning.planner import TaskProgressContext
from yewo.agent.replay.preparation import (
    ReplayProcedureSelection,
    prepare_replay_procedure,
)
from yewo.agent.replay.procedure_store import CompiledProcedureStore
from yewo.models import ModelProfileRegistry
from yewo.observations import RedactionRegion
from yewo.runs import RunPreparationContext


@dataclass(frozen=True)
class DiscoveryPreparation:
    """Memory and replay guidance produced before the automation session starts."""

    memory_id: str | None
    progress_context: TaskProgressContext | None
    procedure_selection: ReplayProcedureSelection | None = None


def prepare_compiled_procedure(
    context: RunPreparationContext,
    request: DiscoveryRequest,
    memory_store: AppMemoryStore,
    procedure_store: CompiledProcedureStore,
    model_profiles: ModelProfileRegistry,
    replay_preparation_profile_id: str,
    scrub_text: Callable[[str], str],
    redaction_regions: Sequence[RedactionRegion],
    maximum_model_requests: int | None,
) -> tuple[DiscoveryPreparation | None, int]:
    """Select one exact or LLM-equivalent compiled procedure before Appium."""
    objective = request.objectives[0]
    compatibility = build_memory_compatibility(
        context.application,
        context.workspace.start.request.application,
    )
    memory = memory_store.load(compatibility)
    procedure_preparation = prepare_replay_procedure(
        objective,
        compatibility,
        memory,
        procedure_store,
        lambda: build_model_client(
            context.workspace.root,
            context.workspace.start.run_id,
            model_profiles,
            scrub_text,
            bool(redaction_regions),
        ),
        replay_preparation_profile_id,
        maximum_model_requests,
    )
    selection = procedure_preparation.selection
    if selection is None:
        return None, procedure_preparation.model_request_count
    return (
        DiscoveryPreparation(
            memory_id=memory.memory_id if memory is not None else None,
            progress_context=TaskProgressContext(
                preparation_planning_requests=selection.preparation_model_requests
            ),
            procedure_selection=selection,
        ),
        procedure_preparation.model_request_count,
    )


def prepare_memory_identity(
    context: RunPreparationContext,
    memory_store: AppMemoryStore,
    initial_model_requests: int,
) -> DiscoveryPreparation:
    """Record the retained memory snapshot when no procedure was selected."""
    compatibility = build_memory_compatibility(
        context.application,
        context.workspace.start.request.application,
    )
    memory = memory_store.load(compatibility)
    return DiscoveryPreparation(
        memory_id=memory.memory_id if memory is not None else None,
        progress_context=(
            TaskProgressContext(preparation_planning_requests=initial_model_requests)
            if initial_model_requests
            else None
        ),
    )


def matching_preparation(
    preparation: object | None,
    memory: AppMemory | None,
) -> DiscoveryPreparation | None:
    """Use preparation only when it matches the selected memory snapshot."""
    if not isinstance(preparation, DiscoveryPreparation):
        return None
    memory_id = memory.memory_id if memory is not None else None
    if preparation.memory_id != memory_id:
        return None
    return preparation
