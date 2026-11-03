"""Public replay-preparation roles for current and retained evidence."""

from collections.abc import Callable
from dataclasses import dataclass

from yewo.agent.discovery.records import DiscoveryObjective
from yewo.agent.memory.records import AppMemory, MemoryCompatibility
from yewo.agent.replay.procedure.hazards import compiled_procedure_has_known_hazard
from yewo.agent.replay.procedure.objective_matching import (
    ObjectiveMatcher,
    ReplayProcedureSelection,
    exact_procedure_selection,
)
from yewo.agent.replay.procedure.store import CompiledProcedureStore
from yewo.agent.replay.reconciliation.compilation import compile_completed_run
from yewo.models import ModelClient


@dataclass(frozen=True)
class ReplayProcedurePreparation:
    """One optional procedure selection and its preparation cost."""

    selection: ReplayProcedureSelection | None
    model_request_count: int = 0


def prepare_replay_procedure(
    objective: DiscoveryObjective,
    compatibility: MemoryCompatibility,
    memory: AppMemory | None,
    procedure_store: CompiledProcedureStore,
    model_client_factory: Callable[[], ModelClient],
    model_profile_id: str,
    maximum_model_requests: int | None,
) -> ReplayProcedurePreparation:
    """Use exact lookup first and one fail-closed LLM match only on a miss."""
    exact_bundle = procedure_store.load(
        compatibility,
        objective.objective_id,
        objective.statement,
    )
    current_conditions = objective.completion_conditions or (objective.statement,)
    if (
        exact_bundle is not None
        and exact_bundle.evidence.task.completion_conditions == current_conditions
        and not compiled_procedure_has_known_hazard(
            exact_bundle.evidence,
            exact_bundle.procedure,
            memory,
        )
    ):
        return ReplayProcedurePreparation(exact_procedure_selection(exact_bundle))
    if maximum_model_requests is not None and maximum_model_requests < 2:
        return ReplayProcedurePreparation(None)
    candidates = tuple(
        candidate
        for candidate in procedure_store.compatible_candidates(compatibility)
        if not compiled_procedure_has_known_hazard(
            candidate.evidence,
            candidate.procedure,
            memory,
        )
    )
    if not candidates:
        return ReplayProcedurePreparation(None)
    matcher = ObjectiveMatcher(model_client_factory(), model_profile_id)
    selection = matcher.match(objective, candidates)
    return ReplayProcedurePreparation(selection, matcher.model_request_count)


__all__ = [
    "ObjectiveMatcher",
    "ReplayProcedurePreparation",
    "ReplayProcedureSelection",
    "compile_completed_run",
    "exact_procedure_selection",
    "prepare_replay_procedure",
]
