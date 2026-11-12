"""Isolate post-discovery procedure compilation from the run terminal."""

from __future__ import annotations

from yewo.agent.discovery.conclusion import persist_discovery_outcome
from yewo.agent.discovery.records import DiscoveryResult
from yewo.observations import ScreenComparisonPolicy
from yewo.agent.memory.records import MemoryCompatibility
from yewo.agent.memory.runtime_writeback import write_back_discovery_knowledge
from yewo.agent.memory.store import AppMemoryStore, load_discovery_memory
from yewo.agent.replay.procedure.store import CompiledProcedureStore
from yewo.agent.replay.procedure.records import ProcedureReplayTerminal
from yewo.agent.replay.reconciliation.compilation import compile_completed_discovery_run
from yewo.agent.replay.reconciliation.records import ProcedureCompilationReport
from yewo.models import ModelClient
from yewo.run_narration import current_narrator
from yewo.runs import RunExecutionContext, RunTerminalCategory, RunWorkflowOutcome


def persist_discovery_outcome_and_compile(
    context: RunExecutionContext,
    result: DiscoveryResult,
    require_reusable_evidence: bool,
    model_client: ModelClient,
    model_profile_id: str,
    comparison: ScreenComparisonPolicy,
    procedure_store: CompiledProcedureStore | None = None,
    compatibility: MemoryCompatibility | None = None,
    precompiled_report: ProcedureCompilationReport | None = None,
    memory_store: AppMemoryStore | None = None,
) -> RunWorkflowOutcome:
    """Persist discovery first and isolate the non-executing compilation step."""
    outcome = persist_discovery_outcome(context.workspace, result, require_reusable_evidence)
    if precompiled_report is not None:
        _narrate_compilation(
            f"Procedure compilation finished with status {precompiled_report.status.value}."
        )
        _write_back_if_configured(context, result, memory_store, compatibility, procedure_store)
        return outcome
    if outcome.category is not RunTerminalCategory.COMPLETED:
        _write_back_if_configured(context, result, memory_store, compatibility, procedure_store)
        return outcome
    if (
        result.procedure_replay is not None
        and result.procedure_replay.terminal is ProcedureReplayTerminal.COMPLETED
    ):
        _narrate_compilation("Completed procedure replay requires no new compilation.")
        _write_back_if_configured(context, result, memory_store, compatibility, procedure_store)
        return outcome
    try:
        memory = (
            load_discovery_memory(memory_store, compatibility)
            if memory_store is not None and compatibility is not None
            else None
        )
        report = compile_completed_discovery_run(
            root=context.workspace.root,
            run_id=context.workspace.start.run_id,
            discovery_result=result,
            model_client=model_client,
            model_profile_id=model_profile_id,
            comparison=comparison,
            procedure_store=procedure_store,
            compatibility=compatibility,
            memory=memory,
        )
    except Exception as error:  # noqa: BLE001 - isolated compilation boundary
        _narrate_compilation(f"Procedure compilation failed: {type(error).__name__}.")
    else:
        _narrate_compilation(f"Procedure compilation finished with status {report.status.value}.")
    _write_back_if_configured(context, result, memory_store, compatibility, procedure_store)
    return outcome


def _write_back_if_configured(
    context: RunExecutionContext,
    result: DiscoveryResult,
    memory_store: AppMemoryStore | None,
    compatibility: MemoryCompatibility | None,
    procedure_store: CompiledProcedureStore | None,
) -> None:
    if memory_store is None or compatibility is None:
        return
    try:
        write_back_discovery_knowledge(
            context.workspace.root,
            context.workspace.start.run_id,
            result,
            memory_store,
            compatibility,
            procedure_store,
        )
    except Exception as error:  # noqa: BLE001 - preserve the durable run terminal
        _narrate_compilation(f"Memory write-back failed: {type(error).__name__}.")


def _narrate_compilation(message: str) -> None:
    narrator = current_narrator()
    if narrator is not None:
        narrator.record("replay-reconciliation", message)
