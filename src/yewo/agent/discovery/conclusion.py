"""Discovery evidence persistence and terminal classification."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from yewo.agent.memory.records import AppMemory
from yewo.agent.memory.store import AppMemoryStore, AppMemoryStoreError
from yewo.agent.runtime.records import AgentResult, AgentTerminalReason, agent_terminal_category
from yewo.agent.replay.procedure.records import ProcedureReplayResult
from yewo.agent.replay.reconciliation.records import ReplayProcedure
from yewo.artifacts import ARTIFACTS_DIRECTORY, AtomicArtifactWriter, ArtifactWriteError
from yewo.runs import RunTerminalCategory, RunWorkflowOutcome, RunWorkspace
from yewo.agent.discovery.narration import narrate_discovery_result
from yewo.agent.discovery.records import (
    DiscoveryCoverage,
    DiscoveryObjectiveState,
    DiscoveryRequest,
    DiscoveryResult,
)
from yewo.agent.planning.goal_verifier import GoalVerification


class DiscoveryEvidenceError(RuntimeError):
    """The discovery workflow could not persist its canonical conclusion."""


DISCOVERY_RESULT_PATH = ARTIFACTS_DIRECTORY / "agent" / "discovery" / "result.json"


def load_discovery_result(run_root: Path) -> DiscoveryResult:
    """Load one regular, schema-valid discovery result from a run workspace."""
    path = run_root / DISCOVERY_RESULT_PATH
    if path.is_symlink() or not path.is_file():
        raise DiscoveryEvidenceError("The discovery result is unavailable.")
    try:
        return DiscoveryResult.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise DiscoveryEvidenceError("Could not read the discovery result.") from error


def build_discovery_result(
    request: DiscoveryRequest,
    agent_result: AgentResult,
    coverage: DiscoveryCoverage,
    procedure_replay: ProcedureReplayResult | None,
    verification: GoalVerification | None,
    compiled_procedure: ReplayProcedure | None,
    memory: AppMemory | None,
) -> DiscoveryResult:
    """Build the canonical successful workflow result."""
    return DiscoveryResult(
        request=request,
        terminal_reason=agent_result.terminal_reason,
        coverage=coverage,
        agent_result=agent_result,
        procedure_replay=procedure_replay,
        verification=verification,
        compiled_procedure_id=compiled_procedure.procedure_id if compiled_procedure else None,
        compiled_procedure_step_count=(
            len(compiled_procedure.steps) if compiled_procedure else None
        ),
        memory_id=memory.memory_id if memory is not None else None,
    )


def persist_discovery_result(workspace: RunWorkspace, result: DiscoveryResult) -> None:
    """Persist and narrate one canonical discovery result."""
    content = result.model_dump_json(indent=2).encode("utf-8") + b"\n"
    try:
        AtomicArtifactWriter(workspace.root).write(DISCOVERY_RESULT_PATH, content)
        narrate_discovery_result(result)
    except ArtifactWriteError as error:
        raise DiscoveryEvidenceError("Could not persist the discovery result.") from error


def persist_discovery_memory_failure(
    workspace: RunWorkspace,
    request: DiscoveryRequest,
    agent_result: AgentResult,
    coverage: DiscoveryCoverage,
    verification: GoalVerification | None,
    durable_memory_id: str | None,
    procedure_replay: ProcedureReplayResult | None = None,
) -> None:
    """Retain the run conclusion before propagating a memory-store failure."""
    unresolved = DiscoveryCoverage(
        entries=tuple(
            entry.model_copy(
                update={
                    "state": DiscoveryObjectiveState.UNRESOLVED,
                    "procedure_id": None,
                    "evidence_observation_ids": (),
                    "reason": "The run produced evidence, but Yewo could not update app memory.",
                }
            )
            for entry in coverage.entries
        )
    )
    persist_discovery_result(
        workspace,
        DiscoveryResult(
            request=request,
            terminal_reason=agent_result.terminal_reason,
            coverage=unresolved,
            agent_result=agent_result,
            procedure_replay=procedure_replay,
            verification=verification,
            memory_id=durable_memory_id,
            memory_persistence_error="Yewo could not update app memory.",
        ),
    )


def save_discovery_memory(
    memory_store: AppMemoryStore,
    memory: AppMemory,
    source_run_root: Path,
    workspace: RunWorkspace,
    request: DiscoveryRequest,
    agent_result: AgentResult,
    coverage: DiscoveryCoverage,
    verification: GoalVerification | None,
    durable_memory_id: str | None,
    procedure_replay: ProcedureReplayResult | None = None,
) -> AppMemory:
    """Save memory or persist a partial conclusion before propagating failure."""
    try:
        return memory_store.save(memory, source_run_root)
    except AppMemoryStoreError:
        persist_discovery_memory_failure(
            workspace,
            request,
            agent_result,
            coverage,
            verification,
            durable_memory_id,
            procedure_replay,
        )
        raise


def persist_discovery_outcome(
    workspace: RunWorkspace,
    result: DiscoveryResult,
    require_reusable_evidence: bool,
) -> RunWorkflowOutcome:
    """Persist and classify one ordinary discovery conclusion."""
    persist_discovery_result(workspace, result)
    return RunWorkflowOutcome(
        classify_discovery_result(result, require_reusable_evidence),
        discovery_result_reason(result),
    )


def classify_discovery_result(
    result: DiscoveryResult,
    require_reusable_evidence: bool,
) -> RunTerminalCategory:
    """Translate discovery evidence into one run terminal category."""
    required = {
        objective.objective_id for objective in result.request.objectives if objective.required
    }
    supported_states = {
        entry.objective_id
        for entry in result.coverage.entries
        if entry.state is DiscoveryObjectiveState.SUPPORTED
    }
    if required.issubset(supported_states):
        return RunTerminalCategory.COMPLETED
    if (
        not require_reusable_evidence
        and result.terminal_reason is AgentTerminalReason.DONE_REQUESTED
        and result.verification is not None
        and result.verification.satisfied
    ):
        return RunTerminalCategory.COMPLETED
    return agent_terminal_category(
        result.terminal_reason,
        done_category=RunTerminalCategory.INCOMPLETE,
    )


def discovery_result_reason(result: DiscoveryResult) -> str:
    """Report task verification separately from reusable artifact coverage."""
    if (
        result.procedure_replay is not None
        and result.verification is not None
        and result.agent_result == result.procedure_replay.agent_result
    ):
        return (
            f"Compiled procedure replay ended as "
            f"{result.procedure_replay.terminal.value}. {result.verification.reason}"
        )
    task_summary = _task_verification_summary(result)
    procedure_summary = _compiled_procedure_summary(result)
    coverage_summary = _objective_coverage_summary(result)
    return " ".join((task_summary, procedure_summary, coverage_summary, result.agent_result.reason))


def _task_verification_summary(result: DiscoveryResult) -> str:
    if result.verification is None:
        return "Task verification was not available."
    condition_count = len(result.request.objectives[0].completion_conditions)
    if result.verification.satisfied:
        if condition_count == 0:
            return "Task completed: the objective was independently verified."
        return f"Task completed: {condition_count} of {condition_count} conditions verified."
    return f"Task verification failed: {result.verification.reason}"


def _compiled_procedure_summary(result: DiscoveryResult) -> str:
    if result.procedure_replay is not None:
        replay = result.procedure_replay
        return (
            f"Stored procedure replay ended as {replay.terminal.value} after "
            f"{replay.completed_step_count} of {replay.procedure_step_count} steps."
        )
    if result.compiled_procedure_id is None:
        return "No compiled procedure was promoted."
    return f"Reusable procedure compiled with {result.compiled_procedure_step_count} steps."


def _objective_coverage_summary(result: DiscoveryResult) -> str:
    supported_count = sum(
        entry.state is DiscoveryObjectiveState.SUPPORTED for entry in result.coverage.entries
    )
    total_count = len(result.coverage.entries)
    return (
        "Reusable objective coverage is a separate metric: "
        f"{supported_count} of {total_count} objectives supported."
    )
