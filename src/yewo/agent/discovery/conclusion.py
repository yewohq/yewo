"""Discovery evidence persistence and terminal classification."""

from __future__ import annotations

from pathlib import Path

from yewo.agent.app_memory import AppMemory
from yewo.agent.memory_store import AppMemoryStore, AppMemoryStoreError
from yewo.agent.runtime_records import AgentResult, AgentTerminalReason, agent_terminal_category
from yewo.agent.replay_evidence import (
    ReplayEvidenceSnapshot,
    persist_initial_replay_evidence,
    persist_replay_evidence,
)
from yewo.agent.route_replay import RouteReplayResult
from yewo.artifacts import ARTIFACTS_DIRECTORY, AtomicArtifactWriter, ArtifactWriteError
from yewo.runs import RunTerminalCategory, RunWorkflowOutcome, RunWorkspace
from yewo.agent.discovery.narration import narrate_discovery_result
from yewo.agent.discovery.records import (
    DiscoveryCoverage,
    DiscoveryObjectiveState,
    DiscoveryRequest,
    DiscoveryResult,
)
from yewo.agent.goal_verifier import GoalVerification


class DiscoveryEvidenceError(RuntimeError):
    """The discovery workflow could not persist its canonical conclusion."""


def persist_discovery_replay_attempt(
    workspace: RunWorkspace,
    replay: RouteReplayResult,
) -> None:
    """Persist replay proof before repair or interruption can end the run."""
    try:
        persist_initial_replay_evidence(workspace.root, workspace.start.run_id, replay)
    except ArtifactWriteError as error:
        raise DiscoveryEvidenceError("Could not persist the replay attempt.") from error


def persist_discovery_result(
    workspace: RunWorkspace,
    result: DiscoveryResult,
    replay_evidence_snapshot: ReplayEvidenceSnapshot | None = None,
) -> None:
    """Persist and narrate one canonical discovery result."""
    relative_path = ARTIFACTS_DIRECTORY / "agent" / "discovery" / "result.json"
    content = result.model_dump_json(indent=2).encode("utf-8") + b"\n"
    try:
        AtomicArtifactWriter(workspace.root).write(relative_path, content)
        persist_replay_evidence(
            workspace.root,
            workspace.start.run_id,
            result,
            replay_evidence_snapshot,
        )
        narrate_discovery_result(result)
    except ArtifactWriteError as error:
        raise DiscoveryEvidenceError("Could not persist the discovery result.") from error


def persist_discovery_memory_failure(
    workspace: RunWorkspace,
    request: DiscoveryRequest,
    agent_result: AgentResult,
    coverage: DiscoveryCoverage,
    replay: RouteReplayResult | None,
    verification: GoalVerification | None,
    durable_memory_id: str | None,
    replay_evidence_snapshot: ReplayEvidenceSnapshot | None = None,
) -> None:
    """Retain the run conclusion before propagating a memory-store failure."""
    unresolved = DiscoveryCoverage(
        entries=tuple(
            entry.model_copy(
                update={
                    "state": DiscoveryObjectiveState.UNRESOLVED,
                    "capability_id": None,
                    "route_id": None,
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
            replay=replay,
            verification=verification,
            memory_id=durable_memory_id,
            memory_persistence_error="Yewo could not update app memory.",
        ),
        replay_evidence_snapshot,
    )


def save_discovery_memory(
    memory_store: AppMemoryStore,
    memory: AppMemory,
    source_run_root: Path,
    workspace: RunWorkspace,
    request: DiscoveryRequest,
    agent_result: AgentResult,
    coverage: DiscoveryCoverage,
    replay: RouteReplayResult | None,
    verification: GoalVerification | None,
    durable_memory_id: str | None,
    replay_evidence_snapshot: ReplayEvidenceSnapshot | None = None,
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
            replay,
            verification,
            durable_memory_id,
            replay_evidence_snapshot,
        )
        raise


def persist_discovery_outcome(
    workspace: RunWorkspace,
    result: DiscoveryResult,
    require_reusable_evidence: bool,
    replay_evidence_snapshot: ReplayEvidenceSnapshot | None = None,
) -> RunWorkflowOutcome:
    """Persist and classify one ordinary discovery conclusion."""
    persist_discovery_result(workspace, result, replay_evidence_snapshot)
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
    supported = {
        entry.objective_id
        for entry in result.coverage.entries
        if entry.state is DiscoveryObjectiveState.SUPPORTED
    }
    if required.issubset(supported):
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
    """Summarize supported objectives and the underlying agent terminal."""
    supported_count = sum(
        entry.state is DiscoveryObjectiveState.SUPPORTED for entry in result.coverage.entries
    )
    return (
        f"Discovery supported {supported_count} of {len(result.coverage.entries)} "
        f"requested objectives. {result.agent_result.reason}"
    )
