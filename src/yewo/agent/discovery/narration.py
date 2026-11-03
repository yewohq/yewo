"""Human-readable and diagnostic narration for discovery."""

from __future__ import annotations

from yewo.agent.discovery.records import DiscoveryRequest, DiscoveryResult
from yewo.agent.replay.procedure_records import ProcedureReplayTerminal
from yewo.run_narration import current_narrator


_SOURCE = "agent-discovery"


def narrate_discovery_start(request: DiscoveryRequest) -> None:
    """Record the start of one bounded discovery request."""
    narrator = current_narrator()
    if narrator is None:
        return
    narrator.record(
        _SOURCE,
        f"Starting discovery for {len(request.objectives)} objective(s).",
    )
    narrator.trace(
        _SOURCE,
        f"request={request.request_id} objectives={len(request.objectives)} "
        f"priorities={len(request.priorities)} declared_values={len(request.declared_value_names)}",
    )


def narrate_no_compiled_procedure() -> None:
    """Record that no compatible compiled procedure is available."""
    narrator = current_narrator()
    if narrator is None:
        return
    narrator.record(_SOURCE, "No compatible compiled procedure is available. Exploring the app.")
    narrator.trace(_SOURCE, "flow_lookup=miss reason=no_compatible_compiled_procedure")


def narrate_compiled_procedure_selection(procedure_id: str, step_count: int) -> None:
    """Record that discovery selected guarded compiled-procedure replay."""
    narrator = current_narrator()
    if narrator is None:
        return
    narrator.record(_SOURCE, "Replaying a compatible compiled procedure.")
    narrator.trace(
        _SOURCE,
        f"flow_lookup=hit flow_kind=compiled_procedure flow={procedure_id} "
        f"steps={step_count}",
    )


def narrate_discovery_result(result: DiscoveryResult) -> None:
    """Record coverage, replay, verification, and memory conclusions."""
    narrator = current_narrator()
    if narrator is None:
        return
    supported = sum(entry.state.value == "supported" for entry in result.coverage.entries)
    verification_state = (
        "satisfied"
        if result.verification is not None and result.verification.satisfied
        else "not_satisfied"
        if result.verification is not None
        else "unavailable"
    )
    procedure_reusable = result.compiled_procedure_id is not None or (
        result.procedure_replay is not None
        and result.procedure_replay.terminal
        in {
            ProcedureReplayTerminal.COMPLETED,
            ProcedureReplayTerminal.REPLAY_COMPLETE,
        }
    )
    if result.procedure_replay is not None:
        narrator.record(
            _SOURCE,
            f"Compiled procedure replay finished as {result.procedure_replay.terminal.value}. "
            f"Task verification was {verification_state.replace('_', ' ')}. "
            f"Compiled-procedure reuse is "
            f"{'available' if procedure_reusable else 'not available from this attempt'}. "
            f"Reusable objective coverage is separate: {supported} of "
            f"{len(result.coverage.entries)} objective(s) supported.",
        )
    elif result.compiled_procedure_id is not None:
        narrator.record(
            _SOURCE,
            "Discovery compiled a reusable procedure. "
            f"Task verification was {verification_state.replace('_', ' ')}. "
            f"Reusable objective coverage is separate: {supported} of "
            f"{len(result.coverage.entries)} objective(s) supported.",
        )
    else:
        narrator.record(
            _SOURCE,
            f"Discovery finished with {supported} of {len(result.coverage.entries)} "
            f"objective(s) supported.",
        )
    narrator.trace(
        _SOURCE,
        f"request={result.request.request_id} terminal={result.terminal_reason.value} "
        f"task_verification={verification_state} "
        f"compiled_procedure_reusable="
        f"{str(procedure_reusable).lower()} "
        f"objectives_supported={supported} total={len(result.coverage.entries)} "
        f"memory={result.memory_id or 'none'} "
        f"procedure_replay="
        f"{result.procedure_replay.terminal.value if result.procedure_replay is not None else 'none'} "
        f"verification={result.verification.verification_id if result.verification is not None else 'none'}",
    )
    for entry in result.coverage.entries:
        narrator.trace(
            _SOURCE,
            f"objective={entry.objective_id} coverage_state={entry.state.value} "
            f"procedure={entry.procedure_id or 'none'} "
            f"evidence_observations={len(entry.evidence_observation_ids)}",
        )
    if result.verification is not None:
        verdict = "satisfied" if result.verification.satisfied else "not satisfied"
        narrator.record(
            _SOURCE,
            f"Independent goal verification {verdict}: {result.verification.reason}",
        )
