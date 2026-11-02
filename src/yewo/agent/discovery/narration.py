"""Human-readable and diagnostic narration for discovery."""

from __future__ import annotations

from yewo.agent.app_memory import VerifiedRoute
from yewo.agent.discovery_checkpoint import DiscoveryCheckpoint
from yewo.agent.discovery.records import DiscoveryRequest, DiscoveryResult
from yewo.agent.route_replay import RouteReplayResult, RouteReplayTerminal
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


def narrate_route_selection(
    route: VerifiedRoute | None,
    checkpoint: DiscoveryCheckpoint | None = None,
) -> None:
    """Record whether discovery can attempt a stored route."""
    narrator = current_narrator()
    if narrator is None:
        return
    if route is None and checkpoint is None:
        narrator.record(_SOURCE, "No compatible verified route is available. Exploring the app.")
        narrator.trace(
            _SOURCE,
            "flow_lookup=miss reason=no_compatible_verified_route_or_checkpoint",
        )
        return
    if checkpoint is not None:
        narrator.record(_SOURCE, "Resuming a compatible incomplete discovery path.")
        narrator.trace(
            _SOURCE,
            f"flow_lookup=hit flow_kind=discovery_checkpoint "
            f"flow={checkpoint.checkpoint_id} "
            f"transitions={len(checkpoint.transition_ids)}",
        )
        return
    if route is None:
        raise ValueError("Discovery replay selection is inconsistent.")
    narrator.record(_SOURCE, "Replaying a compatible verified route before exploration.")
    narrator.trace(
        _SOURCE,
        f"flow_lookup=hit flow_kind=verified_route flow={route.route_id} "
        f"segments={len(route.segments)} "
        f"evidence_observations={len(route.evidence_observation_ids)}",
    )


def narrate_replay_handoff(replay: RouteReplayResult) -> None:
    """Record why stored-flow execution returned control to the planner."""
    narrator = current_narrator()
    if narrator is None:
        return
    if replay.terminal is RouteReplayTerminal.DIVERGED:
        segment = replay.failed_segment_index if replay.failed_segment_index is not None else "unknown"
        narrator.record(
            _SOURCE,
            f"Stored flow was invalidated at segment {segment}. Continuing with the planner.",
        )
        narrator.trace(
            _SOURCE,
            f"flow={replay.path_id} flow_kind={replay.path_kind.value} "
            f"reuse=invalidated failed_segment={segment} "
            f"reason={replay.reason}",
        )
        return
    narrator.record(
        _SOURCE,
        "The retained discovery checkpoint was reached. Remaining work requires planning.",
    )
    narrator.trace(
        _SOURCE,
        f"flow={replay.path_id} flow_kind={replay.path_kind.value} "
        f"reuse=completed completed_segments="
        f"{replay.completed_segment_count}",
    )


def narrate_discovery_result(result: DiscoveryResult) -> None:
    """Record coverage, replay, verification, and memory conclusions."""
    narrator = current_narrator()
    if narrator is None:
        return
    supported = sum(entry.state.value == "supported" for entry in result.coverage.entries)
    narrator.record(
        _SOURCE,
        f"Discovery finished with {supported} of {len(result.coverage.entries)} "
        f"objective(s) supported.",
    )
    narrator.trace(
        _SOURCE,
        f"request={result.request.request_id} terminal={result.terminal_reason.value} "
        f"supported={supported} total={len(result.coverage.entries)} "
        f"memory={result.memory_id or 'none'} "
        f"replay={result.replay.terminal.value if result.replay is not None else 'none'} "
        f"verification={result.verification.verification_id if result.verification is not None else 'none'}",
    )
    for entry in result.coverage.entries:
        narrator.trace(
            _SOURCE,
            f"objective={entry.objective_id} state={entry.state.value} "
            f"capability={entry.capability_id or 'none'} route={entry.route_id or 'none'} "
            f"evidence_observations={len(entry.evidence_observation_ids)}",
        )
    if result.verification is not None:
        verdict = "satisfied" if result.verification.satisfied else "not satisfied"
        narrator.record(
            _SOURCE,
            f"Independent goal verification {verdict}: {result.verification.reason}",
        )
