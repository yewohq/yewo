"""Human-readable and diagnostic narration for discovery."""

from __future__ import annotations

from yewo.agent.app_memory import VerifiedRoute
from yewo.agent.discovery.records import DiscoveryRequest, DiscoveryResult
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


def narrate_route_selection(route: VerifiedRoute | None) -> None:
    """Record whether discovery can attempt a stored route."""
    narrator = current_narrator()
    if narrator is None:
        return
    if route is None:
        narrator.record(_SOURCE, "No compatible verified route is available. Exploring the app.")
        narrator.trace(_SOURCE, "stored_route=none")
        return
    narrator.record(_SOURCE, "Replaying a compatible verified route before exploration.")
    narrator.trace(
        _SOURCE,
        f"stored_route={route.route_id} segments={len(route.segments)} "
        f"evidence_observations={len(route.evidence_observation_ids)}",
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
