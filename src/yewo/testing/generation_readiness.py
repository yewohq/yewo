"""Fail-closed authority for semantic scenario generation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from yewo.testing.generation_evidence import ScenarioGenerationEvidence
from yewo.testing.models import UserJourney


class GenerationReadinessState(str, Enum):
    """The only startup states that may precede semantic generation."""

    READY = "ready"
    INSUFFICIENT_CONTRACT = "insufficient_contract"
    DISCOVERY_REQUIRED = "discovery_required"
    CAPABILITY_UNOBSERVED = "capability_unobserved"


@dataclass(frozen=True)
class GenerationReadiness:
    """A startup decision and the evidence that authorizes it."""

    state: GenerationReadinessState
    reason: str
    evidence: ScenarioGenerationEvidence | None = None

    def __post_init__(self) -> None:
        has_evidence = self.evidence is not None
        if (self.state is GenerationReadinessState.READY) != has_evidence:
            raise ValueError("Only READY generation readiness may carry app evidence.")

    @property
    def is_ready(self) -> bool:
        return self.state is GenerationReadinessState.READY


def assess_generation_readiness(
    journey: UserJourney,
    evidence: ScenarioGenerationEvidence | None,
    *,
    discovery_completed: bool = False,
) -> GenerationReadiness:
    """Decide whether verified app evidence authorizes scenario generation."""
    if not journey.description.strip():
        return GenerationReadiness(
            GenerationReadinessState.INSUFFICIENT_CONTRACT,
            "The journey description is empty.",
        )
    if not journey.coverage:
        return GenerationReadiness(
            GenerationReadinessState.INSUFFICIENT_CONTRACT,
            "The journey declares no coverage cells.",
        )
    if evidence is None:
        state = (
            GenerationReadinessState.CAPABILITY_UNOBSERVED
            if discovery_completed
            else GenerationReadinessState.DISCOVERY_REQUIRED
        )
        reason = (
            "Bounded discovery produced no verified scenario-generation evidence."
            if discovery_completed
            else "No compatible verified app knowledge is available."
        )
        return GenerationReadiness(state, reason)

    reasons = [
        f"{item.intent} ({item.state.value})"
        for item in evidence.unavailable_required_capabilities
    ]
    if not evidence.available_capability_ids:
        reasons.append("the AppMap contains no observed capabilities")
    if not evidence.route_prefixes:
        reasons.append("no verified RouteMemory reaches a test subject")
    if reasons:
        state = (
            GenerationReadinessState.CAPABILITY_UNOBSERVED
            if discovery_completed
            else GenerationReadinessState.DISCOVERY_REQUIRED
        )
        return GenerationReadiness(
            state,
            "Required app evidence is unavailable: " + "; ".join(reasons) + ".",
        )
    return GenerationReadiness(
        GenerationReadinessState.READY,
        "Verified app evidence authorizes semantic scenario generation.",
        evidence,
    )
