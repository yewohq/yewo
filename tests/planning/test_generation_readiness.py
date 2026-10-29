"""Missing app evidence is a named state, never an off-switch for generation."""

import pytest
from generation_fakes import scenario_generation_evidence

from yewo.discovery.contracts import CapabilityState
from yewo.testing.generation_readiness import (
    GenerationReadiness,
    GenerationReadinessState,
    assess_generation_readiness,
)
from yewo.testing.models import UserJourney
from yewo.testing.taxonomy import Category, CoverageCell, Priority, ScenarioType


def _journey(**overrides) -> UserJourney:
    fields = {
        "id": "trainer",
        "name": "Trainer",
        "description": "A coach edits a schedule.",
        "coverage": [CoverageCell(Category.FUNCTIONAL, ScenarioType.HAPPY_PATH, Priority.P0)],
    }
    return UserJourney(**{**fields, **overrides})


def test_verified_evidence_authorizes_generation():
    evidence = scenario_generation_evidence()

    readiness = assess_generation_readiness(_journey(), evidence)

    assert readiness.state is GenerationReadinessState.READY
    assert readiness.is_ready
    assert readiness.evidence is evidence


def test_an_unstated_journey_is_an_insufficient_contract():
    readiness = assess_generation_readiness(_journey(description="  "), None)

    assert readiness.state is GenerationReadinessState.INSUFFICIENT_CONTRACT
    assert readiness.evidence is None


def test_a_journey_declaring_no_cell_is_an_insufficient_contract():
    readiness = assess_generation_readiness(_journey(coverage=[]), None)

    assert readiness.state is GenerationReadinessState.INSUFFICIENT_CONTRACT
    assert "coverage" in readiness.reason


def test_absent_evidence_asks_for_discovery_rather_than_generating():
    readiness = assess_generation_readiness(_journey(), None)

    assert readiness.state is GenerationReadinessState.DISCOVERY_REQUIRED
    assert not readiness.is_ready


def test_absent_evidence_after_discovery_names_the_unobserved_capability():
    readiness = assess_generation_readiness(_journey(), None, discovery_completed=True)

    assert readiness.state is GenerationReadinessState.CAPABILITY_UNOBSERVED


def test_a_required_capability_that_discovery_could_not_reach_blocks_generation():
    evidence = scenario_generation_evidence()
    blocked = evidence.model_copy(
        update={
            "capabilities": (
                evidence.capabilities[0].model_copy(
                    update={"state": CapabilityState.BLOCKED_BY_SETUP}
                ),
            )
        }
    )

    readiness = assess_generation_readiness(_journey(), blocked, discovery_completed=True)

    assert readiness.state is GenerationReadinessState.CAPABILITY_UNOBSERVED
    assert "reach the test subject" in readiness.reason
    assert "no observed capabilities" in readiness.reason


def test_evidence_without_a_verified_route_blocks_generation():
    evidence = scenario_generation_evidence().model_copy(update={"route_prefixes": ()})

    readiness = assess_generation_readiness(_journey(), evidence)

    assert readiness.state is GenerationReadinessState.DISCOVERY_REQUIRED
    assert "RouteMemory" in readiness.reason


def test_a_state_that_is_not_ready_cannot_carry_evidence():
    with pytest.raises(ValueError, match="Only READY"):
        GenerationReadiness(
            GenerationReadinessState.DISCOVERY_REQUIRED,
            "A caller tried to smuggle evidence past the decision.",
            scenario_generation_evidence(),
        )


def test_a_ready_state_cannot_be_built_without_evidence():
    with pytest.raises(ValueError, match="Only READY"):
        GenerationReadiness(GenerationReadinessState.READY, "No evidence was selected.")
