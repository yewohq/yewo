"""Tests for the deterministic discovery-to-testing bridge."""

from datetime import datetime, timezone

from pydantic import ValidationError
import pytest

from yewo.agent import AgentTerminalReason
from yewo.agent.actuation import ActionKind
from yewo.agent.discovery import (
    DiscoveryCoverage,
    DiscoveryCoverageEntry,
    DiscoveryObjective,
    DiscoveryObjectiveState,
    DiscoveryRequest,
    DiscoveryResult,
)
from yewo.agent.memory.knowledge import (
    ActionRiskKind,
    ActionRiskKnowledge,
    AppKnowledge,
    KnowledgeEvidenceReference,
)
from yewo.agent.memory.records import (
    AppMemory,
    LogicalScreen,
    MemoryCompatibility,
    ObservationVariant,
)
from yewo.agent.planning.goal_verifier import GoalVerification
from yewo.agent.replay.reconciliation.compiler import compile_replay_procedure
from yewo.agent.replay.reconciliation.records import ReplayProcedure
from yewo.project_config import (
    CoverageCategory,
    CoverageCell,
    CoveragePriority,
    ResolvedJourney,
    ScenarioPreconditionsSpec,
    ScenarioType,
)
from yewo.scenarios import (
    DiscoveryTestingBridgeError,
    RequirementEvidenceState,
    ScenarioSurfaceKind,
    build_discovery_testing_bridge,
)

from agent.discovery.support import agent_result
from agent.replay.support import evidence as discovery_evidence
from agent.replay.support import observations, response


def _procedure():
    compiled = compile_replay_procedure(discovery_evidence(), response())
    assert isinstance(compiled, ReplayProcedure)
    return compiled


def _journey() -> ResolvedJourney:
    return ResolvedJourney(
        journey_id="objective_1",
        name="Create an item",
        objective="Create and save one named item.",
        coverage=(
            CoverageCell(
                category=CoverageCategory.FUNCTIONAL,
                scenario_type=ScenarioType.HAPPY_PATH,
                priority=CoveragePriority.P0,
                index=0,
            ),
        ),
        preconditions=ScenarioPreconditionsSpec(app_state="clear_data"),
    )


def _memory(*, knowledge: AppKnowledge | None = None) -> AppMemory:
    evidence = discovery_evidence()
    screens = tuple(
        LogicalScreen(
            screen_id=state.state_id,
            variants=(
                ObservationVariant(
                    observation_id=state.representative_observation_id,
                    source_run_id=evidence.run.run_id,
                    presentation=state.representative_presentation,
                    screenshot=state.representative_screenshot,
                ),
            ),
        )
        for state in evidence.states.values()
    )
    return AppMemory(
        memory_id="memory_1",
        compatibility=MemoryCompatibility(application=evidence.run.application),
        source_run_ids=(evidence.run.run_id,),
        screens=screens,
        app_knowledge=knowledge or AppKnowledge(),
        updated_at=datetime(2026, 11, 6, tzinfo=timezone.utc),
    )


def _result(procedure: ReplayProcedure | None = None) -> DiscoveryResult:
    compiled = procedure or _procedure()
    observed = observations()
    result = agent_result(
        observed[0],
        observed[-1],
        terminal=AgentTerminalReason.DONE_REQUESTED,
    )
    request = DiscoveryRequest(
        request_id="discovery_request",
        objectives=(
            DiscoveryObjective(
                objective_id="objective_1",
                statement="Create and save one named item.",
                completion_conditions=("The named item is saved.",),
            ),
        ),
    )
    return DiscoveryResult(
        request=request,
        terminal_reason=AgentTerminalReason.DONE_REQUESTED,
        coverage=DiscoveryCoverage(
            entries=(
                DiscoveryCoverageEntry(
                    objective_id="objective_1",
                    state=DiscoveryObjectiveState.SUPPORTED,
                    procedure_id=compiled.procedure_id,
                    evidence_observation_ids=(observed[-1].observation_id,),
                    reason="The compiled procedure supports the objective.",
                ),
            )
        ),
        agent_result=result,
        verification=GoalVerification(
            verification_id="verification_1",
            objective_id="objective_1",
            final_observation_id=observed[-1].observation_id,
            satisfied=True,
            reason="The final observation shows the saved item.",
            observed_app_state="The saved item is visible.",
        ),
        compiled_procedure_id=compiled.procedure_id,
        compiled_procedure_step_count=len(compiled.steps),
        memory_id="memory_1",
    )


def _bridge(
    *,
    procedure: ReplayProcedure | None = None,
    result: DiscoveryResult | None = None,
    memory: AppMemory | None = None,
):
    compiled = procedure or _procedure()
    evidence = discovery_evidence()
    return build_discovery_testing_bridge(
        _journey(),
        evidence.run.application,
        evidence.run.run_id,
        result or _result(compiled),
        procedure=compiled,
        evidence=evidence,
        memory=memory or _memory(),
    )


def test_bridge_builds_ready_surfaces_from_accepted_discovery() -> None:
    suite = _bridge()

    assert suite.readiness.executable
    assert suite.readiness.independently_auditable
    assert suite.readiness.required_setup.app_state == "clear_data"
    assert [item.state for item in suite.readiness.requirements] == [
        RequirementEvidenceState.SUPPORTED,
        RequirementEvidenceState.SUPPORTED,
    ]
    assert {surface.kind for surface in suite.surfaces} == {
        ScenarioSurfaceKind.INPUT,
        ScenarioSurfaceKind.OPERATION,
        ScenarioSurfaceKind.PERSISTENCE_BOUNDARY,
        ScenarioSurfaceKind.PRESENTATION,
    }
    assert all(surface.provenance.verification_id == "verification_1" for surface in suite.surfaces)
    presentation_surfaces = tuple(
        surface for surface in suite.surfaces if surface.kind is ScenarioSurfaceKind.PRESENTATION
    )
    assert presentation_surfaces
    assert all(surface.state_ids for surface in presentation_surfaces)
    assert all(surface.provenance.observation_ids for surface in presentation_surfaces)


def test_bridge_uses_observed_outcomes_not_planner_predictions() -> None:
    suite = _bridge()
    serialized = suite.model_dump_json()

    assert "The accepted action produced the observed" in serialized
    assert "the expected screen appears" not in serialized


def test_bridge_identifiers_are_stable_for_the_same_evidence() -> None:
    procedure = _procedure()
    result = _result(procedure)
    first = _bridge(procedure=procedure, result=result)
    second = _bridge(procedure=procedure, result=result)

    assert first.suite_id == second.suite_id
    assert first.readiness.readiness_id == second.readiness.readiness_id
    assert [item.surface_id for item in first.surfaces] == [
        item.surface_id for item in second.surfaces
    ]


def test_incomplete_discovery_returns_unready_without_surfaces() -> None:
    observed = observations()
    incomplete = _result().model_copy(
        update={
            "coverage": DiscoveryCoverage(
                entries=(
                    DiscoveryCoverageEntry(
                        objective_id="objective_1",
                        state=DiscoveryObjectiveState.UNRESOLVED,
                        reason="The discovery budget ended.",
                    ),
                )
            ),
            "verification": None,
            "compiled_procedure_id": None,
            "compiled_procedure_step_count": None,
            "memory_id": None,
            "agent_result": agent_result(
                observed[0],
                observed[-1],
                terminal=AgentTerminalReason.BUDGET_EXHAUSTED,
            ),
            "terminal_reason": AgentTerminalReason.BUDGET_EXHAUSTED,
        }
    )
    evidence = discovery_evidence()

    suite = build_discovery_testing_bridge(
        _journey(),
        evidence.run.application,
        evidence.run.run_id,
        incomplete,
    )

    assert not suite.readiness.executable
    assert not suite.readiness.independently_auditable
    assert suite.surfaces == ()
    assert "discovery budget ended" in suite.readiness.reason.lower()


def test_named_but_unavailable_discovery_artifacts_return_unready() -> None:
    evidence = discovery_evidence()

    suite = build_discovery_testing_bridge(
        _journey(),
        evidence.run.application,
        evidence.run.run_id,
        _result(),
    )

    assert not suite.readiness.executable
    assert "compiled procedure" in suite.readiness.reason
    assert "app memory" in suite.readiness.reason


def test_missing_completion_milestone_prevents_readiness() -> None:
    procedure = _procedure()
    steps = tuple(
        step.model_copy(update={"captures_completion_conditions": ()}) for step in procedure.steps
    )
    without_milestone = procedure.model_copy(update={"steps": steps})

    suite = _bridge(procedure=without_milestone, result=_result(without_milestone))

    assert not suite.readiness.executable
    assert suite.readiness.requirements[1].state is RequirementEvidenceState.UNRESOLVED
    assert "lacks compiled milestone evidence" in suite.readiness.reason


def test_compiler_may_add_evidence_backed_completion_conditions() -> None:
    procedure = _procedure()
    result = _result(procedure).model_copy(
        update={
            "request": DiscoveryRequest(
                request_id="discovery_request",
                objectives=(
                    DiscoveryObjective(
                        objective_id="objective_1",
                        statement="Create and save one named item.",
                    ),
                ),
            )
        }
    )

    suite = _bridge(procedure=procedure, result=result)

    assert suite.readiness.executable
    assert [item.statement for item in suite.readiness.requirements] == [
        "Create and save one named item.",
        "The named item is saved.",
    ]


def test_compiler_cannot_omit_a_requested_completion_condition() -> None:
    procedure = _procedure()
    result = _result(procedure)
    evidence = discovery_evidence().model_copy(
        update={"task": discovery_evidence().task.model_copy(update={"completion_conditions": ()})}
    )
    suite = build_discovery_testing_bridge(
        _journey(),
        evidence.run.application,
        evidence.run.run_id,
        result,
        procedure=procedure,
        evidence=evidence,
        memory=_memory(),
    )

    assert not suite.readiness.executable
    assert "omitted a requested completion condition" in suite.readiness.reason


def test_compiler_added_condition_needs_observable_milestone_evidence() -> None:
    procedure = _procedure()
    without_milestones = procedure.model_copy(
        update={
            "steps": tuple(
                step.model_copy(update={"captures_completion_conditions": ()})
                for step in procedure.steps
            )
        }
    )
    result = _result(without_milestones).model_copy(
        update={
            "request": DiscoveryRequest(
                request_id="discovery_request",
                objectives=(
                    DiscoveryObjective(
                        objective_id="objective_1",
                        statement="Create and save one named item.",
                    ),
                ),
            )
        }
    )

    suite = _bridge(procedure=without_milestones, result=result)

    assert not suite.readiness.executable
    assert suite.readiness.requirements[1].state is RequirementEvidenceState.UNRESOLVED
    assert "lacks compiled milestone evidence" in suite.readiness.reason


def test_procedure_without_action_surfaces_retains_observed_presentations() -> None:
    procedure = _procedure()
    mechanics_only = procedure.model_copy(
        update={
            "steps": tuple(
                step.model_copy(
                    update={
                        "proposal": step.proposal.model_copy(update={"action": ActionKind.WAIT})
                    }
                )
                for step in procedure.steps
            )
        }
    )

    suite = _bridge(procedure=mechanics_only, result=_result(mechanics_only))

    assert suite.readiness.executable
    assert {surface.kind for surface in suite.surfaces} == {ScenarioSurfaceKind.PRESENTATION}


def test_known_action_risk_blocks_readiness() -> None:
    risk = ActionRiskKnowledge(
        action_risk_id="risk_1",
        screen_id="state_001",
        action=ActionKind.TAP,
        target_description="the first control",
        kind=ActionRiskKind.CRASH,
        seen_count=1,
        sources=(KnowledgeEvidenceReference(run_id="run_1", step_id="step_1"),),
    )

    suite = _bridge(memory=_memory(knowledge=AppKnowledge(action_risks=(risk,))))

    assert not suite.readiness.executable
    assert suite.readiness.blocking_action_risk_ids == ("risk_1",)
    assert suite.surfaces == ()


def test_bridge_rejects_evidence_from_another_application_build() -> None:
    evidence = discovery_evidence()
    other_application = evidence.run.application.model_copy(update={"build": "other"})

    with pytest.raises(DiscoveryTestingBridgeError, match="different application build"):
        build_discovery_testing_bridge(
            _journey(),
            other_application,
            evidence.run.run_id,
            _result(),
            procedure=_procedure(),
            evidence=evidence,
            memory=_memory(),
        )


def test_bridge_result_is_strictly_validated() -> None:
    suite = _bridge()

    with pytest.raises(ValidationError):
        suite.model_copy(update={"journey_id": "other"}).model_validate(
            {**suite.model_dump(mode="python"), "unknown": True}
        )
