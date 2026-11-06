"""Compiled-action settlement across deterministic environment recovery."""

from typing import cast

from yewo.agent.actuation import ActionKind, ActionProposal
from yewo.agent.planning.evidence import PlannerDecisionMetadata, PlannerDecisionSource
from yewo.agent.planning.history import PlanningHistoryEntry
from yewo.agent.planning.records import PlanningRequest
from yewo.agent.replay.procedure.records import ProcedureRepairTarget
from yewo.agent.replay.procedure.repair import BoundedProcedureRepairActionSource
from yewo.agent.replay.procedure.source import CompiledProcedureActionSource
from yewo.agent.replay.reconciliation.compiler import compile_replay_procedure
from yewo.agent.replay.reconciliation.records import ReplayProcedure
from yewo.grounding import GroundingRequest
from yewo.observations import Observation

from agent.discovery.support import agent_result
from agent.replay.support import observations as _observations, response as _response
from agent.replay.test_procedure_environment_compilation import _permission_evidence
from agent.replay.test_procedure_repair import (
    RepairPlanner,
    _outcome,
    _repair_source,
    _request,
)


class ExpectedSurfaceDivergenceSource:
    """Expose a divergence while the next compiled step expects a system surface."""

    procedure_id = "procedure_environment"
    application_id = "com.example.testing_app"
    completed_step_count = 0
    failed_procedure_index = 1
    milestones = ()
    next_procedure_index = 1
    replay_complete = False
    divergence = None
    divergences = ()

    def __init__(self) -> None:
        self.repair_target = ProcedureRepairTarget(
            expected_state_id="expected-app-state",
            resume_procedure_index=1,
            presentation=_observations()[0].presentation_key,
            reason="The environment-step expectation diverged.",
        )

    def required_model_requests(self, request: PlanningRequest) -> int:
        del request
        return 0

    def plan(self, request: PlanningRequest) -> ActionProposal:
        return ActionProposal(
            proposal_id="environment-divergence",
            source_observation_id=request.observation.observation_id,
            action=ActionKind.CANNOT_PROGRESS,
            reason="The environment-step expectation diverged.",
        )

    def decision_metadata(self) -> PlannerDecisionMetadata:
        return PlannerDecisionMetadata(
            source=PlannerDecisionSource.COMPILED_PROCEDURE,
            reused_flow_id=self.procedure_id,
        )

    def accepts_expected_system_surface(self, observation: Observation) -> bool:
        del observation
        return True

    def matches_repair_target(self, observation: Observation) -> bool:
        del observation
        return False

    def rejoin(self, observation: Observation) -> bool:
        del observation
        return False

    def grounding_match(self, request: GroundingRequest) -> None:
        del request
        return None

    def repair_would_repeat_failed_action(self, proposal: ActionProposal) -> bool:
        del proposal
        return False


def test_system_surface_settles_compiled_action() -> None:
    source, planner = _repair_source()
    before = _observations()[0]
    destination = _observations()[2]
    compiled_action = source.plan(_request(before))
    overlay = destination.model_copy(
        update={
            "observation_id": "system_overlay_after_action",
            "foreground_application_id": "com.android.systemui",
            "platform_context": "notification_shade",
        }
    )
    compiled_feedback = _outcome(compiled_action, before, overlay)

    environment_action = source.plan(_request(overlay, (compiled_feedback,)))
    assert compiled_feedback.outcome is not None
    environment_feedback = PlanningHistoryEntry(
        proposal=environment_action,
        outcome=compiled_feedback.outcome.model_copy(
            update={
                "action": ActionKind.BACK,
                "source_observation_id": overlay.observation_id,
                "settled_observation_id": destination.observation_id,
            }
        ),
    )
    resumed = source.plan(_request(destination, (compiled_feedback, environment_feedback)))

    assert environment_action.action is ActionKind.BACK
    assert resumed.action is ActionKind.TYPE
    assert source.completed_step_count == 1
    assert planner.request_count == 0
    assert source.divergence is None


def test_permission_surface_replays_compiled_choice() -> None:
    evidence = _permission_evidence()
    procedure = compile_replay_procedure(evidence, _response())
    assert isinstance(procedure, ReplayProcedure)
    source = CompiledProcedureActionSource(procedure, evidence)
    repair = BoundedProcedureRepairActionSource(source, RepairPlanner())
    before = _observations()[0]
    permission = _observations()[2].model_copy(
        update={
            "foreground_application_id": "com.google.android.permissioncontroller",
            "platform_context": (
                "com.android.permissioncontroller.permission.ui.GrantPermissionsActivity"
            ),
        }
    )
    trigger = repair.plan(_request(before))
    trigger_outcome = agent_result(
        before,
        permission,
        action=ActionKind.TAP,
    ).steps[0].outcome
    assert trigger_outcome is not None

    permission_action = repair.plan(
        _request(
            permission,
            (PlanningHistoryEntry(proposal=trigger, outcome=trigger_outcome),),
        )
    )

    assert permission_action.action is ActionKind.TAP
    assert permission_action.target_expression == procedure.steps[1].proposal.target_expression
    assert repair.environment_events == ()


def test_expected_system_surface_uses_no_model_call() -> None:
    permission = _observations()[2].model_copy(
        update={
            "foreground_application_id": "com.google.android.permissioncontroller",
            "platform_context": (
                "com.android.permissioncontroller.permission.ui.GrantPermissionsActivity"
            ),
        }
    )
    planner = RepairPlanner()
    procedure_source = cast(
        CompiledProcedureActionSource,
        ExpectedSurfaceDivergenceSource(),
    )
    source = BoundedProcedureRepairActionSource(procedure_source, planner)

    boundary = source.plan(_request(permission))
    assert boundary.action is ActionKind.WAIT
    assert source.required_model_requests(_request(permission)) == 0
    environment_action = source.plan(_request(permission))

    assert environment_action.action is ActionKind.BACK
    assert planner.request_count == 0
    assert source.last_planning_model_request_count() == 0
