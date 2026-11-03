"""Bound repeated planner repairs to one compiled-procedure action allowance."""

from collections.abc import Callable

from yewo.agent.actuation import ActionKind, ActionProposal
from yewo.agent.planning.evidence import (
    PlannerDecisionMetadata,
    PlannerDecisionSource,
    resolve_planner_decision_metadata,
)
from yewo.agent.planning.planner import ActionProposalSource, PlanningRequest
from yewo.agent.replay.procedure.grounding import ProcedureGroundingMatch
from yewo.agent.replay.procedure.records import (
    ProcedureEnvironmentEvent,
    ProcedureRepairEvidence,
    ProcedureRepairTarget,
)
from yewo.agent.replay.procedure.source import CompiledProcedureActionSource
from yewo.agent.runtime.observation_settlement import (
    ActionSettlementPolicy,
    ActionSettlementPolicySource,
)
from yewo.artifacts import new_record_identifier
from yewo.grounding import GroundingRequest
from yewo.devices.application_lifecycle import trusted_system_surface


PROCEDURE_REPAIR_MAXIMUM_ACTIONS_PER_ATTEMPT = 8
PROCEDURE_REPAIR_MAXIMUM_TOTAL_ACTIONS = 32


class BoundedProcedureRepairActionSource:
    """Repair multiple divergences within one total per-run action allowance."""

    def __init__(
        self,
        procedure_source: CompiledProcedureActionSource,
        repair_source: ActionProposalSource,
        *,
        maximum_actions_per_attempt: int = PROCEDURE_REPAIR_MAXIMUM_ACTIONS_PER_ATTEMPT,
        maximum_total_actions: int = PROCEDURE_REPAIR_MAXIMUM_TOTAL_ACTIONS,
        identifier_factory: Callable[[], str] = new_record_identifier,
        settlement_policy_source: ActionSettlementPolicySource | None = None,
    ) -> None:
        if maximum_actions_per_attempt <= 0 or maximum_total_actions <= 0:
            raise ValueError("Procedure repair requires positive action limits.")
        if maximum_actions_per_attempt > maximum_total_actions:
            raise ValueError("The per-attempt repair limit cannot exceed the total limit.")
        self._procedure_source = procedure_source
        self._repair_source = repair_source
        self._maximum_actions_per_attempt = maximum_actions_per_attempt
        self._maximum_total_actions = maximum_total_actions
        self._identifier_factory = identifier_factory
        self._settlement_policy_source = settlement_policy_source
        self._target: ProcedureRepairTarget | None = None
        self._repairs: list[ProcedureRepairEvidence] = []
        self._environment_events: list[ProcedureEnvironmentEvent] = []
        self._repair_active = False
        self._repair_action_count = 0
        self._repair_model_call_count = 0
        self._attempt_start_action_count = 0
        self._attempt_start_model_call_count = 0
        self._pending_model_call_count = 0
        self._last_model_call_count = 0
        self._rejoined_observation_id: str | None = None
        self._keyboard_repair_stage = 0
        self._repair_reason = "Procedure repair has not started."
        self._last_metadata = PlannerDecisionMetadata(
            source=PlannerDecisionSource.COMPILED_PROCEDURE,
            reused_flow_id=procedure_source.procedure_id,
        )

    @property
    def completed_step_count(self) -> int:
        return self._procedure_source.completed_step_count

    @property
    def failed_procedure_index(self) -> int | None:
        return self._procedure_source.failed_procedure_index

    @property
    def divergence(self):
        return self._procedure_source.divergence

    @property
    def divergences(self):
        return self._procedure_source.divergences

    @property
    def milestones(self):
        return self._procedure_source.milestones

    @property
    def next_procedure_index(self) -> int:
        return self._procedure_source.next_procedure_index

    @property
    def replay_complete(self) -> bool:
        return self._procedure_source.replay_complete

    def resolve(
        self,
        default: ActionSettlementPolicy,
        decision: PlannerDecisionMetadata,
    ) -> ActionSettlementPolicy:
        """Tune only exact compiled steps with retained timing evidence."""
        source = self._settlement_policy_source
        return default if source is None else source.resolve(default, decision)

    @property
    def repair_evidence(self) -> ProcedureRepairEvidence | None:
        """Return the current or most recently completed repair record."""
        if self._target is None:
            return self._repairs[-1] if self._repairs else None
        return self._current_repair_evidence()

    @property
    def repairs(self) -> tuple[ProcedureRepairEvidence, ...]:
        """Return completed repairs plus any active attempt."""
        if self._target is None:
            return tuple(self._repairs)
        return (*self._repairs, self._current_repair_evidence())

    @property
    def environment_events(self) -> tuple[ProcedureEnvironmentEvent, ...]:
        """Return deterministic system-surface dismissals in occurrence order."""
        return tuple(self._environment_events)

    def _current_repair_evidence(self) -> ProcedureRepairEvidence:
        target = self._target
        if target is None:
            raise AssertionError("Active procedure repair requires a target.")
        succeeded = self._rejoined_observation_id is not None
        return ProcedureRepairEvidence(
            target=target,
            maximum_actions=self._maximum_actions_per_attempt,
            action_count=self._repair_action_count - self._attempt_start_action_count,
            model_call_count=(self._repair_model_call_count - self._attempt_start_model_call_count),
            succeeded=succeeded,
            rejoined_observation_id=self._rejoined_observation_id,
            reason=self._repair_reason,
        )

    def required_model_requests(self, request: PlanningRequest) -> int:
        """Reserve one repair decision only when the rejoin guard still fails."""
        if self._trusted_system_surface(request) is not None:
            return 0
        if not self._repair_active:
            return self._procedure_source.required_model_requests(request)
        if self._procedure_source.matches_repair_target(request.observation):
            return 0
        if not self._has_repair_capacity():
            return 0
        if self._keyboard_repair_pending(request):
            return 0
        repair_request = self._repair_request(request)
        self._pending_model_call_count = self._repair_source.required_model_requests(repair_request)
        return self._pending_model_call_count

    def plan(self, request: PlanningRequest) -> ActionProposal:
        """Run compiled work and repair each divergence within the shared limit."""
        self._last_model_call_count = 0
        environment_action = self._plan_environment_action(request)
        if environment_action is not None:
            return environment_action
        if self._repair_active:
            return self._plan_repair(request)
        return self._plan_compiled(request)

    def _plan_compiled(self, request: PlanningRequest) -> ActionProposal:
        proposal = self._procedure_source.plan(request)
        self._last_metadata = self._procedure_source.decision_metadata()
        target = self._procedure_source.repair_target
        if proposal.action is not ActionKind.CANNOT_PROGRESS or target is None:
            return proposal
        if self._repair_action_count >= self._maximum_total_actions:
            return self._stop(request, "Procedure repair exhausted its total action limit.")
        self._target = target
        self._repair_active = True
        self._attempt_start_action_count = self._repair_action_count
        self._attempt_start_model_call_count = self._repair_model_call_count
        self._rejoined_observation_id = None
        self._keyboard_repair_stage = 0
        self._repair_action_count += 1
        self._repair_reason = "Repair is waiting for one fresh rejoin observation."
        self._last_metadata = PlannerDecisionMetadata(source=PlannerDecisionSource.SYSTEM)
        return ActionProposal(
            proposal_id=self._identifier_factory(),
            source_observation_id=request.observation.observation_id,
            action=ActionKind.WAIT,
            duration_seconds=0.1,
            expected_outcome="A fresh observation is available for the repair guard.",
            reason="Capture one fresh observation before bounded procedure repair.",
        )

    def decision_metadata(self) -> PlannerDecisionMetadata:
        """Return provenance for the most recent compiled or repair decision."""
        return self._last_metadata

    def last_planning_model_request_count(self) -> int:
        """Return actual model calls consumed by the most recent plan operation."""
        return self._last_model_call_count

    def grounding_match(self, request: GroundingRequest) -> ProcedureGroundingMatch | None:
        """Expose compiled grounding only for a pending procedure action."""
        return self._procedure_source.grounding_match(request)

    def _plan_repair(self, request: PlanningRequest) -> ActionProposal:
        if self._procedure_source.rejoin(request.observation):
            self._rejoined_observation_id = request.observation.observation_id
            self._repair_reason = "The compiled rejoin guard accepted the live observation."
            self._finish_repair()
            return self._plan_compiled(request)
        if not self._has_repair_capacity():
            self._repair_reason = "Procedure repair exhausted its bounded action allowance."
            return self._stop(request, self._repair_reason)
        keyboard_action = self._plan_keyboard_repair(request)
        if keyboard_action is not None:
            self._repair_action_count += 1
            self._last_metadata = PlannerDecisionMetadata(source=PlannerDecisionSource.SYSTEM)
            return keyboard_action
        repair_request = self._repair_request(request)
        self._last_model_call_count = self._pending_model_call_count
        self._repair_model_call_count += self._last_model_call_count
        proposal = self._repair_source.plan(repair_request)
        if proposal.action in {ActionKind.DONE, ActionKind.CANNOT_PROGRESS}:
            self._repair_reason = "The repair planner did not provide another device action."
            return self._stop(request, self._repair_reason)
        if self._procedure_source.repair_would_repeat_failed_action(proposal):
            self._repair_reason = (
                "Procedure repair refused to repeat the failed non-navigation action."
            )
            return self._stop(request, self._repair_reason)
        self._repair_action_count += 1
        self._last_metadata = resolve_planner_decision_metadata(self._repair_source)
        return proposal

    def _keyboard_repair_pending(self, request: PlanningRequest) -> bool:
        target = self._target
        observation = request.observation
        if (
            target is None
            or self._keyboard_repair_stage >= 3
            or ActionKind.FOCUS not in request.available_actions
            or target.keyboard_visible is not True
            or target.established_by_action is not ActionKind.FOCUS
            or target.established_by_target_expression is None
            or target.keyboard_reset_target_expression is None
            or observation.foreground_application_id != target.foreground_application_id
            or observation.platform_context != target.platform_context
            or observation.orientation is not target.orientation
        ):
            return False
        return self._keyboard_repair_stage > 0 or observation.keyboard_visible is False

    def _has_repair_capacity(self) -> bool:
        return (
            self._repair_action_count < self._maximum_total_actions
            and self._repair_action_count - self._attempt_start_action_count
            < self._maximum_actions_per_attempt
        )

    def _plan_keyboard_repair(self, request: PlanningRequest) -> ActionProposal | None:
        if not self._keyboard_repair_pending(request):
            return None
        target = self._target
        if target is None:
            raise AssertionError("Keyboard repair requires one compiled target state.")
        if self._keyboard_repair_stage == 0:
            expression = target.established_by_target_expression
            expected_outcome = "The required focused field reopens the keyboard."
            reason = "Request the keyboard from the compiler-recorded required field."
        elif self._keyboard_repair_stage == 1:
            expression = target.keyboard_reset_target_expression
            expected_outcome = "The alternate text field gains focus and opens the keyboard."
            reason = "Reset keyboard focus through one compiler-recorded alternate text field."
        else:
            expression = target.established_by_target_expression
            expected_outcome = (
                "The required text field regains focus while the keyboard stays open."
            )
            reason = "Return focus to the compiler-recorded field required for procedure rejoin."
        if expression is None:
            raise AssertionError("Keyboard repair requires one target expression.")
        self._keyboard_repair_stage += 1
        proposal = ActionProposal(
            proposal_id=self._identifier_factory(),
            source_observation_id=request.observation.observation_id,
            action=ActionKind.FOCUS,
            target_expression=expression,
            expected_outcome=expected_outcome,
            reason=reason,
        )
        self._procedure_source.authorize_keyboard_repair_grounding(
            proposal,
            request.observation,
        )
        return proposal

    def _repair_request(self, request: PlanningRequest) -> PlanningRequest:
        target = self._target
        if target is None:
            raise AssertionError("Procedure repair requires one compiled target state.")
        divergence = self._procedure_source.active_divergence
        retry_fact = (
            "Do not repeat the procedure action that produced the divergence."
            if divergence is not None and divergence.guard == "expectation"
            else "The compiled action was not dispatched. Restore its source state first."
        )
        target_facts = _target_facts(target)
        return request.model_copy(
            update={
                "objective": (
                    "Restore the application to the expected source state for the remaining "
                    "compiled procedure."
                ),
                "known_facts": (
                    retry_fact,
                    "Do not claim completion. Deterministic code decides whether repair rejoins.",
                    *target_facts,
                ),
                "open_work": (f"Reach compiled procedure state {target.expected_state_id}.",),
                "completion_conditions": (
                    "The compiled procedure rejoin guard accepts the current screen.",
                ),
                "discovery_mode": True,
            }
        )

    def _stop(self, request: PlanningRequest, reason: str) -> ActionProposal:
        if self._repair_active:
            self._repair_reason = reason
            self._finish_repair()
        self._last_metadata = PlannerDecisionMetadata(source=PlannerDecisionSource.SYSTEM)
        return ActionProposal(
            proposal_id=self._identifier_factory(),
            source_observation_id=request.observation.observation_id,
            action=ActionKind.CANNOT_PROGRESS,
            reason=reason,
        )

    def _finish_repair(self) -> None:
        """Freeze the active attempt and reset only its per-attempt state."""
        self._repairs.append(self._current_repair_evidence())
        self._repair_active = False
        self._target = None
        self._rejoined_observation_id = None
        self._keyboard_repair_stage = 0

    def _trusted_system_surface(self, request: PlanningRequest):
        observation = request.observation
        if observation.foreground_application_id == self._procedure_source.application_id:
            return None
        return trusted_system_surface(
            observation.device.platform,
            observation.foreground_application_id,
        )

    def _plan_environment_action(
        self,
        request: PlanningRequest,
    ) -> ActionProposal | None:
        surface = self._trusted_system_surface(request)
        if surface is None:
            return None
        if self._repair_action_count >= self._maximum_total_actions:
            return self._stop(
                request,
                "Procedure recovery exhausted its total action limit on a system surface.",
            )
        if self._repair_active and not self._has_repair_capacity():
            return self._stop(
                request,
                "Procedure recovery exhausted the active repair attempt's action limit.",
            )
        if ActionKind.BACK not in request.available_actions:
            return self._stop(
                request,
                "The active policy cannot dismiss the trusted system surface.",
            )
        self._repair_action_count += 1
        self._environment_events.append(
            ProcedureEnvironmentEvent(
                surface=surface,
                before_observation_id=request.observation.observation_id,
            )
        )
        self._last_metadata = PlannerDecisionMetadata(source=PlannerDecisionSource.SYSTEM)
        return ActionProposal(
            proposal_id=self._identifier_factory(),
            source_observation_id=request.observation.observation_id,
            action=ActionKind.BACK,
            expected_outcome="The trusted system surface closes and the app returns.",
            reason="Dismiss the trusted system surface before compiled replay continues.",
        )


def _target_facts(target: ProcedureRepairTarget) -> tuple[str, ...]:
    """Describe the expected state without claiming that it is current."""
    metadata = (
        "Expected repair state metadata: "
        f"foreground application={target.foreground_application_id or 'unknown'}, "
        f"platform context={target.platform_context or 'unknown'}, "
        f"keyboard visible={target.keyboard_visible}, "
        f"orientation={target.orientation.value if target.orientation is not None else 'unknown'}."
    )
    facts = [metadata]
    if target.established_by_action is not None:
        facts.append(
            "Source evidence established the expected state after action="
            f"{target.established_by_action.value}, target="
            f"{target.established_by_target_expression or 'none'}."
        )
    if target.keyboard_visible is True and target.established_by_action is ActionKind.FOCUS:
        facts.append(
            "If focus on the required field does not open the keyboard, focus one other visible "
            "text field once and then return to the required field."
        )
    if target.keyboard_reset_target_expression is not None:
        facts.append(
            "The compiler-recorded alternate field for one keyboard-focus reset is target="
            f"{target.keyboard_reset_target_expression}."
        )
    if target.next_action is not None:
        facts.append(
            "The remaining compiled procedure resumes with action="
            f"{target.next_action.value}, target={target.next_target_expression or 'none'}."
        )
    return tuple(facts)
