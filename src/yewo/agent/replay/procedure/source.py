"""Deterministic action source for one compiler-approved replay procedure."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from yewo.agent.actuation import ActionKind, ActionProposal
from yewo.agent.planning.evidence import PlannerDecisionMetadata, PlannerDecisionSource
from yewo.agent.planning.history import PlanningHistoryEntry
from yewo.agent.planning.records import (
    PlanningRequest,
)
from yewo.agent.replay.procedure.grounding import (
    ProcedureGroundingAuthorization,
    ProcedureGroundingMatch,
    build_keyboard_repair_grounding_match,
)
from yewo.agent.replay.procedure.feedback import application_recovered, find_action_feedback
from yewo.agent.replay.procedure.records import MilestoneEvidence, ProcedureDivergence
from yewo.agent.replay.procedure.records import ProcedureRepairTarget
from yewo.agent.replay.reconciliation.records import (
    EvidenceDocument,
    ReplayProcedure,
    ReplayProcedureStep,
)
from yewo.agent.replay.procedure.system_surfaces import expected_system_surface_matches
from yewo.agent.replay.procedure.repair_targets import (
    build_procedure_repair_preparation,
)
from yewo.agent.replay.procedure.entry import select_procedure_entry_index
from yewo.artifacts import new_record_identifier
from yewo.grounding import GroundingRequest
from yewo.agent.replay.state_matching import (
    AMBIGUOUS_SOURCE_STATE,
    trace_repair_target_match,
    match_evidence_state,
    repair_state_matches_observation,
    state_matches_observation,
)
from yewo.observations import Observation, ScreenshotReference

class CompiledProcedureActionSource:
    """Supply stored actions only while current-run guards pass."""

    def __init__(
        self,
        procedure: ReplayProcedure,
        evidence: EvidenceDocument,
        *,
        identifier_factory: Callable[[], str] = new_record_identifier,
        screenshot_loader: Callable[[ScreenshotReference], bytes] | None = None,
    ) -> None:
        if procedure.evidence_id != evidence.evidence_id:
            raise ValueError("The replay procedure does not match its evidence document.")
        if procedure.source_run_id != evidence.run.run_id:
            raise ValueError("The replay procedure does not match its source run.")
        if procedure.application != evidence.run.application:
            raise ValueError("The replay procedure does not match its application build.")
        if not procedure.steps or procedure.steps[0].procedure_index != 0:
            raise ValueError("Procedure replay must start at procedure index zero.")
        self._procedure = procedure
        self._evidence = evidence
        self._identifier_factory = identifier_factory
        self._screenshot_loader = screenshot_loader
        self._next_index = 0
        self._completed_indexes: set[int] = set()
        self._entry_selected = False
        self._pending_proposal_id: str | None = None
        self._pending_step: ReplayProcedureStep | None = None
        self._grounding_authorization: ProcedureGroundingAuthorization | None = None
        self._repair_grounding_match: ProcedureGroundingMatch | None = None
        self._repair_grounding_steps_by_expression: dict[str, ReplayProcedureStep] = {}
        self._failed_index: int | None = None
        self._active_divergence: ProcedureDivergence | None = None
        self._divergences: list[ProcedureDivergence] = []
        self._repair_target: ProcedureRepairTarget | None = None
        self._repair_requires_exact_content_match = False
        self._blocked_repair_proposal: ActionProposal | None = None
        self._milestones: list[MilestoneEvidence] = []
        self._last_metadata = PlannerDecisionMetadata(
            source=PlannerDecisionSource.COMPILED_PROCEDURE,
            reused_flow_id=procedure.procedure_id,
        )

    @property
    def completed_step_count(self) -> int:
        """Return the number of live expectations that passed."""
        return len(self._completed_indexes)

    @property
    def procedure_id(self) -> str:
        """Return the compiled procedure identifier."""
        return self._procedure.procedure_id

    @property
    def application_id(self) -> str:
        """Return the application expected to own ordinary procedure screens."""
        return self._procedure.application.application_id

    @property
    def failed_procedure_index(self) -> int | None:
        """Return the first procedure index that failed a live guard."""
        return self._failed_index

    @property
    def divergence(self) -> ProcedureDivergence | None:
        """Return the most recent failed guard, including a repaired guard."""
        return self._divergences[-1] if self._divergences else None

    @property
    def active_divergence(self) -> ProcedureDivergence | None:
        """Return the failed guard currently being repaired."""
        return self._active_divergence

    @property
    def divergences(self) -> tuple[ProcedureDivergence, ...]:
        """Return every failed guard in occurrence order."""
        return tuple(self._divergences)

    @property
    def repair_target(self) -> ProcedureRepairTarget | None:
        """Return the one state that can resume execution after divergence."""
        return self._repair_target

    @property
    def milestones(self) -> tuple[MilestoneEvidence, ...]:
        """Return current-run milestone captures in condition order."""
        return tuple(sorted(self._milestones, key=lambda item: item.completion_condition_index))

    @property
    def next_procedure_index(self) -> int:
        """Return the procedure index that has not passed yet."""
        return min(self._next_index, len(self._procedure.steps) - 1)

    @property
    def replay_complete(self) -> bool:
        """Return whether every compiled expectation passed in this run."""
        return self._next_index == len(self._procedure.steps)

    def accepts_expected_system_surface(self, observation: Observation) -> bool:
        """Return whether a compiled guard expects this platform-owned screen."""
        next_step = (
            self._procedure.steps[self._next_index]
            if self._next_index < len(self._procedure.steps)
            else None
        )
        return expected_system_surface_matches(
            observation,
            self._pending_step,
            next_step,
            self._evidence,
            self._screenshot_loader,
        )

    def rejoin(self, observation: Observation) -> bool:
        """Resume at the repair target only after its compiled guard accepts."""
        target = self._repair_target
        if target is None or not self._repair_target_matches(observation):
            return False
        repaired_step_count = max(0, target.resume_procedure_index - self._next_index)
        if repaired_step_count:
            repaired_step = self._procedure.steps[target.resume_procedure_index - 1]
            for condition_index in repaired_step.captures_completion_conditions:
                self._milestones.append(
                    MilestoneEvidence(
                        completion_condition_index=condition_index,
                        procedure_index=repaired_step.procedure_index,
                        source_step=repaired_step.source_step,
                        observation_id=observation.observation_id,
                        screenshot_artifact_id=observation.redacted_screenshot.artifact_id,
                        screen_fingerprint=observation.screen_fingerprint,
                    )
                )
        previous_index = self._next_index
        self._next_index = target.resume_procedure_index
        self._completed_indexes.update(range(previous_index, target.resume_procedure_index))
        self._clear_repair_state()
        return True

    def matches_repair_target(self, observation: Observation) -> bool:
        """Return whether the current repair target accepts an observation."""
        target = self._repair_target
        return target is not None and self._repair_target_matches(observation)

    def _clear_repair_state(self) -> None:
        self._repair_target = None
        self._repair_requires_exact_content_match = False
        self._repair_grounding_match = None
        self._repair_grounding_steps_by_expression.clear()
        self._blocked_repair_proposal = None
        self._active_divergence = None

    def _repair_target_matches(self, observation: Observation) -> bool:
        target = self._repair_target
        if target is None:
            return False
        source_match = match_evidence_state(
            observation,
            target.expected_state_id,
            self._evidence,
            self._screenshot_loader,
        )
        exact = self._repair_requires_exact_content_match
        if not exact or (
            not source_match.accepted and source_match.reason != AMBIGUOUS_SOURCE_STATE
        ):
            return trace_repair_target_match(source_match.accepted, source_match.reason)
        accepted = repair_state_matches_observation(
            observation,
            self._evidence.states[target.expected_state_id],
            self._evidence.state_grouping_maximum_phash_distance,
            self._screenshot_loader,
        )
        return trace_repair_target_match(accepted, "exact_content_match")

    def repair_would_repeat_failed_action(self, proposal: ActionProposal) -> bool:
        """Reject an exact retry of a dispatched non-navigation action."""
        blocked = self._blocked_repair_proposal
        return blocked is not None and (
            proposal.action == blocked.action
            and proposal.target_expression == blocked.target_expression
        )

    def decision_metadata(self) -> PlannerDecisionMetadata:
        """Return compiled-procedure provenance for the latest proposal."""
        return self._last_metadata

    def required_model_requests(self, request: PlanningRequest) -> int:
        """Return zero because procedure action selection is deterministic."""
        del request
        return 0

    def plan(self, request: PlanningRequest) -> ActionProposal:
        """Return the next stored action only after all preceding guards pass."""
        self._repair_grounding_match = None
        if self._pending_step is not None:
            failure = self._settle_pending(request)
            if failure is not None:
                return failure
        if not self._entry_selected:
            self._select_entry(request.observation)
        if self._next_index >= len(self._procedure.steps):
            return self._terminal(
                request,
                ActionKind.DONE,
                "Every compiled procedure expectation passed. Final audit is still required.",
            )
        step = self._procedure.steps[self._next_index]
        source_match = match_evidence_state(
            request.observation,
            step.source_state_id,
            self._evidence,
            self._screenshot_loader,
        )
        if not source_match.accepted:
            return self._diverged(
                request,
                step,
                "source",
                step.source_state_id,
                f"The procedure source guard failed: {source_match.reason}.",
            )
        if step.proposal.action not in request.available_actions:
            return self._diverged(
                request,
                step,
                "runtime",
                step.source_state_id,
                "The next procedure action is unavailable under the active policy.",
            )
        proposal_id = self._identifier_factory()
        proposal = step.proposal.model_copy(
            update={
                "proposal_id": proposal_id,
                "source_observation_id": request.observation.observation_id,
                "grounding_correction_for_proposal_id": None,
                "interaction_refinement_for_proposal_id": None,
            }
        )
        self._pending_proposal_id = proposal_id
        self._pending_step = step
        if source_match.expected_distance is None:
            raise AssertionError("An accepted source state requires a perceptual distance.")
        screenshot = request.observation.redacted_screenshot
        self._grounding_authorization = ProcedureGroundingAuthorization(
            source_step_id=step.source_step_id,
            procedure_index=step.procedure_index,
            expected_state_id=step.source_state_id,
            observation_id=request.observation.observation_id,
            screenshot_artifact_id=screenshot.artifact_id,
            screenshot_sha256=screenshot.sha256,
            source_fingerprint=request.observation.screen_fingerprint,
            source_orientation=request.observation.orientation,
            source_keyboard_visible=request.observation.keyboard_visible,
            width_px=screenshot.width_px,
            height_px=screenshot.height_px,
            expected_phash_distance=source_match.expected_distance,
            competing_phash_distance=source_match.competing_distance,
            source_visual_match_ratio=source_match.visual_match_ratio,
        )
        self._last_metadata = PlannerDecisionMetadata(
            source=PlannerDecisionSource.COMPILED_PROCEDURE,
            reused_flow_id=self._procedure.procedure_id,
            reused_procedure_step_ids=(step.source_step_id,),
            reused_segment_index=step.procedure_index,
        )
        return proposal

    def grounding_match(self, request: GroundingRequest) -> ProcedureGroundingMatch | None:
        """Return stored target or destination grounding for the pending action."""
        if self._pending_step is None or self._grounding_authorization is None:
            repair_match = self._repair_grounding_match
            if repair_match is None:
                return None
            evidence_request = repair_match.evidence.request
            if (
                evidence_request.expression == request.expression
                and evidence_request.operation is request.operation
                and evidence_request.output_shape is request.output_shape
            ):
                return repair_match
            return None
        for evidence in (
            self._pending_step.target_grounding,
            self._pending_step.destination_grounding,
        ):
            if (
                evidence is not None
                and evidence.request.expression == request.expression
                and evidence.request.operation is request.operation
                and evidence.request.output_shape is request.output_shape
            ):
                return ProcedureGroundingMatch(
                    source_step_id=self._pending_step.source_step_id,
                    evidence=evidence,
                    authorization=self._grounding_authorization,
                    target_visual_descriptor=self._pending_step.target_visual_descriptor,
                )
        return None

    def authorize_keyboard_repair_grounding(
        self,
        proposal: ActionProposal,
        observation: Observation,
    ) -> bool:
        """Authorize one compiler-recorded field on the current repair observation."""
        self._repair_grounding_match = None
        expression = proposal.target_expression
        if proposal.action is not ActionKind.FOCUS or expression is None:
            return False
        step = self._repair_grounding_steps_by_expression.get(expression)
        self._repair_grounding_match = build_keyboard_repair_grounding_match(
            expression,
            observation,
            self._repair_target,
            step,
            self._evidence,
        )
        return self._repair_grounding_match is not None

    def _settle_pending(self, request: PlanningRequest) -> ActionProposal | None:
        step = self._pending_step
        proposal_id = self._pending_proposal_id
        if step is None or proposal_id is None:
            raise AssertionError("A pending procedure step requires its proposal identifier.")
        feedback = find_action_feedback(request.history, proposal_id)
        if feedback is None:
            reason = (
                "Procedure action feedback does not match the pending action."
                if request.history
                else "Procedure action feedback is unavailable."
            )
            return self._diverged(
                request,
                step,
                "runtime",
                step.expects.destination_state_id,
                reason,
            )
        if application_recovered(feedback):
            self._clear_pending_action()
            self._clear_repair_state()
            self._next_index = 0
            self._completed_indexes.clear()
            self._milestones.clear()
            self._entry_selected = False
            return None
        failure = self._settlement_failure(request, step, feedback)
        if failure is not None:
            guard, expected_state_id, reason = failure
            return self._diverged(
                request,
                step,
                guard,
                expected_state_id,
                reason,
            )
        self._record_step_milestones(step, request.observation)
        self._next_index += 1
        self._completed_indexes.add(step.procedure_index)
        self._clear_pending_action()
        return None

    def _settlement_failure(
        self,
        request: PlanningRequest,
        step: ReplayProcedureStep,
        feedback: PlanningHistoryEntry,
    ) -> tuple[Literal["runtime", "expectation"], str | None, str] | None:
        expected_state_id = step.expects.destination_state_id
        if feedback.refusal is not None or feedback.receipt is not None:
            return (
                "runtime",
                expected_state_id,
                "The procedure action was refused before accepted settlement.",
            )
        outcome = feedback.outcome
        if outcome is None:
            return (
                "expectation",
                expected_state_id,
                "The procedure action has no settled outcome evidence.",
            )
        if step.expects.required_signals - outcome.signals:
            return (
                "expectation",
                expected_state_id,
                "The procedure action did not produce every required outcome signal.",
            )
        if expected_state_id is not None and not state_matches_observation(
            request.observation,
            self._evidence.states[expected_state_id],
            self._evidence.state_grouping_maximum_phash_distance,
            self._screenshot_loader,
        ):
            return (
                "expectation",
                expected_state_id,
                "The current observation does not match the procedure expectation.",
            )
        return None

    def _record_step_milestones(
        self,
        step: ReplayProcedureStep,
        observation: Observation,
    ) -> None:
        for condition_index in step.captures_completion_conditions:
            self._milestones.append(
                MilestoneEvidence(
                    completion_condition_index=condition_index,
                    procedure_index=step.procedure_index,
                    source_step=step.source_step,
                    observation_id=observation.observation_id,
                    screenshot_artifact_id=observation.redacted_screenshot.artifact_id,
                    screen_fingerprint=observation.screen_fingerprint,
                )
            )

    def _clear_pending_action(self) -> None:
        self._pending_step = None
        self._pending_proposal_id = None
        self._grounding_authorization = None

    def _select_entry(self, observation: Observation) -> None:
        """Enter at the earliest matching safe source state."""
        entry_index = select_procedure_entry_index(
            observation,
            self._procedure,
            self._evidence,
            self._screenshot_loader,
        )
        self._entry_selected = True
        if entry_index is None:
            return
        self._next_index = entry_index
        self._completed_indexes.update(range(self._next_index))

    def _diverged(
        self,
        request: PlanningRequest,
        step: ReplayProcedureStep,
        guard: Literal["source", "expectation", "runtime"],
        expected_state_id: str | None,
        reason: str,
    ) -> ActionProposal:
        if self._failed_index is None:
            self._failed_index = step.procedure_index
        self._grounding_authorization = None
        if self._active_divergence is None:
            divergence = ProcedureDivergence(
                procedure_id=self._procedure.procedure_id,
                procedure_index=step.procedure_index,
                source_step=step.source_step,
                guard=guard,
                expected_state_id=expected_state_id,
                actual_observation_id=request.observation.observation_id,
                actual_fingerprint=request.observation.screen_fingerprint,
                reason=reason,
            )
            self._active_divergence = divergence
            self._divergences.append(divergence)
            self._prepare_repair(step, guard, expected_state_id, reason)
        return self._terminal(request, ActionKind.CANNOT_PROGRESS, reason)

    def _prepare_repair(
        self,
        step: ReplayProcedureStep,
        guard: Literal["source", "expectation", "runtime"],
        expected_state_id: str | None,
        reason: str,
    ) -> None:
        preparation = build_procedure_repair_preparation(
            self._procedure,
            self._evidence,
            step,
            guard,
            expected_state_id,
            reason,
        )
        if preparation is None:
            return
        if preparation.clears_pending_step:
            self._pending_step = None
            self._pending_proposal_id = None
        if preparation.blocked_proposal is not None:
            self._blocked_repair_proposal = preparation.blocked_proposal
        self._repair_grounding_match = None
        self._repair_grounding_steps_by_expression = preparation.grounding_steps_by_expression
        self._repair_target = preparation.target
        self._repair_requires_exact_content_match = preparation.requires_exact_content_match

    def _terminal(
        self,
        request: PlanningRequest,
        action: ActionKind,
        reason: str,
    ) -> ActionProposal:
        self._last_metadata = PlannerDecisionMetadata(
            source=PlannerDecisionSource.COMPILED_PROCEDURE,
            reused_flow_id=self._procedure.procedure_id,
        )
        return ActionProposal(
            proposal_id=self._identifier_factory(),
            source_observation_id=request.observation.observation_id,
            action=action,
            reason=reason,
        )
