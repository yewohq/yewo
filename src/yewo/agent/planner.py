"""Select one semantic next action from current task evidence."""

from __future__ import annotations

from collections.abc import Callable
import json
from typing import Annotated, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from yewo.artifacts import RecordIdentifier, new_record_identifier
from yewo.models import (
    ImageModelContent,
    ModelCallDisposition,
    ModelClient,
    ModelMessage,
    ModelMessageRole,
    ModelRequest,
    ModelResponse,
    ModelRole,
    TextModelContent,
    build_model_response_contract,
)
from yewo.grounding import GroundedPoint, GroundingCorrection, GroundingEvidence, PixelPoint
from yewo.observations import Observation
from yewo.agent.planner_prompt import TASK_PLANNER_SYSTEM_PROMPT, TASK_PLANNER_USER_PROMPT

from yewo.agent.actuation import (
    ACTION_DEFINITIONS,
    ActionHandling,
    ActionKind,
    ActionOutcome,
    ActionOutcomeDisposition,
    ActionProposal,
    ActionRefusal,
    DispatchDisposition,
    DispatchReceipt,
    DeclaredValueReference,
    LiteralText,
)


PlanningText = Annotated[str, Field(min_length=1, max_length=4_000)]
RECENT_PLANNING_HISTORY_LIMIT = 14
GROUNDING_CORRECTION_MINIMUM_DISTANCE_FRACTION = 0.02
GROUNDING_CORRECTION_MINIMUM_DISTANCE_PX = 8

_TERMINAL_ACTIONS = frozenset(
    action
    for action, definition in ACTION_DEFINITIONS.items()
    if definition.handling is ActionHandling.TERMINAL
)


class GroundedTargetPoint(BaseModel):
    """A dispatched target point and its source-screenshot dimensions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    point: PixelPoint
    screenshot_width_px: int = Field(gt=0)
    screenshot_height_px: int = Field(gt=0)

    @model_validator(mode="after")
    def _validate_point_bounds(self) -> Self:
        if self.point.x >= self.screenshot_width_px or self.point.y >= self.screenshot_height_px:
            raise ValueError("The grounded target point must be within the source screenshot.")
        return self

    def correction(self) -> GroundingCorrection:
        """Build the explicit minimum-distance rule for one correction."""
        minimum_dimension = min(self.screenshot_width_px, self.screenshot_height_px)
        return GroundingCorrection(
            previous_point=self.point,
            minimum_distance_px=max(
                GROUNDING_CORRECTION_MINIMUM_DISTANCE_PX,
                round(minimum_dimension * GROUNDING_CORRECTION_MINIMUM_DISTANCE_FRACTION),
            ),
        )


class PlanningHistoryEntry(BaseModel):
    """One previous proposal and its trusted runtime feedback."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    proposal: ActionProposal
    refusal: ActionRefusal | None = None
    receipt: DispatchReceipt | None = None
    outcome: ActionOutcome | None = None
    grounded_target_point: GroundedTargetPoint | None = None

    @model_validator(mode="after")
    def _validate_feedback(self) -> Self:
        feedback = tuple(
            item for item in (self.refusal, self.receipt, self.outcome) if item is not None
        )
        if len(feedback) != 1:
            raise ValueError("Planning history requires exactly one feedback record.")
        if self.refusal is not None:
            if self.refusal.proposal_id != self.proposal.proposal_id:
                raise ValueError("The action refusal does not match the proposal.")
        elif self.receipt is not None:
            if self.receipt.disposition is not DispatchDisposition.TARGET_REJECTED:
                raise ValueError("Only a target rejection is recoverable planning feedback.")
            if (
                self.receipt.proposal_id != self.proposal.proposal_id
                or self.receipt.action is not self.proposal.action
            ):
                raise ValueError("The dispatch rejection does not match the proposal.")
        elif self.outcome is not None and self.outcome.action is not self.proposal.action:
            raise ValueError("The action outcome does not match the proposal action.")
        elif (
            self.outcome is not None
            and self.outcome.source_observation_id != self.proposal.source_observation_id
        ):
            raise ValueError("The action outcome does not match the proposal observation.")
        has_point_feedback = (
            self.outcome is not None
            and self.outcome.disposition is ActionOutcomeDisposition.NO_OBSERVABLE_CHANGE
        ) or (
            self.receipt is not None
            and self.receipt.disposition is DispatchDisposition.TARGET_REJECTED
        )
        if self.grounded_target_point is not None and (
            not has_point_feedback or self.proposal.target_expression is None
        ):
            raise ValueError(
                "A grounded target point requires targeted no-progress feedback."
            )
        return self


def build_planning_history_entry(
    proposal: ActionProposal,
    outcome: ActionOutcome,
    target_grounding: GroundingEvidence | None,
) -> PlanningHistoryEntry:
    """Build planner feedback with trusted point evidence when it can aid recovery."""
    point_evidence = None
    if (
        target_grounding is not None
        and outcome.disposition is ActionOutcomeDisposition.NO_OBSERVABLE_CHANGE
        and isinstance(target_grounding.result.location, GroundedPoint)
    ):
        screenshot = target_grounding.request.screenshot
        point_evidence = GroundedTargetPoint(
            point=target_grounding.result.location.pixel,
            screenshot_width_px=screenshot.width_px,
            screenshot_height_px=screenshot.height_px,
        )
    return PlanningHistoryEntry(
        proposal=proposal,
        outcome=outcome,
        grounded_target_point=point_evidence,
    )


def build_target_rejection_history_entry(
    proposal: ActionProposal,
    receipt: DispatchReceipt,
    target_grounding: GroundingEvidence | None,
) -> PlanningHistoryEntry:
    """Preserve a rejected target's device reason and grounded point for recovery."""
    point_evidence = None
    if target_grounding is not None and isinstance(target_grounding.result.location, GroundedPoint):
        screenshot = target_grounding.request.screenshot
        point_evidence = GroundedTargetPoint(
            point=target_grounding.result.location.pixel,
            screenshot_width_px=screenshot.width_px,
            screenshot_height_px=screenshot.height_px,
        )
    return PlanningHistoryEntry(
        proposal=proposal,
        receipt=receipt,
        grounded_target_point=point_evidence,
    )


def grounding_correction_failure_reason(
    proposal: ActionProposal,
    outcome: ActionOutcome,
) -> str | None:
    """Return the terminal reason for an ineffective bounded correction."""
    if (
        proposal.grounding_correction_for_proposal_id is not None
        and outcome.disposition is ActionOutcomeDisposition.NO_OBSERVABLE_CHANGE
    ):
        return "The grounding correction produced no observable change."
    return None


class PlanningRequest(BaseModel):
    """Current evidence and task state needed to select one action."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    observation: Observation
    objective: PlanningText
    known_facts: tuple[PlanningText, ...] = Field(default=(), max_length=100)
    open_work: tuple[PlanningText, ...] = Field(default=(), max_length=100)
    completion_conditions: tuple[PlanningText, ...] = Field(min_length=1, max_length=100)
    declared_value_names: tuple[RecordIdentifier, ...] = Field(default=(), max_length=100)
    history: tuple[PlanningHistoryEntry, ...] = Field(default=(), max_length=50)
    available_actions: frozenset[ActionKind] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_unique_declared_values(self) -> Self:
        if len(set(self.declared_value_names)) != len(self.declared_value_names):
            raise ValueError("Declared value names must be unique.")
        return self


class TaskPlanningError(RuntimeError):
    """A model call did not produce one usable action proposal."""

    def __init__(self, response: ModelResponse, reason: str) -> None:
        super().__init__(reason)
        self.response = response
        self.disposition = response.disposition


class ActionProposalSource(Protocol):
    """Supply one validated semantic action for the current agent evidence."""

    def plan(self, request: PlanningRequest) -> ActionProposal: ...


class TaskPlanner:
    """Ask one configured model profile for the next semantic action."""

    def __init__(
        self,
        model_client: ModelClient,
        model_profile_id: str,
        *,
        identifier_factory: Callable[[], str] = new_record_identifier,
    ) -> None:
        self._model_client = model_client
        self._model_profile_id = model_profile_id
        self._identifier_factory = identifier_factory

    def plan(self, request: PlanningRequest) -> ActionProposal:
        """Return one locally validated next-action proposal."""
        proposal_id = self._identifier_factory()
        model_request = self._build_model_request(request, proposal_id)
        response = self._model_client.complete(model_request, ActionProposal)
        if response.disposition is not ModelCallDisposition.COMPLETED:
            raise TaskPlanningError(
                response,
                response.reason or "The task-planner model call did not complete.",
            )
        if response.structured_output is None:
            raise TaskPlanningError(response, "The task-planner response has no structured output.")
        try:
            proposal = ActionProposal.model_validate(response.structured_output)
        except ValidationError as error:
            raise TaskPlanningError(
                response,
                "The task-planner response does not contain a valid action proposal.",
            ) from error
        if (
            proposal.proposal_id != proposal_id
            or proposal.source_observation_id != request.observation.observation_id
        ):
            raise TaskPlanningError(
                response,
                "The action proposal does not match the planning request.",
            )
        if proposal.action not in _available_actions(request):
            raise TaskPlanningError(
                response,
                "The action proposal is not available for this task.",
            )
        if (
            isinstance(proposal.text, DeclaredValueReference)
            and proposal.text.value_name not in request.declared_value_names
        ):
            raise TaskPlanningError(
                response,
                "The action proposal names an unavailable declared value.",
            )
        correction_error = _grounding_correction_error(proposal, request.history)
        if correction_error is not None:
            raise TaskPlanningError(response, correction_error)
        return proposal

    def _build_model_request(
        self,
        request: PlanningRequest,
        proposal_id: str,
    ) -> ModelRequest:
        earlier_history = request.history[:-RECENT_PLANNING_HISTORY_LIMIT]
        recent_history = request.history[-RECENT_PLANNING_HISTORY_LIMIT:]
        context = {
            "proposal_id": proposal_id,
            "source_observation_id": request.observation.observation_id,
            "objective": request.objective,
            "known_facts": request.known_facts,
            "open_work": request.open_work,
            "completion_conditions": request.completion_conditions,
            "declared_value_names": request.declared_value_names,
            "available_actions": sorted(action.value for action in _available_actions(request)),
            "current_observation": _observation_context(request.observation),
            "earlier_history_summary": _history_summary(earlier_history),
            "history": [_history_context(entry) for entry in recent_history],
        }
        user_prompt = (
            TASK_PLANNER_USER_PROMPT
            + "\n"
            + json.dumps(context, sort_keys=True, separators=(",", ":"))
        )
        return ModelRequest(
            request_id=self._identifier_factory(),
            profile_id=self._model_profile_id,
            role=ModelRole.TASK_PLANNER,
            messages=(
                ModelMessage(
                    role=ModelMessageRole.SYSTEM,
                    content=(TextModelContent(text=TASK_PLANNER_SYSTEM_PROMPT),),
                ),
                ModelMessage(
                    role=ModelMessageRole.USER,
                    content=(
                        TextModelContent(text=user_prompt),
                        ImageModelContent(screenshot=request.observation.redacted_screenshot),
                    ),
                ),
            ),
            response_contract=build_model_response_contract(
                "action-proposal-v1",
                ActionProposal.model_json_schema(),
            ),
            sensitivity=request.observation.redacted_screenshot.sensitivity,
            retention=request.observation.redacted_screenshot.retention,
        )


def _observation_context(observation: Observation) -> dict[str, object]:
    return {
        "application_id": observation.application.application_id,
        "application_state": observation.application_state.name.lower(),
        "foreground_application_id": observation.foreground_application_id,
        "platform_context": observation.platform_context,
        "keyboard_visible": observation.keyboard_visible,
        "orientation": observation.orientation.value
        if observation.orientation is not None
        else None,
        "screen_fingerprint": observation.screen_fingerprint.value,
        "screenshot_sha256": observation.redacted_screenshot.sha256,
    }


def _available_actions(request: PlanningRequest) -> frozenset[ActionKind]:
    return request.available_actions | _TERMINAL_ACTIONS


def _history_context(entry: PlanningHistoryEntry) -> dict[str, object]:
    proposal = entry.proposal
    context: dict[str, object] = {
        "proposal_id": proposal.proposal_id,
        "action": proposal.action.value,
        "grounding_correction_for_proposal_id": (
            proposal.grounding_correction_for_proposal_id
        ),
        "target_expression": proposal.target_expression,
        "destination_expression": proposal.destination_expression,
        "duration_seconds": proposal.duration_seconds,
        "direction": proposal.direction.value if proposal.direction is not None else None,
        "distance": proposal.distance.value if proposal.distance is not None else None,
        "expected_outcome": proposal.expected_outcome,
    }
    if proposal.text is not None:
        context["text"] = _text_context(proposal.text)
    if entry.refusal is not None:
        context["feedback"] = {
            "kind": "refusal",
            "code": entry.refusal.code.value,
            "reason": entry.refusal.reason,
        }
    elif entry.receipt is not None:
        context["feedback"] = {
            "kind": "receipt",
            "disposition": entry.receipt.disposition.value,
            "reason": entry.receipt.reason,
        }
    elif entry.outcome is not None:
        context["feedback"] = {
            "kind": "outcome",
            "disposition": entry.outcome.disposition.value,
            "signals": sorted(signal.value for signal in entry.outcome.signals),
            "reason": entry.outcome.reason,
        }
    if entry.grounded_target_point is not None:
        context["grounded_target_point"] = entry.grounded_target_point.model_dump(mode="json")
    return context


def _history_summary(
    entries: tuple[PlanningHistoryEntry, ...],
) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    for entry in entries:
        context = _compact_history_context(entry)
        key = json.dumps(context, sort_keys=True, separators=(",", ":"))
        existing = grouped.get(key)
        if existing is None:
            grouped[key] = {**context, "count": 1}
        else:
            count = existing["count"]
            if not isinstance(count, int):
                raise TypeError("A compact history count must be an integer.")
            existing["count"] = count + 1
    return list(grouped.values())


def _compact_history_context(entry: PlanningHistoryEntry) -> dict[str, object]:
    proposal = entry.proposal
    context: dict[str, object] = {"action": proposal.action.value}
    for field_name, value in (
        ("target_expression", proposal.target_expression),
        ("destination_expression", proposal.destination_expression),
        ("direction", proposal.direction.value if proposal.direction is not None else None),
        ("distance", proposal.distance.value if proposal.distance is not None else None),
    ):
        if value is not None:
            context[field_name] = value
    if entry.refusal is not None:
        context["feedback"] = {
            "kind": "refusal",
            "code": entry.refusal.code.value,
        }
    elif entry.receipt is not None:
        context["feedback"] = {
            "kind": "receipt",
            "disposition": entry.receipt.disposition.value,
            "reason": entry.receipt.reason,
        }
    elif entry.outcome is not None:
        context["feedback"] = {
            "kind": "outcome",
            "disposition": entry.outcome.disposition.value,
            "signals": sorted(signal.value for signal in entry.outcome.signals),
        }
    return context


def _text_context(text: LiteralText | DeclaredValueReference) -> dict[str, object]:
    if isinstance(text, DeclaredValueReference):
        return {"source": text.source, "declared_value_name": text.value_name}
    return {"source": text.source, "literal_value_retained": False}


def _grounding_correction_error(
    proposal: ActionProposal,
    history: tuple[PlanningHistoryEntry, ...],
) -> str | None:
    failed_proposal_id = proposal.grounding_correction_for_proposal_id
    if failed_proposal_id is None:
        return None
    failed_entry = next(
        (entry for entry in reversed(history) if entry.proposal.proposal_id == failed_proposal_id),
        None,
    )
    if failed_entry is None or failed_entry.grounded_target_point is None:
        return "The grounding correction does not reference failed target-point evidence."
    failed_proposal = failed_entry.proposal
    if failed_proposal.grounding_correction_for_proposal_id is not None:
        return "A grounding correction cannot correct another grounding correction."
    if any(
        entry.proposal.grounding_correction_for_proposal_id == failed_proposal_id
        for entry in history
    ):
        return "The failed grounding already has a correction attempt."
    if proposal.action is not failed_proposal.action:
        return "A grounding correction must use the failed proposal's action."
    if proposal.target_expression == failed_proposal.target_expression:
        return "A grounding correction must refine the failed target expression."
    return None
