"""Select one semantic next action from current task evidence."""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
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
from yewo.observations import Observation
from yewo.agent.planning_history import (
    EARLIER_HISTORY_SUMMARY_CHARACTER_LIMIT as EARLIER_HISTORY_SUMMARY_CHARACTER_LIMIT,
    EARLIER_HISTORY_SUMMARY_LIMIT as EARLIER_HISTORY_SUMMARY_LIMIT,
    GroundedTargetPoint as GroundedTargetPoint,
    PlanningHistoryEntry,
    PlanningScreenFact,
    planning_history_context,
    planning_screen_fact_context,
    planning_history_summary,
)
from yewo.agent.planner_prompt import TASK_PLANNER_SYSTEM_PROMPT, TASK_PLANNER_USER_PROMPT

from yewo.agent.actuation import (
    ACTION_DEFINITIONS,
    ActionHandling,
    ActionKind,
    ActionProposal,
    DeclaredValueReference,
)


PlanningText = Annotated[str, Field(min_length=1, max_length=4_000)]
RECENT_PLANNING_HISTORY_LIMIT = 14
MAXIMUM_PLANNING_HISTORY_ENTRIES = 110  # 100 task steps plus 10 recovery-credited steps.
PLANNER_MAX_COMPLETION_TOKENS = 8_192

_TERMINAL_ACTIONS = frozenset(
    action
    for action, definition in ACTION_DEFINITIONS.items()
    if definition.handling is ActionHandling.TERMINAL
)


class TaskProgressState(StrEnum):
    """How reconciled evidence relates to one original work item."""

    CURRENT_PROGRESS = "current_progress"
    KNOWN_PROCEDURE = "known_procedure"
    UNRESOLVED = "unresolved"


class TaskProgressContext(BaseModel):
    """Run-scoped memory guidance without changing the original task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    current_progress: tuple[PlanningText, ...] = Field(default=(), max_length=100)
    known_procedures: tuple[PlanningText, ...] = Field(default=(), max_length=100)
    remaining_work: tuple[PlanningText, ...] = Field(default=(), max_length=100)
    historical_memory_facts: tuple[PlanningText, ...] = Field(default=(), max_length=20)
    exhausted_branches: tuple[PlanningText, ...] = Field(default=(), max_length=20)
    recovery_guidance: tuple[PlanningText, ...] = Field(default=(), max_length=20)
    replay_target_transition_id: RecordIdentifier | None = Field(default=None, exclude=True)


class PlanningRequest(BaseModel):
    """Current evidence and task state needed to select one action."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    observation: Observation
    objective: PlanningText
    known_facts: tuple[PlanningText, ...] = Field(default=(), max_length=100)
    screen_facts: tuple[PlanningScreenFact, ...] = Field(default=(), max_length=100)
    open_work: tuple[PlanningText, ...] = Field(default=(), max_length=100)
    completion_conditions: tuple[PlanningText, ...] = Field(min_length=1, max_length=100)
    declared_value_names: tuple[RecordIdentifier, ...] = Field(default=(), max_length=100)
    history: tuple[PlanningHistoryEntry, ...] = Field(
        default=(), max_length=MAXIMUM_PLANNING_HISTORY_ENTRIES
    )
    available_actions: frozenset[ActionKind] = Field(min_length=1)
    progress_context: TaskProgressContext | None = None
    discovery_mode: bool = False

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

    def required_model_requests(self, request: PlanningRequest) -> int:
        """Return the model requests needed for the next decision."""
        ...

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

    def required_model_requests(self, request: PlanningRequest) -> int:
        """Count one model request for one planner decision."""
        del request
        return 1

    def plan(self, request: PlanningRequest) -> ActionProposal:
        """Return one locally validated next-action proposal."""
        proposal_id = self._identifier_factory()
        model_request = self._build_model_request(request, proposal_id)
        response = self._model_client.complete(
            model_request,
            ActionProposal,
            validation_context={
                "expected_proposal_id": proposal_id,
                "expected_observation_id": request.observation.observation_id,
                "available_actions": tuple(
                    sorted(action.value for action in _available_actions(request))
                ),
                "declared_value_names": request.declared_value_names,
            },
        )
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
            "screen_facts": [planning_screen_fact_context(fact) for fact in request.screen_facts],
            "open_work": request.open_work,
            "completion_conditions": request.completion_conditions,
            "declared_value_names": request.declared_value_names,
            "available_actions": sorted(action.value for action in _available_actions(request)),
            "current_observation": _observation_context(request.observation),
            "earlier_history_summary": planning_history_summary(earlier_history),
            "history": [planning_history_context(entry) for entry in recent_history],
            "progress_context": (
                request.progress_context.model_dump(mode="json")
                if request.progress_context is not None
                else None
            ),
            "discovery_mode": request.discovery_mode,
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
            max_completion_tokens=PLANNER_MAX_COMPLETION_TOKENS,
            sensitivity=request.observation.redacted_screenshot.sensitivity,
            retention=request.observation.redacted_screenshot.retention,
        )


def plan_or_stop(
    source: ActionProposalSource,
    request: PlanningRequest,
    identifier_factory: Callable[[], str],
) -> tuple[ActionProposal, bool]:
    """Convert an unusable model decision into a safe recorded terminal."""
    try:
        return source.plan(request), False
    except TaskPlanningError as error:
        return (
            ActionProposal(
                proposal_id=identifier_factory(),
                source_observation_id=request.observation.observation_id,
                action=ActionKind.CANNOT_PROGRESS,
                reason=f"Planning stopped: {error}",
            ),
            True,
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
