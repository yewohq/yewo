"""Select one semantic next action from current task evidence."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
from typing import Protocol, runtime_checkable

from pydantic import ValidationError

from yewo.artifacts import new_record_identifier
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
from yewo.agent.planning.evidence import first_source_of
from yewo.agent.planning.history import (
    correctable_grounding_proposals,
    trusted_history_content,
    EARLIER_HISTORY_SUMMARY_CHARACTER_LIMIT as EARLIER_HISTORY_SUMMARY_CHARACTER_LIMIT,
    EARLIER_HISTORY_SUMMARY_LIMIT as EARLIER_HISTORY_SUMMARY_LIMIT,
    GroundedTargetPoint as GroundedTargetPoint,
    PlanningHistoryEntry,
    planning_screen_fact_context,
)
from yewo.agent.planning.context import planning_observation_context
from yewo.agent.planning.prompt import (
    planner_action_catalog,
    TASK_PLANNER_FOCUS_USER_PROMPT,
    TASK_PLANNER_SYSTEM_PROMPT,
    TASK_PLANNER_USER_PROMPT,
)
from yewo.agent.screen_analysis import (
    interactive_element_references,
    screen_analysis_context,
)
from yewo.agent.actuation import (
    ACTION_DEFINITIONS,
    ActionHandling,
    ActionKind,
    ActionProposal,
    ActionProposalDecision,
    DeclaredValueReference,
)
from yewo.agent.planning.records import (
    PlanningRequest,
    planning_progress_context,
    planning_focus_context,
)

PLANNER_MAX_COMPLETION_TOKENS = 8_192

TERMINAL_ACTIONS = frozenset(
    action
    for action, definition in ACTION_DEFINITIONS.items()
    if definition.handling is ActionHandling.TERMINAL
)
# The planner prompt carries many independent constraints, so one rejected
# decision is weak evidence that the model cannot serve the request at all.
PLANNING_ATTEMPT_LIMIT = 2
# A transport that never returned a usable decision has already retried itself.
RECOVERABLE_PLANNING_DISPOSITIONS = frozenset(
    {ModelCallDisposition.COMPLETED, ModelCallDisposition.MALFORMED}
)


# A provider that returned no answer has said nothing about the task. Yewo already retries once
# inside the transport, so a second empty answer still describes the provider.
_NO_ANSWER = frozenset(
    {
        ModelCallDisposition.EMPTY,
        ModelCallDisposition.MALFORMED,
        ModelCallDisposition.PROVIDER_ERROR,
        ModelCallDisposition.REASONING_EXHAUSTED,
        ModelCallDisposition.TRUNCATED,
    }
)


class TaskPlanningError(RuntimeError):
    """A model call did not produce one usable action proposal."""

    def __init__(self, response: ModelResponse, reason: str, *, no_answer: bool = False) -> None:
        super().__init__(reason)
        self.response = response
        self.disposition = response.disposition
        self.no_answer = no_answer


class ActionProposalSource(Protocol):
    """Supply one validated semantic action for the current agent evidence."""

    def required_model_requests(self, request: PlanningRequest) -> int:
        """Return the model requests needed for the next decision."""
        ...

    def plan(self, request: PlanningRequest) -> ActionProposal: ...


@dataclass(frozen=True)
class ActionVerificationContext:
    """Task evidence for one model-selected action."""

    request: PlanningRequest
    assess_task_progress: bool = True


@runtime_checkable
class ActionVerificationContextSource(Protocol):
    """Override verification evidence for a delegated planning mode."""

    def action_verification_context(
        self,
        default_request: PlanningRequest,
    ) -> ActionVerificationContext: ...


@runtime_checkable
class PlanningFocusDecliningSource(Protocol):
    """Decline the isolated focus view while a delegated planning mode owns the request."""

    def declines_planning_focus(self) -> bool: ...


def source_accepts_planning_focus(source: object) -> bool:
    """Report whether the active source can use an isolated focus view."""
    decision_source = first_source_of(source, PlanningFocusDecliningSource)
    if decision_source is not None:
        return not decision_source.declines_planning_focus()
    return True


def resolve_action_verification_context(
    source: object,
    default_request: PlanningRequest,
) -> ActionVerificationContext:
    """Use delegated verification evidence when the active source supplies it."""
    decision_source = first_source_of(source, ActionVerificationContextSource)
    if decision_source is not None:
        return decision_source.action_verification_context(default_request)
    return ActionVerificationContext(request=default_request)


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
        self._planning_model_request_count = 0

    def required_model_requests(self, request: PlanningRequest) -> int:
        """Reserve one model request for each permitted planning attempt."""
        del request
        return PLANNING_ATTEMPT_LIMIT

    def last_planning_model_request_count(self) -> int:
        """Return actual model calls consumed by the most recent plan operation."""
        return self._planning_model_request_count

    def plan(self, request: PlanningRequest) -> ActionProposal:
        """Return one validated proposal, resampling a rejected model decision."""
        self._planning_model_request_count = 0
        first_failure: TaskPlanningError | None = None
        while True:
            self._planning_model_request_count += 1
            try:
                return self._plan_once(request)
            except TaskPlanningError as error:
                if first_failure is None:
                    first_failure = error
                if error.disposition not in RECOVERABLE_PLANNING_DISPOSITIONS:
                    raise
                if self._planning_model_request_count >= PLANNING_ATTEMPT_LIMIT:
                    # A later attempt can fail for a derived reason, so the first
                    # rejection stays the reported cause of the stopped decision.
                    raise first_failure from error

    def _plan_once(self, request: PlanningRequest) -> ActionProposal:
        """Ask the planner model once and validate the returned proposal."""
        proposal_id = self._identifier_factory()
        model_request = self._build_model_request(request)
        validation_context = {
            "available_actions": tuple(
                sorted(action.value for action in _available_actions(request))
            ),
            "declared_value_names": request.declared_value_names,
            "correctable_grounding_proposals": correctable_grounding_proposals(request.history),
            "available_element_refs": interactive_element_references(
                request.current_screen_analysis
            ),
        }
        response = self._model_client.complete(
            model_request,
            ActionProposalDecision,
            validation_context=validation_context,
        )
        if response.disposition is not ModelCallDisposition.COMPLETED:
            raise TaskPlanningError(
                response,
                response.reason or "The task-planner model call did not complete.",
                no_answer=response.disposition in _NO_ANSWER,
            )
        if response.structured_output is None:
            raise TaskPlanningError(
                response,
                "The task-planner response has no structured output.",
                no_answer=True,
            )
        try:
            decision = ActionProposalDecision.model_validate(
                response.structured_output,
                context={"available_element_refs": validation_context["available_element_refs"]},
            )
        except ValidationError as error:
            raise TaskPlanningError(
                response,
                "The task-planner response does not contain a valid action proposal.",
            ) from error
        proposal = ActionProposal(
            proposal_id=proposal_id,
            source_observation_id=request.observation.observation_id,
            **decision.model_dump(),
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
    ) -> ModelRequest:
        task_context = {
            "objective": request.objective,
            "known_facts": request.known_facts,
            "completion_conditions": request.completion_conditions,
            "declared_value_names": request.declared_value_names,
            "discovery_mode": request.discovery_mode,
        }
        context = {
            "open_work": request.open_work,
            "available_actions": planner_action_catalog(_available_actions(request)),
            "screen_facts": [planning_screen_fact_context(fact) for fact in request.screen_facts],
            "current_observation": planning_observation_context(request.observation),
            "current_screen_analysis": (
                screen_analysis_context(request.current_screen_analysis)
                if request.current_screen_analysis is not None
                else None
            ),
            "progress_context": (
                planning_progress_context(request.progress_context)
                if request.progress_context is not None
                else None
            ),
        }
        task_prompt = "Task context:\n" + json.dumps(
            task_context,
            sort_keys=True,
            separators=(",", ":"),
        )
        if request.planning_focus is not None:
            messages = _focus_messages(request, task_prompt, context)
        else:
            messages = _standard_messages(request, task_prompt, context)
        return ModelRequest(
            request_id=self._identifier_factory(),
            profile_id=self._model_profile_id,
            role=ModelRole.TASK_PLANNER,
            messages=messages,
            response_contract=build_model_response_contract(
                "action-proposal-v1",
                ActionProposalDecision.model_json_schema(),
            ),
            max_completion_tokens=PLANNER_MAX_COMPLETION_TOKENS,
            retention=request.observation.screenshot.retention,
        )


def _standard_messages(
    request: PlanningRequest,
    task_prompt: str,
    context: dict[str, object],
) -> tuple[ModelMessage, ...]:
    current_prompt = "Current decision context:\n" + json.dumps(
        context,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        ModelMessage(
            role=ModelMessageRole.SYSTEM,
            content=(TextModelContent(text=TASK_PLANNER_SYSTEM_PROMPT),),
        ),
        ModelMessage(
            role=ModelMessageRole.USER,
            content=(TextModelContent(text=TASK_PLANNER_USER_PROMPT),),
            cache_prefix_end=True,
        ),
        ModelMessage(
            role=ModelMessageRole.USER,
            content=(
                *trusted_history_content(request.history),
                TextModelContent(text=task_prompt),
                TextModelContent(text=current_prompt),
                ImageModelContent(screenshot=request.observation.screenshot),
                *(
                    (
                        TextModelContent(
                            text=(
                                "Compiled target screenshot. Restore the application to this state."
                            )
                        ),
                        ImageModelContent(screenshot=request.reference_screenshot),
                    )
                    if request.reference_screenshot is not None
                    else ()
                ),
            ),
        ),
    )


def _focus_messages(
    request: PlanningRequest,
    task_prompt: str,
    context: dict[str, object],
) -> tuple[ModelMessage, ...]:
    focus_context = request.planning_focus
    assert focus_context is not None
    current_context = {
        "available_actions": context["available_actions"],
        "current_observation": context["current_observation"],
        "current_screen_analysis": context["current_screen_analysis"],
        "focus_context": planning_focus_context(focus_context),
        "screen_facts": context["screen_facts"],
        "completed_work_items": (
            request.progress_context.completed_work_items
            if request.progress_context is not None
            else ()
        ),
    }
    return (
        ModelMessage(
            role=ModelMessageRole.SYSTEM,
            content=(TextModelContent(text=TASK_PLANNER_SYSTEM_PROMPT),),
        ),
        ModelMessage(
            role=ModelMessageRole.USER,
            content=(TextModelContent(text=TASK_PLANNER_FOCUS_USER_PROMPT),),
            cache_prefix_end=True,
        ),
        ModelMessage(
            role=ModelMessageRole.USER,
            content=(
                TextModelContent(text=task_prompt),
                TextModelContent(
                    text="Focus decision context:\n"
                    + json.dumps(current_context, sort_keys=True, separators=(",", ":"))
                ),
                ImageModelContent(screenshot=request.observation.screenshot),
            ),
        ),
    )


def plan_or_stop(
    source: ActionProposalSource,
    request: PlanningRequest,
    identifier_factory: Callable[[], str],
) -> tuple[ActionProposal, bool]:
    """Convert an unusable model decision into a safe recorded terminal.

    A planner that returned no answer leaves the step without a decision. It does not establish
    that the task cannot progress, so the run waits and asks again rather than concluding from
    the provider's silence. The step budget bounds how long that can continue.
    """
    try:
        return source.plan(request), False
    except TaskPlanningError as error:
        if error.no_answer:
            return (
                ActionProposal(
                    proposal_id=identifier_factory(),
                    source_observation_id=request.observation.observation_id,
                    action=ActionKind.WAIT,
                    duration_seconds=1,
                    expected_outcome="The screen is unchanged and the planner is asked again.",
                    reason=f"The planner returned no decision: {error}",
                ),
                True,
            )
        return (
            ActionProposal(
                proposal_id=identifier_factory(),
                source_observation_id=request.observation.observation_id,
                action=ActionKind.CANNOT_PROGRESS,
                reason=f"Planning stopped: {error}",
            ),
            True,
        )


def _available_actions(request: PlanningRequest) -> frozenset[ActionKind]:
    return request.available_actions | TERMINAL_ACTIONS


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
