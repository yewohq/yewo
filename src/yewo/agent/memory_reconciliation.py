"""Reconcile retained app memory with the current run before free planning."""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
import json

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from yewo.agent.app_memory import AppMemory, AcceptedTransition
from yewo.agent.actuation import ActionProposal
from yewo.agent.planner import (
    ActionProposalSource,
    PlanningRequest,
    PlanningText,
    TaskProgressContext,
    TaskProgressState,
)
from yewo.agent.planning_evidence import (
    PlannerDecisionMetadata,
    resolve_planner_decision_metadata,
)
from yewo.artifacts import RecordIdentifier, new_record_identifier
from yewo.models import (
    ImageModelContent,
    ModelCallDisposition,
    ModelClient,
    ModelMessage,
    ModelMessageRole,
    ModelRequest,
    ModelRole,
    TextModelContent,
    build_model_response_contract,
)
from yewo.run_narration import current_narrator


MEMORY_RECONCILIATION_SYSTEM_PROMPT = """\
You are Yewo's task-state reconciler. Before the GUI planner acts, compare the current screenshot
with retained facts from earlier runs. Your output removes work only from this planner invocation.
It does not change application memory, the original objective, or the final success criteria.

## Evidence rules

- Treat the current screenshot as the only evidence of what is true in the application now.
- A retained transition records an observed source-to-destination change from an earlier run.
  Treat it as a reusable procedure. It does not prove that the change already happened now.
- Treat a retained capability as verified prior application behavior, not current task completion.
- Treat an unsuccessful attempt as attempt history, not proof that a control or behavior is absent.
- Expected outcomes, target descriptions, and planner claims are intentions. They are not facts.
- Cite only supplied fact IDs. Do not infer hidden values, off-screen content, or persisted state.

## Classify each work item

- current_progress: the current screenshot directly proves the whole work item is true now. Cite
  the current_observation fact. If any required part is hidden, ambiguous, or only remembered,
  do not use current_progress.
- known_procedure: retained facts show a relevant route or action, but the current screenshot does
  not prove that the work item is complete now.
- unresolved: the evidence neither proves current progress nor supplies a relevant known procedure.

Classify work items independently. A visible parent record does not prove that a required nested
record, setting, attachment, or child item exists. A confirmation action can support a known
procedure when it returned to a parent collection. The current screenshot must visibly show the
named result before it can establish current progress. A required nested item can remain unresolved.
If the current screen offers an Add, Create, or equivalent control for that item, use this evidence.
Do not mistake a Finish, Save, Done, or checkmark control for creating the nested item.

Use exhausted_branch facts only when supplied evidence explicitly identifies a failed route. A
successful transition observed in more than one run is reusable evidence, not an exhausted branch.
Keep summaries short, factual, and specific to the corresponding work item. Do not propose the next
action, rewrite the objective, declare overall success, or invent facts.

## Select the replay frontier

Select one transition fact as resume_transition_fact_id when retained evidence contains a
successful transition whose destination is the best place to continue the unresolved work. Select
the latest useful frontier, not every action that previously occurred. Yewo will construct the
shortest validated path from the fresh application screen to that destination. Do not select a
transition merely because it occurred often. Do not select a branch contradicted by an unsuccessful
attempt. Return null when retained transitions do not provide a useful frontier.
"""

MEMORY_RECONCILIATION_USER_PROMPT = """\
Return one work_items entry for every supplied zero-based work index. Cite only supplied fact IDs.
Use relevant_fact_ids for the small set of prior facts that will help the task planner. Use
exhausted_branch_fact_ids only for facts whose kind is exhausted_branch.
Set resume_transition_fact_id to one supplied transition fact or null.
Context JSON follows:
"""

RECONCILIATION_FACT_LIMIT = 60
RECONCILIATION_ATTEMPT_LIMIT = 10
RECONCILIATION_CAPABILITY_LIMIT = 10
RECONCILIATION_MAX_COMPLETION_TOKENS = 8_192


class ReconciliationFactKind(StrEnum):
    """The evidence role of one bounded reconciliation fact."""

    CURRENT_OBSERVATION = "current_observation"
    CAPABILITY = "capability"
    TRANSITION = "transition"
    UNSUCCESSFUL_ATTEMPT = "unsuccessful_attempt"
    EXHAUSTED_BRANCH = "exhausted_branch"


class ReconciliationFact(BaseModel):
    """One identifier-addressable fact supplied to reconciliation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fact_id: RecordIdentifier
    kind: ReconciliationFactKind
    statement: PlanningText
    transition_id: RecordIdentifier | None = Field(default=None, exclude=True)


class ReconciledWorkItem(BaseModel):
    """The run-scoped state of one original work item."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    work_index: int = Field(ge=0)
    state: TaskProgressState
    evidence_fact_ids: tuple[RecordIdentifier, ...] = Field(default=(), max_length=20)
    summary: PlanningText


class MemoryReconciliationOutput(BaseModel):
    """Structured progress selected from supplied memory facts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    work_items: tuple[ReconciledWorkItem, ...]
    relevant_fact_ids: tuple[RecordIdentifier, ...] = Field(default=(), max_length=20)
    exhausted_branch_fact_ids: tuple[RecordIdentifier, ...] = Field(default=(), max_length=20)
    resume_transition_fact_id: RecordIdentifier | None

    @model_validator(mode="after")
    def _validate_unique_references(self) -> MemoryReconciliationOutput:
        indexes = [item.work_index for item in self.work_items]
        if len(indexes) != len(set(indexes)):
            raise ValueError("Reconciliation work indexes must be unique.")
        if len(self.relevant_fact_ids) != len(set(self.relevant_fact_ids)):
            raise ValueError("Reconciliation relevant facts must be unique.")
        if len(self.exhausted_branch_fact_ids) != len(set(self.exhausted_branch_fact_ids)):
            raise ValueError("Reconciliation exhausted branches must be unique.")
        return self


class MemoryReconcilerActionSource:
    """Perform one bounded run-scoped reconciliation before free planning."""

    def __init__(
        self,
        source: ActionProposalSource,
        model_client: ModelClient,
        model_profile_id: str,
        memory: AppMemory,
        objective_id: RecordIdentifier,
        *,
        identifier_factory: Callable[[], str] = new_record_identifier,
    ) -> None:
        self._source = source
        self._model_client = model_client
        self._model_profile_id = model_profile_id
        self._memory = memory
        self._objective_id = objective_id
        self._identifier_factory = identifier_factory
        self._attempted = False
        self._progress_context: TaskProgressContext | None = None

    def required_model_requests(self, request: PlanningRequest) -> int:
        """Count reconciliation once, followed by the delegated planner call."""
        return self._source.required_model_requests(request) + int(not self._attempted)

    def decision_metadata(self) -> PlannerDecisionMetadata:
        """Preserve the evidence provenance selected by the delegated source."""
        return resolve_planner_decision_metadata(self._source)

    def plan(self, request: PlanningRequest) -> ActionProposal:
        """Reconcile once and delegate with validated run-scoped progress."""
        if not self._attempted:
            self._attempted = True
            self._progress_context = self._reconcile(request)
        context = self._progress_context
        if context is None:
            return self._source.plan(request)
        return self._source.plan(
            request.model_copy(
                update={
                    "open_work": context.remaining_work,
                    "progress_context": context,
                }
            )
        )

    def _reconcile(self, request: PlanningRequest) -> TaskProgressContext | None:
        facts = _reconciliation_facts(
            self._memory,
            request,
            self._objective_id,
        )
        fact_by_id = {fact.fact_id: fact for fact in facts}
        model_request = _model_request(
            request,
            facts,
            self._model_profile_id,
            self._identifier_factory(),
        )
        response = self._model_client.complete(model_request, MemoryReconciliationOutput)
        if (
            response.disposition is not ModelCallDisposition.COMPLETED
            or response.structured_output is None
        ):
            _narrate_reconciliation("fallback", len(facts), response.reason or response.disposition)
            return None
        try:
            output = MemoryReconciliationOutput.model_validate(response.structured_output)
            context = _validated_progress_context(request, output, fact_by_id)
        except (ValidationError, ValueError) as error:
            _narrate_reconciliation("fallback", len(facts), type(error).__name__)
            return None
        _narrate_reconciliation(
            "applied",
            len(context.historical_memory_facts),
            f"remaining={len(context.remaining_work)}",
        )
        return context


def _model_request(
    request: PlanningRequest,
    facts: tuple[ReconciliationFact, ...],
    profile_id: str,
    request_id: str,
) -> ModelRequest:
    context = {
        "objective": request.objective,
        "open_work": tuple(enumerate(request.open_work)),
        "completion_conditions": request.completion_conditions,
        "facts": [fact.model_dump(mode="json") for fact in facts],
    }
    return ModelRequest(
        request_id=request_id,
        profile_id=profile_id,
        role=ModelRole.TASK_PLANNER,
        max_completion_tokens=RECONCILIATION_MAX_COMPLETION_TOKENS,
        messages=(
            ModelMessage(
                role=ModelMessageRole.SYSTEM,
                content=(TextModelContent(text=MEMORY_RECONCILIATION_SYSTEM_PROMPT),),
            ),
            ModelMessage(
                role=ModelMessageRole.USER,
                content=(
                    TextModelContent(
                        text=MEMORY_RECONCILIATION_USER_PROMPT
                        + json.dumps(context, sort_keys=True, separators=(",", ":"))
                    ),
                    ImageModelContent(screenshot=request.observation.redacted_screenshot),
                ),
            ),
        ),
        response_contract=build_model_response_contract(
            "memory-reconciliation-v1",
            MemoryReconciliationOutput.model_json_schema(),
        ),
        sensitivity=request.observation.redacted_screenshot.sensitivity,
        retention=request.observation.redacted_screenshot.retention,
    )


def _validated_progress_context(
    request: PlanningRequest,
    output: MemoryReconciliationOutput,
    fact_by_id: dict[str, ReconciliationFact],
) -> TaskProgressContext:
    if {item.work_index for item in output.work_items} != set(range(len(request.open_work))):
        raise ValueError("Reconciliation must classify every open work item exactly once.")
    referenced_ids = {
        *output.relevant_fact_ids,
        *output.exhausted_branch_fact_ids,
        *(fact_id for item in output.work_items for fact_id in item.evidence_fact_ids),
    }
    if output.resume_transition_fact_id is not None:
        referenced_ids.add(output.resume_transition_fact_id)
    if not referenced_ids.issubset(fact_by_id):
        raise ValueError("Reconciliation referenced an unknown fact.")
    if any(
        fact_by_id[fact_id].kind is not ReconciliationFactKind.EXHAUSTED_BRANCH
        for fact_id in output.exhausted_branch_fact_ids
    ):
        raise ValueError("Only exhausted-branch facts may be marked exhausted.")
    replay_target_transition_id = None
    if output.resume_transition_fact_id is not None:
        resume_fact = fact_by_id[output.resume_transition_fact_id]
        if (
            resume_fact.kind is not ReconciliationFactKind.TRANSITION
            or resume_fact.transition_id is None
        ):
            raise ValueError("A replay resume target must reference a transition fact.")
        replay_target_transition_id = resume_fact.transition_id
    current_ids = {
        fact.fact_id
        for fact in fact_by_id.values()
        if fact.kind is ReconciliationFactKind.CURRENT_OBSERVATION
    }
    for item in output.work_items:
        if item.state is TaskProgressState.CURRENT_PROGRESS and not (
            set(item.evidence_fact_ids) & current_ids
        ):
            raise ValueError("Current progress requires current-observation evidence.")
    ordered = sorted(output.work_items, key=lambda item: item.work_index)
    established = tuple(
        item.summary for item in ordered if item.state is TaskProgressState.CURRENT_PROGRESS
    )
    partial = tuple(
        item.summary for item in ordered if item.state is TaskProgressState.KNOWN_PROCEDURE
    )
    remaining = tuple(
        request.open_work[item.work_index]
        for item in ordered
        if item.state is not TaskProgressState.CURRENT_PROGRESS
    )
    return TaskProgressContext(
        current_progress=established,
        known_procedures=partial,
        remaining_work=remaining,
        historical_memory_facts=tuple(
            fact_by_id[fact_id].statement for fact_id in output.relevant_fact_ids
        ),
        exhausted_branches=tuple(
            fact_by_id[fact_id].statement for fact_id in output.exhausted_branch_fact_ids
        ),
        replay_target_transition_id=replay_target_transition_id,
    )


def _reconciliation_facts(
    memory: AppMemory,
    request: PlanningRequest,
    objective_id: str,
) -> tuple[ReconciliationFact, ...]:
    facts = [
        ReconciliationFact(
            fact_id="current_observation",
            kind=ReconciliationFactKind.CURRENT_OBSERVATION,
            statement=(
                "Current screenshot and observation: platform context "
                f"{request.observation.platform_context or 'unknown'}. Keyboard visible: "
                f"{request.observation.keyboard_visible}."
            ),
        )
    ]
    screen_context_by_id = {
        screen.screen_id: screen.variants[-1].platform_context or "unknown screen"
        for screen in memory.screens
    }
    for index, attempt in enumerate(
        reversed(memory.unsuccessful_attempts[-RECONCILIATION_ATTEMPT_LIMIT:])
    ):
        facts.append(
            ReconciliationFact(
                fact_id=f"attempt_{index}",
                kind=ReconciliationFactKind.UNSUCCESSFUL_ATTEMPT,
                statement=(
                    f"Prior {attempt.action.value} attempt on "
                    f"{screen_context_by_id.get(attempt.screen_id, 'unknown screen')} targeting "
                    f"{attempt.target_description or 'the screen'} ended as "
                    f"{attempt.disposition.value}: {attempt.reason}"
                ),
            )
        )
    capability_facts = tuple(
        ReconciliationFact(
            fact_id=f"capability_{index}",
            kind=ReconciliationFactKind.CAPABILITY,
            statement=capability.statement,
        )
        for index, capability in enumerate(memory.capabilities)
        if capability.objective_id == objective_id
    )
    facts.extend(capability_facts[-RECONCILIATION_CAPABILITY_LIMIT:])
    selected = _selected_transitions(memory, objective_id)
    seen_transition_keys: set[tuple[str, ...]] = set()
    for index, transition in enumerate(selected):
        if len(facts) >= RECONCILIATION_FACT_LIMIT:
            break
        key = _transition_key(transition)
        if key in seen_transition_keys:
            continue
        seen_transition_keys.add(key)
        facts.append(
            ReconciliationFact(
                fact_id=f"transition_{index}",
                kind=ReconciliationFactKind.TRANSITION,
                statement=_transition_fact(memory, transition),
                transition_id=transition.transition_id,
            )
        )
    return tuple(facts)


def _selected_transitions(memory: AppMemory, objective_id: str) -> tuple[AcceptedTransition, ...]:
    relevant_ids = {
        segment.transition_id
        for route in memory.routes
        if route.objective_id == objective_id
        for segment in route.segments
    }
    relevant_ids.update(
        transition_id
        for checkpoint in memory.checkpoints
        if checkpoint.objective_id == objective_id
        for transition_id in checkpoint.transition_ids
    )
    return tuple(
        transition
        for transition in memory.transitions
        if transition.transition_id in relevant_ids
    )


def _transition_key(transition: AcceptedTransition) -> tuple[str, ...]:
    return (
        transition.source_screen_id,
        transition.action.action.value,
        transition.action.target_description or "",
        transition.destination_screen_id,
    )


def _transition_fact(memory: AppMemory, transition: AcceptedTransition) -> str:
    screens = {screen.screen_id: screen for screen in memory.screens}
    source = screens[transition.source_screen_id].variants[-1].platform_context or "unknown screen"
    destination = (
        screens[transition.destination_screen_id].variants[-1].platform_context or "unknown screen"
    )
    signals = ", ".join(sorted(signal.value for signal in transition.outcome_signals))
    return (
        f"Prior observed transition: {transition.action.action.value} targeting "
        f"{transition.action.target_description or 'the current screen'} moved from {source} to "
        f"{destination}. Observed signals: {signals}. This proves prior behavior, not current "
        "completion."
    )


def _narrate_reconciliation(disposition: str, fact_count: int, detail: object) -> None:
    narrator = current_narrator()
    if narrator is None:
        return
    if disposition == "applied":
        narrator.record(
            "planner-memory",
            "Reconciled retained app memory with the current task state.",
        )
    else:
        narrator.record(
            "planner-memory",
            "Memory reconciliation was unavailable. Continuing with the original task context.",
        )
    narrator.trace(
        "planner-memory",
        f"reconciliation={disposition} facts={fact_count} detail={detail}",
    )
