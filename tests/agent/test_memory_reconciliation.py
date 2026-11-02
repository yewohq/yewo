"""Run-scoped reconciliation of retained memory and current task state."""

from datetime import datetime, timezone
import json

from yewo.agent import AppMemory, MemoryCompatibility
from yewo.agent.actuation import ActionKind, ActionProposal
from yewo.agent.app_memory import UnsuccessfulAttempt, UnsuccessfulAttemptDisposition
from yewo.agent.discovery import AppMemoryBuilder, DiscoveryObjective, DiscoveryRequest
from yewo.agent.memory_reconciliation import (
    MEMORY_RECONCILIATION_SYSTEM_PROMPT,
    MemoryReconcilerActionSource,
)
from yewo.agent.planner import PlanningRequest
from yewo.agent.planning_evidence import PlannerDecisionMetadata, PlannerDecisionSource
from yewo.models import ImageModelContent, TextModelContent
from yewo.observations import Observation

from agent.actuation.support import proposal
from agent.discovery.support import agent_result, observation
from agent.planner_support import StubPlannerModel


class RecordingSource:
    """Record the planner request after reconciliation."""

    def __init__(self) -> None:
        self.requests: list[PlanningRequest] = []

    def required_model_requests(self, request: PlanningRequest) -> int:
        """Represent one delegated planner call."""
        del request
        return 1

    def decision_metadata(self) -> PlannerDecisionMetadata:
        """Return retained-path evidence through transparent planner decorators."""
        return PlannerDecisionMetadata(
            source=PlannerDecisionSource.DISCOVERY_CHECKPOINT,
            reused_flow_id="checkpoint_1",
            reused_transition_ids=("transition_0",),
            reused_segment_index=0,
        )

    def plan(self, request: PlanningRequest) -> ActionProposal:
        self.requests.append(request)
        return proposal(
            ActionKind.DONE,
            source_observation_id=request.observation.observation_id,
        )


def _memory() -> tuple[AppMemory, Observation]:
    before = observation("obs_1", "0000000000000000")
    after = observation(
        "obs_2",
        "1111111111111111",
        predecessor_observation_id=before.observation_id,
        offset_seconds=1,
    )
    promotion = AppMemoryBuilder().promote(
        DiscoveryRequest(
            request_id="request_1",
            objectives=(
                DiscoveryObjective(
                    objective_id="schedule",
                    statement="A schedule is saved.",
                ),
            ),
        ),
        agent_result(before, after),
        {before.observation_id: before, after.observation_id: after},
        MemoryCompatibility(application=before.application),
        verified_objective_ids=frozenset({"schedule"}),
        source_run_id="run_1",
        updated_at=datetime(2026, 10, 31, tzinfo=timezone.utc),
    )
    assert promotion.memory is not None
    return promotion.memory, after


def _request(current: Observation) -> PlanningRequest:
    return PlanningRequest(
        observation=current,
        objective="Save a schedule with one exercise.",
        open_work=("Save the schedule.", "Add one exercise."),
        completion_conditions=("The saved schedule contains one exercise.",),
        available_actions=frozenset({ActionKind.DONE}),
    )


def test_reconciliation_filters_only_currently_proven_work() -> None:
    memory, current = _memory()
    unrelated = memory.transitions[0].model_copy(
        update={
            "transition_id": "unrelated_transition",
            "action": memory.transitions[0].action.model_copy(
                update={"target_description": "an unrelated historical control"}
            ),
        }
    )
    memory = AppMemory.model_validate(
        {
            **memory.model_dump(mode="python"),
            "transitions": (*memory.transitions, unrelated),
        }
    )
    model = StubPlannerModel(
        {
            "work_items": [
                {
                    "work_index": 0,
                    "state": "current_progress",
                    "evidence_fact_ids": ["current_observation"],
                    "summary": "The current screen visibly shows the saved schedule.",
                },
                {
                    "work_index": 1,
                    "state": "known_procedure",
                    "evidence_fact_ids": ["transition_0"],
                    "summary": "Memory contains a relevant prior transition.",
                },
            ],
            "relevant_fact_ids": ["transition_0"],
            "exhausted_branch_fact_ids": [],
            "resume_transition_fact_id": "transition_0",
        }
    )
    source = RecordingSource()
    identifiers = iter(("reconciliation_request",))
    reconciler = MemoryReconcilerActionSource(
        source,
        model,
        "planner-primary",
        memory,
        "schedule",
        identifier_factory=lambda: next(identifiers),
    )
    request = _request(current)

    assert reconciler.required_model_requests(request) == 2
    reconciler.plan(request)

    assert reconciler.required_model_requests(request) == 1
    assert reconciler.decision_metadata() == source.decision_metadata()
    assert len(model.requests) == 1
    model_request = model.requests[0]
    assert model_request.response_contract is not None
    assert model_request.response_contract.name == "memory-reconciliation-v1"
    assert model_request.max_completion_tokens == 8_192
    system_content = model_request.messages[0].content[0]
    assert isinstance(system_content, TextModelContent)
    assert system_content.text == MEMORY_RECONCILIATION_SYSTEM_PROMPT
    assert "only evidence of what is true in the application now" in (
        MEMORY_RECONCILIATION_SYSTEM_PROMPT
    )
    assert "visible parent record does not prove" in MEMORY_RECONCILIATION_SYSTEM_PROMPT
    assert "do not use current_progress" in MEMORY_RECONCILIATION_SYSTEM_PROMPT
    text_part, image_part = model_request.messages[1].content
    assert text_part.kind == "text"
    assert isinstance(image_part, ImageModelContent)
    supplied = json.loads(text_part.text.split("Context JSON follows:\n", 1)[1])
    transition_facts = [fact for fact in supplied["facts"] if fact["kind"] == "transition"]
    assert len(transition_facts) == 1
    assert "unrelated historical control" not in transition_facts[0]["statement"]
    delegated = source.requests[0]
    assert delegated.open_work == ("Add one exercise.",)
    assert delegated.progress_context is not None
    assert delegated.progress_context.current_progress == (
        "The current screen visibly shows the saved schedule.",
    )
    assert "Prior observed transition" in (
        delegated.progress_context.historical_memory_facts[0]
    )
    assert (
        delegated.progress_context.replay_target_transition_id
        == memory.transitions[0].transition_id
    )


def test_invalid_reconciliation_preserves_original_planner_context() -> None:
    memory, current = _memory()
    model = StubPlannerModel(
        {
            "work_items": [
                {
                    "work_index": 0,
                    "state": "current_progress",
                    "evidence_fact_ids": ["unknown_fact"],
                    "summary": "Unsupported claim.",
                }
            ],
            "relevant_fact_ids": [],
            "exhausted_branch_fact_ids": [],
            "resume_transition_fact_id": None,
        }
    )
    source = RecordingSource()
    reconciler = MemoryReconcilerActionSource(
        source,
        model,
        "planner-primary",
        memory,
        "schedule",
        identifier_factory=lambda: "reconciliation_request",
    )
    request = _request(current)

    reconciler.plan(request)

    assert source.requests[0] == request


def test_reconciliation_receives_failures_from_an_earlier_screen() -> None:
    memory, current = _memory()
    earlier_screen = next(
        screen
        for screen in memory.screens
        if any(variant.observation_id == "obs_1" for variant in screen.variants)
    )
    attempt = UnsuccessfulAttempt(
        attempt_id="attempt_1",
        screen_id=earlier_screen.screen_id,
        source_observation_id="obs_1",
        action=ActionKind.TAP,
        target_description="the repeated floating action button",
        disposition=UnsuccessfulAttemptDisposition.NO_OBSERVABLE_CHANGE,
        reason="The screen did not visibly change.",
        source_step_id="step_1",
        source_run_id="run_1",
    )
    memory = memory.model_copy(update={"unsuccessful_attempts": (attempt,)})
    model = StubPlannerModel(
        {
            "work_items": [
                {
                    "work_index": index,
                    "state": "unresolved",
                    "evidence_fact_ids": [],
                    "summary": "The work remains unresolved.",
                }
                for index in range(2)
            ],
            "relevant_fact_ids": ["attempt_0"],
            "exhausted_branch_fact_ids": [],
            "resume_transition_fact_id": None,
        }
    )
    reconciler = MemoryReconcilerActionSource(
        RecordingSource(),
        model,
        "planner-primary",
        memory,
        "schedule",
        identifier_factory=lambda: "reconciliation_request",
    )

    reconciler.plan(_request(current))

    text_part = model.requests[0].messages[1].content[0]
    assert isinstance(text_part, TextModelContent)
    supplied = json.loads(text_part.text.split("Context JSON follows:\n", 1)[1])
    attempts = [fact for fact in supplied["facts"] if fact["kind"] == "unsuccessful_attempt"]
    assert len(attempts) == 1
    assert "repeated floating action button" in attempts[0]["statement"]


def test_reconciliation_collapses_repeated_successful_transition_facts() -> None:
    memory, current = _memory()
    duplicate = memory.transitions[0].model_copy(update={"transition_id": "transition_copy"})
    memory = memory.model_copy(update={"transitions": (*memory.transitions, duplicate)})
    model = StubPlannerModel(
        {
            "work_items": [
                {
                    "work_index": index,
                    "state": "known_procedure",
                    "evidence_fact_ids": ["transition_0"],
                    "summary": "The retained transition provides a known procedure.",
                }
                for index in range(2)
            ],
            "relevant_fact_ids": ["transition_0"],
            "exhausted_branch_fact_ids": [],
            "resume_transition_fact_id": "transition_0",
        }
    )
    reconciler = MemoryReconcilerActionSource(
        RecordingSource(),
        model,
        "planner-primary",
        memory,
        "schedule",
        identifier_factory=lambda: "reconciliation_request",
    )

    reconciler.plan(_request(current))

    text_part = model.requests[0].messages[1].content[0]
    assert isinstance(text_part, TextModelContent)
    supplied = json.loads(text_part.text.split("Context JSON follows:\n", 1)[1])
    transition_facts = [fact for fact in supplied["facts"] if fact["kind"] == "transition"]
    assert len(transition_facts) == 1
    assert all(fact["kind"] != "exhausted_branch" for fact in supplied["facts"])
