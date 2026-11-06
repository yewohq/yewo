"""Evidence-linked same-run workflow progress tests."""

from collections.abc import Callable
import json
from pathlib import Path

import pytest

from yewo.agent.actuation import (
    ActionKind,
    ActionOutcome,
    ActionOutcomeDisposition,
    ActionOutcomeSignal,
    ActionRefusal,
    ActionRefusalCode,
)
from yewo.agent.actuation import ActionBudget, ActionPolicy
from yewo.agent.memory.records import ScreenComparisonPolicy
from yewo.agent.planning.evidence import PlannerDecisionMetadata, PlannerDecisionSource
from yewo.agent.planning.history import build_refusal_history_entry
from yewo.agent.planning.records import (
    TaskProgressContext,
)
from yewo.agent.runtime.interaction_intent import interaction_intent_key
from yewo.agent.runtime.no_progress import MAXIMUM_SCREEN_NO_PROGRESS_EVENTS
from yewo.agent.runtime.progress import (
    MAXIMUM_FEEDBACK_CHARACTERS,
    RunProgressLedger,
    refusal_before_grounding,
)
from yewo.agent.runtime.records import AgentStep, AgentTerminalReason
from yewo.agent.runtime.verification import PreDispatchVerification
from yewo.agent.screen_analysis import (
    CurrentScreenAnalysis,
    ScreenElement,
    ScreenElementConfidence,
    ScreenElementKind,
    ScreenSemanticRole,
    ScreenVisualKind,
    ScreenWorkflowRole,
)
from yewo.artifacts import ArtifactRetention, ArtifactSensitivity

from agent.actuation.support import limits, proposal
from agent.discovery.support import agent_result
from agent.runtime.test_agent import _observation


CREATE = "Create and save one reusable record."
REOPEN = "Return to the saved record and establish that it can be reopened or updated."


def _ledger(root: Path, identifier_factory: Callable[[], str]) -> RunProgressLedger:
    return RunProgressLedger(
        root,
        screen_comparison=ScreenComparisonPolicy(),
        identifier_factory=identifier_factory,
    )


def test_progress_ledger_retains_completed_work_and_routes(tmp_path: Path) -> None:
    before = _observation("obs_1", "0000000000000000").model_copy(
        update={"platform_context": "RecordList"}
    )
    after = _observation("obs_2", "1111111111111111").model_copy(
        update={"platform_context": "RecordEditor"}
    )
    candidate = proposal(
        ActionKind.TAP,
        target_expression="the visible create control",
        source_observation_id=before.observation_id,
    )
    ledger = _ledger(tmp_path, iter(("event_1", "event_2", "event_3")).__next__)
    ledger.start((CREATE, REOPEN), before, None)
    ledger.record_assessment(
        before,
        candidate,
        (CREATE,),
        REOPEN,
        None,
    )
    ledger.record_route(
        1,
        candidate,
        CREATE,
        before,
        after,
        ActionOutcome(
            outcome_id="outcome_1",
            authorization_id="authorization_1",
            receipt_id="receipt_1",
            action=ActionKind.TAP,
            source_observation_id=before.observation_id,
            settled_observation_id=after.observation_id,
            disposition=ActionOutcomeDisposition.OBSERVED,
            signals=frozenset({ActionOutcomeSignal.CONTEXT_CHANGED}),
            reason="The editor opened.",
            sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
            retention=ArtifactRetention.RUN,
        ),
    )

    context = ledger.progress_context(None)

    assert context.completed_work_items == (CREATE,)
    assert context.remaining_work == (REOPEN,)
    assert ledger.remaining_work() == (REOPEN,)
    assert len(context.route_attempts) == 1
    route = context.route_attempts[0]
    assert route.work_item == CREATE
    assert route.target == candidate.target_expression
    assert route.source_context == before.platform_context
    assert route.source_screen_fingerprint == "0000000000000000"
    assert route.interaction_intent_key == interaction_intent_key(candidate)
    assert route.destination_context == after.platform_context
    assert route.destination_screen_fingerprint == "1111111111111111"
    events = [
        json.loads(line)
        for line in (tmp_path / "assets/agent/progress-ledger.jsonl").read_text().splitlines()
    ]
    assert [event["kind"] for event in events] == ["initialized", "assessment", "route"]


def test_observed_non_english_workflow_role_advances_stage_without_completing_task(
    tmp_path: Path,
) -> None:
    before = _observation("obs-confirm", "0000000000000000").model_copy(
        update={"platform_context": "editor"}
    )
    after = _observation("obs-parent", "1111111111111111").model_copy(
        update={"platform_context": "collection"}
    )
    candidate = proposal(
        ActionKind.TAP,
        proposal_id="confirm-stage",
        source_observation_id=before.observation_id,
        target_element_ref="confirm-control",
        target_expression=None,
    )
    base_step = agent_result(
        before,
        after,
        disposition=ActionOutcomeDisposition.OBSERVED,
        terminal=AgentTerminalReason.BUDGET_EXHAUSTED,
    ).steps[0]
    assert base_step.outcome is not None
    step = base_step.model_copy(
        update={
            "proposal": candidate,
            "source_observation_id": before.observation_id,
            "outcome": base_step.outcome.model_copy(
                update={
                    "action": ActionKind.TAP,
                    "source_observation_id": before.observation_id,
                    "settled_observation_id": after.observation_id,
                    "disposition": ActionOutcomeDisposition.OBSERVED,
                }
            ),
        }
    )
    analysis = CurrentScreenAnalysis(
        analysis_id="analysis-confirm",
        observation_id=before.observation_id,
        screen_fingerprint=before.screen_fingerprint.value,
        screenshot_sha256=before.redacted_screenshot.sha256,
        platform_context=before.platform_context,
        keyboard_visible=before.keyboard_visible,
        orientation=before.orientation,
        elements=(
            ScreenElement(
                element_ref="confirm-control",
                name="Guardar",
                kind=ScreenElementKind(
                    visual=ScreenVisualKind.ICON,
                    semantic=ScreenSemanticRole.BUTTON,
                ),
                confidence=ScreenElementConfidence.HIGH,
                workflow_role=ScreenWorkflowRole.ADVANCE_STAGE,
                evidence="A visible action advances the current form workflow.",
                visual_cues=("filled action at the lower-right edge",),
            ),
        ),
    )
    ledger = _ledger(tmp_path, iter(("event-1", "event-2", "event-3")).__next__)
    ledger.start((CREATE,), before, None)

    ledger.record_step(
        step,
        PreDispatchVerification(
            progress_assessed=True,
            proposal_work_item=CREATE,
        ),
        before,
        after,
        analysis,
    )

    context = ledger.progress_context(None)
    assert context.completed_work_items == ()
    assert context.remaining_work == (CREATE,)
    assert len(context.current_progress) == 1
    assert "advanced one workflow stage" in context.current_progress[0]
    events = [
        json.loads(line)
        for line in (tmp_path / "assets/agent/progress-ledger.jsonl").read_text().splitlines()
    ]
    assert [event["kind"] for event in events] == ["initialized", "stage", "route"]


def test_progress_ledger_persists_refusal_and_projects_screen_facts(tmp_path: Path) -> None:
    screen = _observation("obs_1", "0000000000000000")
    candidate = proposal(
        ActionKind.TAP,
        target_expression="the saved record",
        source_observation_id=screen.observation_id,
    )
    ledger = _ledger(tmp_path, iter(("event_1", "event_2")).__next__)
    ledger.start((REOPEN,), screen, None)

    progress_facts, terminal_reason = ledger.record_refusal(
        candidate,
        screen,
        ActionRefusal(
            proposal_id=candidate.proposal_id,
            code=ActionRefusalCode.INVALID_GROUNDING,
            reason="Grounding was not accepted.",
        ),
        PlannerDecisionSource.MODEL,
    )

    assert terminal_reason is None
    assert progress_facts == ("Do not repeat tap: Grounding was not accepted.",)
    assert ledger.screen_facts(screen)[0].fact == progress_facts[0]
    history = (
        build_refusal_history_entry(
            candidate,
            ActionRefusal(
                proposal_id=candidate.proposal_id,
                code=ActionRefusalCode.INVALID_GROUNDING,
                reason="Grounding was not accepted.",
            ),
            screen,
        ),
    )
    assert ledger.screen_facts(screen, history) == ()
    events = [
        json.loads(line)
        for line in (tmp_path / "assets/agent/progress-ledger.jsonl").read_text().splitlines()
    ]
    assert events[-1]["kind"] == "refusal"
    assert events[-1]["interaction_intent_key"] == "tap|the saved record||||"
    assert events[-1]["refusal_code"] == "invalid_grounding"


def test_recent_history_deduplication_keeps_durable_screen_constraints(tmp_path: Path) -> None:
    screen = _observation("obs_1", "0000000000000000")
    ledger = _ledger(tmp_path, iter(_identifiers()).__next__)
    ledger.start((REOPEN,), screen, None)
    history = []
    for index in range(MAXIMUM_SCREEN_NO_PROGRESS_EVENTS):
        candidate = proposal(
            ActionKind.TAP,
            proposal_id=f"refused-{index}",
            target_expression=f"failed target {index}",
            source_observation_id=screen.observation_id,
        )
        refusal = ActionRefusal(
            proposal_id=candidate.proposal_id,
            code=ActionRefusalCode.INVALID_GROUNDING,
            reason="The target was not accepted.",
        )
        ledger.record_refusal(
            candidate,
            screen,
            refusal,
            PlannerDecisionSource.MODEL,
        )
        history.append(build_refusal_history_entry(candidate, refusal, screen))

    facts = ledger.screen_facts(screen, tuple(history))

    assert tuple(fact.fact for fact in facts) == (
        "This screen reached the no-progress limit. Leave this screen or stop the task.",
    )


def test_recovery_guidance_does_not_replace_canonical_task_work(tmp_path: Path) -> None:
    screen = _observation("obs_1", "0000000000000000")
    ledger = _ledger(tmp_path, lambda: "event_1")
    ledger.start((CREATE, REOPEN), screen, None)

    context = ledger.progress_context(
        TaskProgressContext(remaining_work=("Refill the fields lost after the application crash.",))
    )

    assert context.remaining_work == (CREATE, REOPEN)
    assert ledger.remaining_work() == context.remaining_work


def test_unknown_verifier_work_item_becomes_a_bounded_refusal(tmp_path: Path) -> None:
    screen = _observation("obs_1", "0000000000000000")
    ledger = _ledger(tmp_path, iter(_identifiers()).__next__)
    ledger.start((CREATE,), screen, None)
    candidate = proposal(
        ActionKind.TAP,
        source_observation_id=screen.observation_id,
    )

    refusal = ledger.record_verification(
        PreDispatchVerification(
            progress_assessed=True,
            completed_work_items=("Model-invented work item",),
        ),
        screen,
        candidate,
    )

    assert refusal is not None
    assert refusal.code is ActionRefusalCode.VERIFICATION_UNAVAILABLE
    assert ledger.remaining_work() == (CREATE,)


def test_progress_ledger_persists_a_bounded_planning_focus(tmp_path: Path) -> None:
    screen = _observation("obs_1", "0000000000000000")
    identifiers = iter(("event-1", "event-2", "event-3", "event-4", "focus-1", "event-5"))
    ledger = _ledger(tmp_path, identifiers.__next__)
    ledger.start((REOPEN,), screen, None)
    for index in range(3):
        candidate = proposal(
            ActionKind.TAP,
            proposal_id=f"proposal-{index}",
            target_expression=f"failed target {index}",
            source_observation_id=screen.observation_id,
        )
        ledger.record_refusal(
            candidate,
            screen,
            ActionRefusal(
                proposal_id=candidate.proposal_id,
                code=ActionRefusalCode.INVALID_GROUNDING,
                reason="The target was not accepted.",
            ),
            PlannerDecisionSource.MODEL,
        )

    focus = ledger.pending_planning_focus(screen, object())
    assert focus is not None
    ledger.commit_planning_focus(screen, focus, PlannerDecisionSource.MODEL)

    assert focus is not None
    assert focus.focus_id == "focus-1"
    assert focus.unresolved_work == (REOPEN,)
    events = [
        json.loads(line)
        for line in (tmp_path / "assets/agent/progress-ledger.jsonl").read_text().splitlines()
    ]
    assert events[-1]["kind"] == "planning_focus"
    assert events[-1]["planning_focus"]["focus_id"] == "focus-1"


def test_an_unattributed_route_is_not_recorded(tmp_path: Path) -> None:
    screen = _observation("obs_1", "0000000000000000")
    after = _observation("obs_2", "1111111111111111", offset_seconds=1)
    ledger = _ledger(tmp_path, lambda: "event_1")
    ledger.start((CREATE, REOPEN), screen, None)

    # Unattributed routes ride in the uncached tail of both prompts and carry a null
    # work item, so the one rule that reads routes can never use them.
    ledger.record_route(
        step_index=1,
        proposal=proposal(ActionKind.TAP, source_observation_id=screen.observation_id),
        before=screen,
        after=after,
        proposal_work_item=None,
        outcome=ActionOutcome(
            outcome_id="outcome_route",
            authorization_id="authorization_route",
            receipt_id="receipt_route",
            action=ActionKind.TAP,
            source_observation_id=screen.observation_id,
            settled_observation_id=after.observation_id,
            disposition=ActionOutcomeDisposition.OBSERVED,
            signals=frozenset({ActionOutcomeSignal.PIXELS_CHANGED}),
            reason="The settled observation contains action-outcome signals.",
            sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
            retention=ArtifactRetention.RUN,
        ),
    )

    assert ledger.progress_context(None).route_attempts == ()


def _no_change_step(
    screen,
    index: int,
    source: PlannerDecisionSource,
) -> AgentStep:
    step = agent_result(
        screen,
        screen,
        disposition=ActionOutcomeDisposition.NO_OBSERVABLE_CHANGE,
        terminal=AgentTerminalReason.BUDGET_EXHAUSTED,
    ).steps[0]
    return step.model_copy(
        update={
            "step_id": f"step-{source.value}-{index}",
            "step_index": index + 1,
            "decision": PlannerDecisionMetadata(source=source),
        }
    )


def test_system_actions_never_charge_the_screen_no_progress_counter(tmp_path: Path) -> None:
    screen = _observation("obs_1", "0000000000000000")
    ledger = _ledger(tmp_path, iter(_identifiers()).__next__)
    ledger.start((REOPEN,), screen, None)

    # Procedure repair dispatches a SYSTEM wait on the screen it is repairing.
    for index in range(MAXIMUM_SCREEN_NO_PROGRESS_EVENTS):
        ledger.record_step(
            _no_change_step(screen, index, PlannerDecisionSource.SYSTEM),
            PreDispatchVerification(),
            screen,
            screen,
        )

    fresh = proposal(
        ActionKind.TAP,
        proposal_id="a-route-the-model-has-not-tried",
        target_expression="an untried control",
        source_observation_id=screen.observation_id,
    )
    assert ledger.refusal_for(fresh, screen) is None


def test_a_walled_screen_still_admits_a_system_action(tmp_path: Path) -> None:
    screen = _observation("obs_1", "0000000000000000")
    ledger = _ledger(tmp_path, iter(_identifiers()).__next__)
    ledger.start((REOPEN,), screen, None)
    for index in range(MAXIMUM_SCREEN_NO_PROGRESS_EVENTS):
        ledger.record_step(
            _no_change_step(screen, index, PlannerDecisionSource.MODEL),
            PreDispatchVerification(),
            screen,
            screen,
        )
    wait = proposal(
        ActionKind.WAIT,
        proposal_id="repair-settlement-wait",
        target_expression=None,
        duration_seconds=0.1,
        source_observation_id=screen.observation_id,
    )
    policy = ActionPolicy(limits())

    assert (
        refusal_before_grounding(
            ledger, policy, wait, screen, ActionBudget(dispatched_actions=0), PlannerDecisionSource.MODEL
        )
        is not None
    )
    assert (
        refusal_before_grounding(
            ledger, policy, wait, screen, ActionBudget(dispatched_actions=0), PlannerDecisionSource.SYSTEM
        )
        is None
    )


@pytest.mark.parametrize(
    "decision_source",
    (PlannerDecisionSource.SYSTEM, PlannerDecisionSource.COMPILED_PROCEDURE),
)
def test_a_non_model_decision_leaves_the_planning_focus_owed(
    tmp_path: Path,
    decision_source: PlannerDecisionSource,
) -> None:
    screen = _observation("obs_1", "0000000000000000")
    ledger = _ledger(tmp_path, iter(_identifiers()).__next__)
    ledger.start((REOPEN,), screen, None)
    for index in range(3):
        candidate = proposal(
            ActionKind.TAP,
            proposal_id=f"refused-{index}",
            target_expression=f"failed target {index}",
            source_observation_id=screen.observation_id,
        )
        ledger.record_refusal(
            candidate,
            screen,
            ActionRefusal(
                proposal_id=candidate.proposal_id,
                code=ActionRefusalCode.INVALID_GROUNDING,
                reason="The target was not accepted.",
            ),
            PlannerDecisionSource.MODEL,
        )

    owed = ledger.pending_planning_focus(screen, object())
    assert owed is not None
    # Divergence is detected after the focus is built, so the procedure source
    # answers with a SYSTEM wait that no focus-aware planner ever read.
    assert ledger.commit_planning_focus(screen, owed, decision_source) is None

    again = ledger.pending_planning_focus(screen, object())
    assert again is not None
    assert ledger.commit_planning_focus(screen, again, PlannerDecisionSource.MODEL) is again
    assert ledger.pending_planning_focus(screen, object()) is None


def test_an_over_long_verifier_refusal_reaches_the_ledger_truncated(tmp_path: Path) -> None:
    screen = _observation("obs_1", "0000000000000000")
    ledger = _ledger(tmp_path, iter(_identifiers()).__next__)
    ledger.start((REOPEN,), screen, None)
    candidate = proposal(
        ActionKind.TAP,
        proposal_id="the-verifier-rejected-this",
        target_expression="the save control",
        source_observation_id=screen.observation_id,
    )
    reason = "The action verifier explained its rejection at length. " * 200

    ledger.record_refusal(
        candidate,
        screen,
        ActionRefusal(
            proposal_id=candidate.proposal_id,
            code=ActionRefusalCode.ACTION_NOT_ALLOWED,
            reason=reason,
        ),
        PlannerDecisionSource.MODEL,
    )

    assert len(reason) > MAXIMUM_FEEDBACK_CHARACTERS
    events = [
        json.loads(line)
        for line in (tmp_path / "assets/agent/progress-ledger.jsonl").read_text().splitlines()
    ]
    assert events[-1]["feedback"] == reason[:MAXIMUM_FEEDBACK_CHARACTERS]


def _identifiers() -> tuple[str, ...]:
    return tuple(f"event-{index}" for index in range(1, 40))
