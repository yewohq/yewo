"""Fresh planner context after deterministic no-progress evidence."""

import json

from yewo.agent.actuation import ActionKind
from yewo.agent.planning.planner import (
    TASK_PLANNER_FOCUS_USER_PROMPT,
)
from yewo.agent.planning.records import (
    PlanningFocusCause,
    PlanningFocusContext,
    ProgressRouteAttempt,
)
from yewo.models import TextModelContent

from agent.planning.support import (
    StubPlannerModel,
    planner,
    planning_request,
    proposal_output,
)


def test_focus_planning_uses_an_isolated_current_state_request() -> None:
    model = StubPlannerModel(proposal_output())
    request = planning_request(
        planning_focus=PlanningFocusContext(
            focus_id="focus-1",
            cause=PlanningFocusCause.SCREEN_NO_PROGRESS,
            reason="Three interactions did not advance the task.",
            unresolved_work=("Reopen the saved record.",),
            prohibited_interaction_intents=("tap|the record row||||",),
            recent_routes=(
                ProgressRouteAttempt(
                    step_index=4,
                    action=ActionKind.TAP,
                    target="the record row",
                    outcome="no_observable_change",
                ),
            ),
        )
    )

    planner(model).plan(request)

    model_request = model.requests[0]
    assert len(model_request.messages) == 3
    # The stable prompt is the cached prefix; the focus payload changes every step.
    assert model_request.messages[1].cache_prefix_end
    assert model_request.messages[2].cache_prefix_end is False
    prompt = model_request.messages[1].content[0]
    context = model_request.messages[2].content[0]
    assert isinstance(prompt, TextModelContent)
    assert isinstance(context, TextModelContent)
    assert prompt.text == TASK_PLANNER_FOCUS_USER_PROMPT
    assert "Trusted history event" not in str(model_request.messages)
    focus_context = json.loads(context.text.split("\n", 1)[1])
    assert focus_context["focus_context"]["unresolved_work"] == ["Reopen the saved record."]
    route = focus_context["focus_context"]["recent_routes"][0]
    assert route == {
        "action": "tap",
        "outcome": "no_observable_change",
        "target": "the record row",
    }
    assert "done" in focus_context["available_actions"]
    assert "cannot_progress" in focus_context["available_actions"]


def test_focus_request_carries_the_evidence_its_rules_depend_on() -> None:
    from yewo.agent.planning.history import PlanningScreenFact, PlanningScreenReference
    from yewo.agent.planning.records import (
    TaskProgressContext,
)

    model = StubPlannerModel(proposal_output())
    base = planning_request()
    fact = PlanningScreenFact(
        fact="The application exited after this action.",
        screen=PlanningScreenReference(
            observation_id=base.observation.observation_id,
            screen_fingerprint=base.observation.screen_fingerprint.value,
        ),
    )
    request = base.model_copy(
        update={
            "planning_focus": PlanningFocusContext(
                focus_id="focus-1",
                cause=PlanningFocusCause.SCREEN_NO_PROGRESS,
                reason="Three interactions did not advance the task.",
                unresolved_work=("Reopen the saved record.",),
            ),
            "screen_facts": (fact,),
            "progress_context": TaskProgressContext(completed_work_items=("Create the record.",)),
        }
    )

    planner(model).plan(request)

    context = model.requests[0].messages[2].content[0]
    assert isinstance(context, TextModelContent)
    payload = json.loads(context.text.split("\n", 1)[1])
    # The focus prompt forbids restarting completed work and cites screen evidence,
    # so both must reach the payload it is given.
    assert payload["screen_facts"]
    assert payload["completed_work_items"] == ["Create the record."]


def test_focus_can_request_independently_verified_completion() -> None:
    model = StubPlannerModel(proposal_output(ActionKind.DONE))
    request = planning_request(
        planning_focus=PlanningFocusContext(
            focus_id="focus-1",
            cause=PlanningFocusCause.SCREEN_NO_PROGRESS,
            reason="The screen did not advance.",
        )
    )

    selected = planner(model).plan(request)

    assert selected.action is ActionKind.DONE
