"""Stable planner cache-prefix behavior."""

from yewo.agent.actuation import ActionKind

from agent.planning.support import StubPlannerModel, planner, planning_request, proposal_output


def test_planner_keeps_task_context_after_the_stable_cache_prefix() -> None:
    first_model = StubPlannerModel(proposal_output())
    second_model = StubPlannerModel(proposal_output())
    planner(first_model).plan(planning_request(open_work=("Create the record.",)))
    planner(second_model).plan(
        planning_request(
            objective="Reopen one record.",
            open_work=("Reopen the record.",),
            available_actions=frozenset(
                {ActionKind.CANNOT_PROGRESS, ActionKind.DISMISS_KEYBOARD, ActionKind.TAP}
            ),
        )
    )

    first_messages = first_model.requests[0].messages
    second_messages = second_model.requests[0].messages
    assert first_messages[0] == second_messages[0]
    assert first_messages[1] == second_messages[1]
    assert first_messages[2] != second_messages[2]
