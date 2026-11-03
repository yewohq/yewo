"""Compiled-action settlement across deterministic environment recovery."""

from yewo.agent.actuation import ActionKind
from yewo.agent.planning.history import PlanningHistoryEntry

from agent.replay.test_procedure_compiler import _observations
from agent.replay.test_procedure_repair import _outcome, _repair_source, _request


def test_system_surface_recovery_settles_the_interrupted_compiled_action() -> None:
    source, planner = _repair_source()
    before = _observations()[0]
    destination = _observations()[2]
    compiled_action = source.plan(_request(before))
    overlay = destination.model_copy(
        update={
            "observation_id": "system_overlay_after_action",
            "foreground_application_id": "com.android.systemui",
            "platform_context": "notification_shade",
        }
    )
    compiled_feedback = _outcome(compiled_action, before, overlay)

    environment_action = source.plan(_request(overlay, (compiled_feedback,)))
    assert compiled_feedback.outcome is not None
    environment_feedback = PlanningHistoryEntry(
        proposal=environment_action,
        outcome=compiled_feedback.outcome.model_copy(
            update={
                "action": ActionKind.BACK,
                "source_observation_id": overlay.observation_id,
                "settled_observation_id": destination.observation_id,
            }
        ),
    )
    resumed = source.plan(_request(destination, (compiled_feedback, environment_feedback)))

    assert environment_action.action is ActionKind.BACK
    assert resumed.action is ActionKind.TYPE
    assert source.completed_step_count == 1
    assert planner.request_count == 0
    assert source.divergence is None
