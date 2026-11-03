"""Visible-control AppMemory capture tests."""

from datetime import datetime, timezone
from pathlib import Path

from yewo.agent.actuation import VisibleControl, VisibleControlKind
from yewo.agent.memory.knowledge_capture import build_memory_delta
from yewo.agent.memory.writeback import apply_memory_delta
from yewo.agent.runtime.control_evidence import VisibleControlObservation

from agent.discovery.support import agent_result, observation
from agent.memory.test_store import _memory


def test_verifier_controls_are_retained_as_screen_knowledge(tmp_path: Path) -> None:
    unused_compatibility, memory = _memory(tmp_path)
    before = observation("obs_1", "0000000000000000")
    after = observation(
        "obs_2",
        "1111111111111111",
        predecessor_observation_id="obs_1",
        offset_seconds=1,
    )
    result = agent_result(before, after)
    exercised_step = result.steps[0].model_copy(
        update={
            "proposal": result.steps[0].proposal.model_copy(
                update={"target_expression": "the Heart text field"}
            )
        }
    )
    result = result.model_copy(
        update={
            "steps": (exercised_step,),
            "visible_control_observations": (
                VisibleControlObservation(
                    observation_id=before.observation_id,
                    proposal_id=exercised_step.proposal.proposal_id,
                    controls=(
                        VisibleControl(
                            name="Heart",
                            kind=VisibleControlKind.TEXT_FIELD,
                            state="empty",
                            matches_proposal_target=True,
                        ),
                        VisibleControl(
                            name="Intensity",
                            kind=VisibleControlKind.DROPDOWN,
                            state="soft",
                            options=("soft", "hard"),
                        ),
                    ),
                ),
                VisibleControlObservation(
                    observation_id=before.observation_id,
                    proposal_id="later_unrelated_proposal",
                    controls=(
                        VisibleControl(
                            name="Heart",
                            kind=VisibleControlKind.TEXT_FIELD,
                            state="filled",
                        ),
                    ),
                ),
            ),
        }
    )

    delta = build_memory_delta(
        memory,
        "run_2",
        result,
        {before.observation_id: before, after.observation_id: after},
        datetime(2026, 11, 3, tzinfo=timezone.utc),
    )
    updated = apply_memory_delta(memory, delta)

    controls = {control.name: control for control in updated.app_knowledge.controls}
    assert controls["Heart"].exercised is True
    assert controls["Heart"].state == "filled"
    assert controls["Intensity"].options == ("soft", "hard")
    assert controls["Intensity"].state == "soft"
