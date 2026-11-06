"""Human-readable narration from the Device Agent runtime."""

from pathlib import Path

from yewo.agent.actuation import ActionKind
from yewo.agent import PlannerDecisionMetadata, PlannerDecisionSource
from yewo.run_narration import RunNarrator, narration_scope

from agent.actuation.support import proposal
from agent.runtime.test_agent import (
    NeverInterrupted,
    ObservationSequence,
    RuntimeDevice,
    SequencePlanner,
    _mode_policy,
    _observation,
    _runtime,
    _task,
)


def test_runtime_narrates_agent_decisions_actions_and_outcomes(tmp_path: Path) -> None:
    planner = SequencePlanner([proposal(), proposal(ActionKind.DONE)])
    observations = ObservationSequence(
        [
            _observation("obs_1", "0000000000000000"),
            _observation("obs_2", "1111111111111111", offset_seconds=1),
            _observation("obs_3", "1111111111111111", offset_seconds=2),
        ]
    )
    with narration_scope(RunNarrator(tmp_path)):
        _runtime(tmp_path, planner, observations, RuntimeDevice()).run(
            _task(), _mode_policy(), NeverInterrupted()
        )

    narration = (tmp_path / "runtime/yewo-session.log").read_text()
    assert "device-agent: Starting task: Open the record form." in narration
    assert "Step 1 requested a new planner decision." in narration
    assert "Step 1 decision: This is the next useful action" in narration
    assert "Step 1 planned tap on the add button." in narration
    assert "Step 1 grounded the target: accepted via model." in narration
    assert "Step 1 dispatching tap." in narration
    assert "Step 1 outcome observed. Signals: pixels_changed." in narration
    assert "Step 2 decision: Task is complete." in narration
    assert "Terminal done_requested: Task is complete." in narration
    trace = (tmp_path / "runtime/trace.log").read_text()
    assert "observation=obs_1 label=Initial" in trace
    assert "proposal=proposal-1 observation=obs_1 action=tap" in trace
    assert "planner_source=model reused_flow=none" in trace
    assert "grounding=" in trace
    assert "point=(100,200)" in trace
    assert "authorization=" in trace
    assert "receipt=" in trace
    assert "outcome=" in trace
    assert "fingerprint=1111111111111111" in trace


def test_runtime_narrates_reused_compiled_procedure_decisions(tmp_path: Path) -> None:
    class ReusedPlanner(SequencePlanner):
        def decision_metadata(self) -> PlannerDecisionMetadata:
            return PlannerDecisionMetadata(
                source=PlannerDecisionSource.COMPILED_PROCEDURE,
                reused_flow_id="procedure_1",
                reused_procedure_step_ids=(
                    ("procedure_step_1",) if len(self.requests) == 1 else ()
                ),
                reused_segment_index=0 if len(self.requests) == 1 else None,
            )

    planner = ReusedPlanner([proposal(), proposal(ActionKind.DONE)])
    observations = ObservationSequence(
        [
            _observation("obs_1", "0000000000000000"),
            _observation("obs_2", "1111111111111111", offset_seconds=1),
            _observation("obs_3", "1111111111111111", offset_seconds=2),
        ]
    )
    with narration_scope(RunNarrator(tmp_path)):
        _runtime(
            tmp_path,
            planner,
            observations,
            RuntimeDevice(),
            planner_decision_source=PlannerDecisionSource.COMPILED_PROCEDURE,
            reused_flow_id="procedure_1",
        ).run(_task(), _mode_policy(), NeverInterrupted())

    narration = (tmp_path / "runtime/yewo-session.log").read_text()
    assert "Step 1 reused a compiled procedure decision." in narration
    trace = (tmp_path / "runtime/trace.log").read_text()
    assert "planner_source=compiled_procedure reused_flow=procedure_1" in trace
