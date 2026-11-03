"""Evidence-driven settlement cadence for compiled procedure actions."""

from __future__ import annotations

from yewo.agent.memory.knowledge import TransitionTimingSample
from yewo.agent.memory.records import AppMemory
from yewo.agent.planning.evidence import PlannerDecisionMetadata, PlannerDecisionSource
from yewo.agent.replay.reconciliation.records import ReplayProcedure
from yewo.agent.runtime.observation_settlement import ActionSettlementPolicy
from yewo.agent.runtime.observation_settlement import maximum_settlement_window_seconds


MINIMUM_TIMING_SAMPLE_COUNT = 3
TIMING_SAFETY_MULTIPLIER = 1.5
MINIMUM_SETTLEMENT_INTERVAL_SECONDS = 0.05


class ProcedureSettlementPolicySource:
    """Resolve compiled-step timing without affecting repair actions."""

    def __init__(self, procedure: ReplayProcedure, memory: AppMemory | None) -> None:
        procedure_step_ids = {step.source_step_id for step in procedure.steps}
        if memory is None:
            self._samples_by_step: dict[str, tuple[TransitionTimingSample, ...]] = {}
            return
        transition_ids_by_step: dict[str, list[str]] = {}
        for transition in memory.transitions:
            if transition.source_step_id in procedure_step_ids:
                transition_ids_by_step.setdefault(transition.source_step_id, []).append(
                    transition.transition_id
                )
        timing_by_transition = {
            timing.transition_id: timing.samples for timing in memory.app_knowledge.timing
        }
        self._samples_by_step = {
            step_id: timing_by_transition[transition_ids[0]]
            for step_id, transition_ids in transition_ids_by_step.items()
            if len(transition_ids) == 1 and transition_ids[0] in timing_by_transition
        }

    def resolve(
        self,
        default: ActionSettlementPolicy,
        decision: PlannerDecisionMetadata,
    ) -> ActionSettlementPolicy:
        """Use only repeated dispatch-to-settled samples for one exact step."""
        if (
            decision.source is not PlannerDecisionSource.COMPILED_PROCEDURE
            or len(decision.reused_procedure_step_ids) != 1
        ):
            return default
        replay_default = _replay_default_policy(default)
        samples = self._samples_by_step.get(decision.reused_procedure_step_ids[0], ())
        reliable = tuple(
            sample for sample in samples if sample.measurement == "dispatch_to_settled_v1"
        )
        if len(reliable) < MINIMUM_TIMING_SAMPLE_COUNT:
            return replay_default
        return _timed_policy(replay_default, reliable)


def _replay_default_policy(default: ActionSettlementPolicy) -> ActionSettlementPolicy:
    interval_count = default.maximum_observations - 1
    maximum_window_seconds = maximum_settlement_window_seconds(default)
    return default.model_copy(
        update={
            "initial_delay_seconds": 0,
            "interval_seconds": maximum_window_seconds / interval_count,
            "required_stable_observations": 1,
        }
    )


def _timed_policy(
    default: ActionSettlementPolicy,
    samples: tuple[TransitionTimingSample, ...],
) -> ActionSettlementPolicy:
    maximum_duration_seconds = max(sample.duration_ms for sample in samples) / 1_000
    interval_count = default.maximum_observations - 1
    evidence_interval_seconds = maximum_duration_seconds * TIMING_SAFETY_MULTIPLIER / interval_count
    interval_seconds = min(
        default.interval_seconds,
        max(MINIMUM_SETTLEMENT_INTERVAL_SECONDS, evidence_interval_seconds),
    )
    return default.model_copy(
        update={
            "initial_delay_seconds": 0,
            "interval_seconds": interval_seconds,
        }
    )
