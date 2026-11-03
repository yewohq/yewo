"""Build deterministic targets for restoring a compiled procedure."""

from dataclasses import dataclass
from typing import Literal

from yewo.agent.actuation import ActionKind, ActionProposal
from yewo.agent.replay.procedure_grounding import build_keyboard_reflow_step
from yewo.agent.replay.procedure_records import ProcedureRepairTarget
from yewo.agent.replay.reconciliation_records import (
    EvidenceDocument,
    ProcedureEffect,
    ReplayProcedure,
    ReplayProcedureStep,
)


@dataclass(frozen=True)
class ProcedureRepairPreparation:
    """Deterministic repair target and grounding candidates for one divergence."""

    target: ProcedureRepairTarget
    grounding_steps_by_expression: dict[str, ReplayProcedureStep]
    blocked_proposal: ActionProposal | None
    clears_pending_step: bool


def build_procedure_repair_preparation(
    procedure: ReplayProcedure,
    evidence: EvidenceDocument,
    step: ReplayProcedureStep,
    guard: Literal["source", "expectation", "runtime"],
    expected_state_id: str | None,
    reason: str,
) -> ProcedureRepairPreparation | None:
    """Build one repair target without mutating procedure execution state."""
    if expected_state_id is None or guard == "runtime":
        return None
    resume_index = step.procedure_index + (1 if guard == "expectation" else 0)
    establishing_step = select_repair_establishing_step(
        procedure,
        step,
        guard,
        expected_state_id,
    )
    reset_step = select_keyboard_reset_step(procedure, establishing_step)
    keyboard_reflow_step = build_keyboard_reflow_step(
        establishing_step,
        reset_step,
        procedure,
        evidence,
    )
    grounding_steps = {
        candidate.proposal.target_expression: candidate
        for candidate in (keyboard_reflow_step or establishing_step, reset_step)
        if candidate is not None and candidate.proposal.target_expression is not None
    }
    next_step = procedure.steps[resume_index] if resume_index < len(procedure.steps) else None
    target_state = evidence.states[expected_state_id]
    return ProcedureRepairPreparation(
        target=ProcedureRepairTarget(
            expected_state_id=expected_state_id,
            resume_procedure_index=resume_index,
            foreground_application_id=target_state.foreground_application_id,
            platform_context=target_state.platform_context,
            keyboard_visible=target_state.keyboard_visible,
            orientation=target_state.orientation,
            established_by_action=(
                establishing_step.proposal.action if establishing_step is not None else None
            ),
            established_by_target_expression=(
                establishing_step.proposal.target_expression
                if establishing_step is not None
                else None
            ),
            keyboard_reset_target_expression=(
                reset_step.proposal.target_expression if reset_step is not None else None
            ),
            next_action=next_step.proposal.action if next_step is not None else None,
            next_target_expression=(
                next_step.proposal.target_expression if next_step is not None else None
            ),
            reason=reason,
        ),
        grounding_steps_by_expression=grounding_steps,
        blocked_proposal=(
            step.proposal
            if guard == "expectation"
            and step.procedure_effect
            not in {ProcedureEffect.NAVIGATION, ProcedureEffect.ENVIRONMENT}
            else None
        ),
        clears_pending_step=guard == "expectation",
    )


def select_repair_establishing_step(
    procedure: ReplayProcedure,
    step: ReplayProcedureStep,
    guard: Literal["source", "expectation", "runtime"],
    expected_state_id: str,
) -> ReplayProcedureStep | None:
    """Select the retained action that originally established a repair target."""
    if guard == "expectation":
        return step
    if step.procedure_index == 0:
        return None
    previous = procedure.steps[step.procedure_index - 1]
    return previous if previous.expects.destination_state_id == expected_state_id else None


def select_keyboard_reset_step(
    procedure: ReplayProcedure,
    establishing_step: ReplayProcedureStep | None,
) -> ReplayProcedureStep | None:
    """Select a prior field that can reset keyboard reflow deterministically."""
    if (
        establishing_step is None
        or establishing_step.proposal.action is not ActionKind.FOCUS
        or establishing_step.proposal.target_expression is None
    ):
        return None
    required_target = establishing_step.proposal.target_expression
    return next(
        (
            candidate
            for candidate in reversed(procedure.steps[: establishing_step.procedure_index])
            if candidate.proposal.action is ActionKind.FOCUS
            and candidate.proposal.target_expression is not None
            and candidate.proposal.target_expression != required_target
        ),
        None,
    )
