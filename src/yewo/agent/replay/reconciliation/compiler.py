"""Fail-closed compilation of untrusted reconciliation selections."""

from __future__ import annotations

from collections.abc import Callable

from yewo.agent.actuation import (
    ACTION_DEFINITIONS,
    ActionHandling,
    ActionKind,
    ActionOutcomeSignal,
)
from yewo.agent.memory.records import AppMemory
from yewo.agent.replay.procedure.action_risks import known_action_risk_reasons
from yewo.agent.replay.reconciliation.evidence import is_commit_target
from yewo.agent.replay.reconciliation.records import (
    CompilationOutcome,
    CompilationWarning,
    CompilerCheck,
    CompilerReason,
    CompilerRefusal,
    ControlRole,
    EvidenceDocument,
    EvidenceResult,
    EvidenceStep,
    EvidenceStepKind,
    ProcedureEffect,
    ProcedureExpectation,
    ReconcilerResponse,
    ReplayProcedure,
    ReplayProcedureStep,
)
from yewo.artifacts import new_record_identifier


_REUSABLE_RESULTS = frozenset(
    {
        EvidenceResult.OBSERVED,
        EvidenceResult.AMBIGUOUS,
        EvidenceResult.NOT_CAPTURED,
    }
)
_STATE_CONTROL_ROLES = frozenset(
    {
        ControlRole.CHECKBOX,
        ControlRole.RADIO_BUTTON,
        ControlRole.SWITCH,
        ControlRole.OPTION,
    }
)
_NAVIGATION_ROLES = frozenset(
    {
        ControlRole.MENU_ITEM,
        ControlRole.TAB,
        ControlRole.LINK,
        ControlRole.LIST_ITEM,
        ControlRole.ROW,
    }
)


def compile_replay_procedure(
    evidence: EvidenceDocument,
    response: ReconcilerResponse,
    *,
    memory: AppMemory | None = None,
    identifier_factory: Callable[[], str] = new_record_identifier,
) -> CompilationOutcome:
    """Apply deterministic gates and never change the model selection."""
    selection_reasons = _check_selection(evidence, response)
    if selection_reasons:
        return _refusal(evidence, response, selection_reasons)

    by_index = {step.step: step for step in evidence.steps}
    selected = tuple(by_index[index] for index in response.procedure)
    reasons = (
        *_check_launch(evidence, selected),
        *_check_required_work(evidence, response, selected),
        *_check_reusable(evidence, selected),
        *_check_coverage(evidence, response),
        *known_action_risk_reasons(evidence, selected, memory),
    )
    if reasons:
        return _refusal(evidence, response, reasons)

    warnings = _warnings(evidence, selected)
    compiled_steps = tuple(
        _compile_step(
            index,
            step,
            response,
            selected[index - 1] if index > 0 else None,
            selected[index + 1].source_state_id if index + 1 < len(selected) else None,
        )
        for index, step in enumerate(selected)
    )
    return ReplayProcedure(
        visual_policy_version=1,
        procedure_id=identifier_factory(),
        request_id=response.request_id,
        evidence_id=evidence.evidence_id,
        source_run_id=evidence.run.run_id,
        application=evidence.run.application,
        source_step_ids=tuple(step.source_step_id for step in selected),
        steps=compiled_steps,
        warnings=warnings,
    )


def procedure_effect(step: EvidenceStep) -> ProcedureEffect:
    """Classify replay behavior without changing authorization side effects."""
    if step.kind is EvidenceStepKind.ENVIRONMENT:
        return ProcedureEffect.ENVIRONMENT
    if step.action in {ActionKind.TYPE, ActionKind.CLEAR}:
        return ProcedureEffect.STATE_SETTING
    if step.control.role in _STATE_CONTROL_ROLES:
        return ProcedureEffect.STATE_SETTING
    if step.action in {
        ActionKind.BACK,
        ActionKind.DISMISS_KEYBOARD,
        ActionKind.FOCUS,
        ActionKind.SCROLL,
    }:
        return ProcedureEffect.NAVIGATION
    if step.control.role in _NAVIGATION_ROLES:
        return ProcedureEffect.NAVIGATION
    if is_commit_target(step.proposal.target_expression):
        return ProcedureEffect.COMMIT
    return ProcedureEffect.UNKNOWN


def _check_selection(
    evidence: EvidenceDocument,
    response: ReconcilerResponse,
) -> tuple[CompilerReason, ...]:
    available = {step.step for step in evidence.steps}
    reasons: list[CompilerReason] = []
    for index in response.procedure:
        if index not in available:
            reasons.append(
                CompilerReason(
                    check=CompilerCheck.SELECTION_ONLY,
                    source_step=index if index > 0 else None,
                    reason=f"Selected step {index} does not exist in the source run.",
                )
            )
    if len(set(response.procedure)) != len(response.procedure):
        reasons.append(
            CompilerReason(
                check=CompilerCheck.SELECTION_ONLY,
                reason="Selected step indexes must be unique.",
            )
        )
    if any(
        right <= left
        for left, right in zip(response.procedure, response.procedure[1:], strict=False)
    ):
        reasons.append(
            CompilerReason(
                check=CompilerCheck.SELECTION_ONLY,
                reason="Selected step indexes must be strictly increasing.",
            )
        )
    return tuple(reasons)


def _check_launch(
    evidence: EvidenceDocument,
    selected: tuple[EvidenceStep, ...],
) -> tuple[CompilerReason, ...]:
    if not selected:
        return (
            CompilerReason(
                check=CompilerCheck.STARTS_AT_LAUNCH,
                reason="A replay procedure must select at least one source step.",
            ),
        )
    first = selected[0]
    if first.source_state_id == evidence.run.launch_state_id:
        return ()
    return (
        CompilerReason(
            check=CompilerCheck.STARTS_AT_LAUNCH,
            source_step=first.step,
            reason="The first selected step does not use the recorded launch state.",
        ),
    )


def _check_required_work(
    evidence: EvidenceDocument,
    response: ReconcilerResponse,
    selected: tuple[EvidenceStep, ...],
) -> tuple[CompilerReason, ...]:
    selected_indexes = {step.step for step in selected}
    reasons: list[CompilerReason] = []
    for step in evidence.steps:
        if not _is_required_source_step(step) or step.step in selected_indexes:
            continue
        if _has_selected_correction(step, evidence, response, selected_indexes):
            continue
        reasons.append(
            CompilerReason(
                check=CompilerCheck.NO_LOST_STATE,
                source_step=step.step,
                reason=(
                    f"Step {step.step} is reusable {procedure_effect(step).value} work. "
                    "Keep it or identify a selected corrected attempt for the same control."
                ),
            )
        )
    return tuple(reasons)


def _is_required_source_step(step: EvidenceStep) -> bool:
    if step.result not in _REUSABLE_RESULTS:
        return False
    return procedure_effect(step) in {
        ProcedureEffect.STATE_SETTING,
        ProcedureEffect.COMMIT,
        ProcedureEffect.UNKNOWN,
    }


def _has_selected_correction(
    missing: EvidenceStep,
    evidence: EvidenceDocument,
    response: ReconcilerResponse,
    selected_indexes: set[int],
) -> bool:
    identity = missing.control.identity_key
    if identity is None:
        return False
    by_index = {step.step: step for step in evidence.steps}
    for decision in response.decisions:
        if missing.step not in decision.drop:
            continue
        for kept_index in decision.keep:
            kept = by_index.get(kept_index)
            if (
                kept_index in selected_indexes
                and kept is not None
                and kept.step > missing.step
                and not any(
                    missing.step < selected_index < kept.step for selected_index in selected_indexes
                )
                and kept.control.identity_key == identity
                and kept.action == missing.action
                and kept.result in _REUSABLE_RESULTS
            ):
                return True
    return False


def _check_reusable(
    evidence: EvidenceDocument,
    selected: tuple[EvidenceStep, ...],
) -> tuple[CompilerReason, ...]:
    reasons = [reason for step in selected for reason in _step_reuse_reasons(step)]
    final_reason = _final_expectation_reason(selected)
    if final_reason is not None:
        reasons.append(final_reason)
    for previous, current in zip(selected, selected[1:], strict=False):
        context_reason = _disconnected_context_reason(evidence, previous, current)
        if context_reason is not None:
            reasons.append(context_reason)
    return tuple(reasons)


def _step_reuse_reasons(step: EvidenceStep) -> tuple[CompilerReason, ...]:
    reasons: list[CompilerReason] = []
    if ACTION_DEFINITIONS[step.action].handling is not ActionHandling.EXECUTE:
        reasons.append(
            CompilerReason(
                check=CompilerCheck.RESULTS_REUSABLE,
                source_step=step.step,
                reason="A replay procedure can contain only executable device actions.",
            )
        )
    if step.kind is EvidenceStepKind.ENVIRONMENT:
        reasons.append(
            CompilerReason(
                check=CompilerCheck.RESULTS_REUSABLE,
                source_step=step.step,
                reason="An environment step cannot be stored in a replay procedure.",
            )
        )
    if step.result not in _REUSABLE_RESULTS:
        reasons.append(
            CompilerReason(
                check=CompilerCheck.RESULTS_REUSABLE,
                source_step=step.step,
                reason=f"Step {step.step} has non-reusable result '{step.result.value}'.",
            )
        )
    return tuple(reasons)


def _final_expectation_reason(
    selected: tuple[EvidenceStep, ...],
) -> CompilerReason | None:
    if not selected:
        return None
    final_step = selected[-1]
    if final_step.destination_state_id is not None or final_step.signals:
        return None
    return CompilerReason(
        check=CompilerCheck.RESULTS_REUSABLE,
        source_step=final_step.step,
        reason="The final selected step has no deterministic replay expectation.",
    )


def _disconnected_context_reason(
    evidence: EvidenceDocument,
    previous: EvidenceStep,
    current: EvidenceStep,
) -> CompilerReason | None:
    destination_state_id = previous.destination_state_id
    if destination_state_id is None or destination_state_id == current.source_state_id:
        return None
    destination = evidence.states[destination_state_id]
    source = evidence.states[current.source_state_id]
    shares_application_context = (
        destination.foreground_application_id == source.foreground_application_id
        and destination.platform_context == source.platform_context
    )
    if shares_application_context:
        return None
    return CompilerReason(
        check=CompilerCheck.RESULTS_REUSABLE,
        source_step=current.step,
        reason=(
            "Selected steps cross disconnected application contexts between "
            f"steps {previous.step} and {current.step}."
        ),
    )


def _check_coverage(
    evidence: EvidenceDocument,
    response: ReconcilerResponse,
) -> tuple[CompilerReason, ...]:
    selected = set(response.procedure)
    expected_conditions = set(range(len(evidence.task.completion_conditions)))
    reasons: list[CompilerReason] = []
    missing = sorted(expected_conditions - set(response.milestones))
    extra = sorted(set(response.milestones) - expected_conditions)
    if missing:
        reasons.append(
            CompilerReason(
                check=CompilerCheck.COVERS_TASK,
                reason=f"Completion conditions without milestones: {missing}.",
            )
        )
    if extra:
        reasons.append(
            CompilerReason(
                check=CompilerCheck.COVERS_TASK,
                reason=f"Milestones name unknown completion conditions: {extra}.",
            )
        )
    for condition, source_step in response.milestones.items():
        if source_step not in selected:
            reasons.append(
                CompilerReason(
                    check=CompilerCheck.COVERS_TASK,
                    source_step=source_step if source_step > 0 else None,
                    reason=(f"Milestone {condition} must reference a selected source step."),
                )
            )
    return tuple(reasons)


def _warnings(
    evidence: EvidenceDocument,
    selected: tuple[EvidenceStep, ...],
) -> tuple[CompilationWarning, ...]:
    warnings: list[CompilationWarning] = []
    for step in selected:
        if step.result in {EvidenceResult.AMBIGUOUS, EvidenceResult.NOT_CAPTURED}:
            warnings.append(
                CompilationWarning(
                    source_step=step.step,
                    reason=(
                        f"Source result is {step.result.value}. Replay must prove the expectation."
                    ),
                )
            )
    for previous, current in zip(selected, selected[1:], strict=False):
        if previous.destination_state_id is None:
            warnings.append(
                CompilationWarning(
                    source_step=current.step,
                    reason=(
                        f"Step {previous.step} has no captured destination. Replay must prove "
                        "connectivity."
                    ),
                )
            )
        elif previous.destination_state_id != current.source_state_id:
            before = evidence.states[previous.destination_state_id]
            after = evidence.states[current.source_state_id]
            if (
                before.foreground_application_id == after.foreground_application_id
                and before.platform_context == after.platform_context
            ):
                warnings.append(
                    CompilationWarning(
                        source_step=current.step,
                        reason=(
                            "Selected steps have a same-context state gap. Replay must prove "
                            "the next precondition."
                        ),
                    )
                )
    return tuple(warnings)


def _compile_step(
    procedure_index: int,
    step: EvidenceStep,
    response: ReconcilerResponse,
    previous_step: EvidenceStep | None,
    next_source_state_id: str | None,
) -> ReplayProcedureStep:
    captures = tuple(
        condition
        for condition, source_step in sorted(response.milestones.items())
        if source_step == step.step
    )
    return ReplayProcedureStep(
        procedure_index=procedure_index,
        source_step=step.step,
        source_step_id=step.source_step_id,
        source_state_id=step.source_state_id,
        proposal=step.proposal,
        target_grounding=step.target_grounding,
        target_visual_descriptor=step.target_visual_descriptor,
        destination_grounding=step.destination_grounding,
        procedure_effect=procedure_effect(step),
        expects=ProcedureExpectation(
            destination_state_id=step.destination_state_id or next_source_state_id,
            required_signals=_required_signals(step, previous_step),
            source_outcome_disposition=step.outcome_disposition,
        ),
        captures_completion_conditions=captures,
    )


def _required_signals(
    step: EvidenceStep,
    previous_step: EvidenceStep | None,
) -> frozenset[ActionOutcomeSignal]:
    signals = step.signals
    if step.action is not ActionKind.TYPE or not _is_secure_text_step(step, previous_step):
        return signals
    return signals - {
        ActionOutcomeSignal.TEXT_LANDED,
        ActionOutcomeSignal.TEXT_NOT_LANDED,
    }


def _is_secure_text_step(step: EvidenceStep, previous_step: EvidenceStep | None) -> bool:
    traits = step.control.input_traits
    if traits is not None and traits.secure is not None:
        return traits.secure
    return _is_masked_field_focus(previous_step)


def _is_masked_field_focus(step: EvidenceStep | None) -> bool:
    if step is None or step.action is not ActionKind.FOCUS:
        return False
    description = " ".join(
        value
        for value in (
            step.control.label,
            step.control.identity_key,
            step.proposal.target_expression,
        )
        if value is not None
    ).lower()
    return any(
        term in description for term in ("password", "passcode", "security code", "cvv", "pin code")
    )


def _refusal(
    evidence: EvidenceDocument,
    response: ReconcilerResponse,
    reasons: tuple[CompilerReason, ...],
) -> CompilerRefusal:
    return CompilerRefusal(
        request_id=response.request_id,
        evidence_id=evidence.evidence_id,
        reasons=reasons,
    )
