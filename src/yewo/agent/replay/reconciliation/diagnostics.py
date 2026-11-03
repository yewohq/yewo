"""Derive deterministic diagnostics from recorded reconciliation evidence."""

from collections import defaultdict

from yewo.agent.actuation import ActionKind, ActionOutcomeSignal
from yewo.agent.replay.reconciliation.records import (
    DiagnosticDerivation,
    DiagnosticInput,
    EvidenceDiagnostic,
    EvidenceDiagnostics,
    EvidenceResult,
    EvidenceStep,
)


def build_evidence_diagnostics(steps: tuple[EvidenceStep, ...]) -> EvidenceDiagnostics:
    """Build deterministic diagnostics for one ordered evidence timeline."""
    return EvidenceDiagnostics(
        conflicts=_conflict_diagnostics(steps),
        no_effect_repeats=_no_effect_diagnostics(steps),
        returned_to_same_state=tuple(
            _diagnostic(
                (step.step,),
                "The action returned to the same grouped state.",
                "same-state-v1",
                ("recorded_outcome", "grouped_state"),
            )
            for step in steps
            if step.destination_state_id == step.source_state_id
            and step.destination_state_id is not None
        ),
        data_entry=tuple(
            _diagnostic(
                (step.step,),
                "The step records a data-entry action.",
                "data-entry-v1",
                ("recorded_action", "inferred_control"),
                uses_expression=True,
            )
            for step in steps
            if step.action in {ActionKind.TYPE, ActionKind.CLEAR}
        ),
        crashes=tuple(
            _diagnostic(
                (step.step,),
                "Trusted runtime evidence records an application failure.",
                "application-failure-v1",
                ("recorded_outcome",),
            )
            for step in steps
            if step.application_failed
        ),
        unexpected_signals=tuple(
            _diagnostic(
                (step.step,),
                "The recorded outcome contains a failure or text-delivery signal.",
                "unexpected-signal-v1",
                ("recorded_outcome",),
            )
            for step in steps
            if step.signals
            & {
                ActionOutcomeSignal.APP_BACKGROUNDED,
                ActionOutcomeSignal.APP_EXITED,
                ActionOutcomeSignal.TEXT_NOT_LANDED,
            }
        ),
    )


def _conflict_diagnostics(
    steps: tuple[EvidenceStep, ...],
) -> tuple[EvidenceDiagnostic, ...]:
    values: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for step in steps:
        text = step.proposal.text
        key = step.control.identity_key
        if key is not None and text is not None and text.source == "literal":
            values[key][text.value].append(step.step)
    return tuple(
        _diagnostic(
            tuple(index for indexes in by_value.values() for index in indexes),
            "The same inferred control received different literal values.",
            "control-value-conflict-v1",
            ("recorded_action", "inferred_control"),
            uses_expression=True,
        )
        for by_value in values.values()
        if len(by_value) > 1
    )


def _no_effect_diagnostics(
    steps: tuple[EvidenceStep, ...],
) -> tuple[EvidenceDiagnostic, ...]:
    repeated: dict[tuple[ActionKind, str | None], list[int]] = defaultdict(list)
    for step in steps:
        if step.result is EvidenceResult.NO_EFFECT and step.control.identity_key is not None:
            repeated[(step.action, step.control.identity_key)].append(step.step)
    return tuple(
        _diagnostic(
            tuple(indexes),
            "The same inferred action target repeatedly produced no observable change.",
            "no-effect-repeat-v1",
            ("recorded_action", "recorded_outcome", "inferred_control"),
            uses_expression=True,
        )
        for indexes in repeated.values()
        if len(indexes) > 1
    )


def _diagnostic(
    indexes: tuple[int, ...],
    reason: str,
    method: str,
    inputs: tuple[DiagnosticInput, ...],
    *,
    uses_expression: bool = False,
) -> EvidenceDiagnostic:
    return EvidenceDiagnostic(
        step_indexes=indexes,
        reason=reason,
        derivation=DiagnosticDerivation(
            method=method,
            inputs=inputs,
            uses_model_authored_expression=uses_expression,
        ),
    )
