"""Deterministic compiler gates for platform-observed application action risks."""

from __future__ import annotations

from yewo.agent.memory.knowledge import ActionRiskKnowledge
from yewo.agent.runtime.interaction_intent import normalize_target_expression
from yewo.agent.memory.records import AppMemory, LogicalScreen, ObservationVariant
from yewo.agent.replay.reconciliation.records import (
    CompilerCheck,
    CompilerReason,
    EvidenceDocument,
    EvidenceState,
    EvidenceStep,
    ReplayProcedure,
)
from yewo.observations import ScreenComparisonPolicy, ScreenMatcher


def known_action_risk_reasons(
    evidence: EvidenceDocument,
    selected_steps: tuple[EvidenceStep, ...],
    memory: AppMemory | None,
) -> tuple[CompilerReason, ...]:
    """Reject selected actions that match compatible platform-observed action risks."""
    if memory is None or not memory.app_knowledge.action_risks:
        return ()
    if memory.compatibility.application != evidence.run.application:
        return (
            CompilerReason(
                check=CompilerCheck.AVOIDS_KNOWN_ACTION_RISKS,
                reason="Known-action-risk checks require memory for the same application build.",
            ),
        )
    action_risks_by_screen = _action_risks_by_screen(memory)
    reasons: list[CompilerReason] = []
    for step in selected_steps:
        state = evidence.states[step.source_state_id]
        for screen in memory.screens:
            if not _state_matches_screen(state, screen, memory):
                continue
            reasons.extend(
                _step_action_risk_reasons(step, action_risks_by_screen.get(screen.screen_id, ()))
            )
    return tuple(reasons)


def compiled_procedure_action_risk_reasons(
    evidence: EvidenceDocument,
    procedure: ReplayProcedure,
    memory: AppMemory | None,
) -> tuple[CompilerReason, ...]:
    """Recheck a stored procedure against action risks learned after compilation."""
    evidence_by_step = {step.step: step for step in evidence.steps}
    selected_steps = tuple(
        evidence_by_step[procedure_step.source_step] for procedure_step in procedure.steps
    )
    return known_action_risk_reasons(evidence, selected_steps, memory)


def compiled_procedure_has_known_action_risk(
    evidence: EvidenceDocument,
    procedure: ReplayProcedure,
    memory: AppMemory | None,
) -> bool:
    """Return whether current memory blocks a stored procedure."""
    return bool(compiled_procedure_action_risk_reasons(evidence, procedure, memory))


def _action_risks_by_screen(memory: AppMemory) -> dict[str, tuple[ActionRiskKnowledge, ...]]:
    grouped: dict[str, list[ActionRiskKnowledge]] = {}
    for action_risk in memory.app_knowledge.action_risks:
        grouped.setdefault(action_risk.screen_id, []).append(action_risk)
    return {
        screen_id: tuple(sorted(action_risks, key=lambda item: item.action_risk_id))
        for screen_id, action_risks in grouped.items()
    }


def _step_action_risk_reasons(
    step: EvidenceStep,
    action_risks: tuple[ActionRiskKnowledge, ...],
) -> tuple[CompilerReason, ...]:
    reasons: list[CompilerReason] = []
    for action_risk in action_risks:
        if action_risk.action is not step.proposal.action or not _target_matches(action_risk, step):
            continue
        reasons.append(
            CompilerReason(
                check=CompilerCheck.AVOIDS_KNOWN_ACTION_RISKS,
                source_step=step.step,
                reason=(
                    f"Step {step.step} matches platform-observed {action_risk.kind.value} action risk "
                    f"'{action_risk.action_risk_id}'. No deterministic safe precondition is recorded."
                ),
                action_risk_id=action_risk.action_risk_id,
                action_risk_sources=action_risk.sources,
            )
        )
    return tuple(reasons)


def _target_matches(action_risk: ActionRiskKnowledge, step: EvidenceStep) -> bool:
    if action_risk.target_description is None:
        return True
    target = step.proposal.target_expression
    return target is not None and _normalize_target(target) == _normalize_target(
        action_risk.target_description
    )


def _normalize_target(value: str) -> str:
    return normalize_target_expression(value)


def _state_matches_screen(
    state: EvidenceState,
    screen: LogicalScreen,
    memory: AppMemory,
) -> bool:
    if set(state.observation_ids) & {variant.observation_id for variant in screen.variants}:
        return True
    return any(
        _state_matches_variant(state, variant, memory.compatibility.screen_comparison)
        for variant in screen.variants
    )


def _state_matches_variant(
    state: EvidenceState,
    variant: ObservationVariant,
    policy: ScreenComparisonPolicy,
) -> bool:
    matcher = ScreenMatcher(policy)
    return any(
        matcher.replay_equivalent(presentation, variant.presentation)
        for presentation in state.presentations
    )
