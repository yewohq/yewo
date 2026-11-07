"""Build testable surfaces and readiness from accepted discovery evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256

from yewo.agent.actuation import ActionKind, DeclaredValueReference
from yewo.agent.discovery import DiscoveryObjectiveState, DiscoveryResult
from yewo.agent.memory.records import AppMemory
from yewo.agent.replay.reconciliation.records import (
    ControlValueType,
    EvidenceDocument,
    EvidenceState,
    EvidenceStep,
    ProcedureEffect,
    ReplayProcedure,
    ReplayProcedureStep,
)
from yewo.artifacts import EvidenceApplicationIdentity, RecordIdentifier
from yewo.project_config import ResolvedJourney
from yewo.scenarios.records import (
    DiscoveredTestSurface,
    DiscoveryProvenance,
    JourneyRequirementEvidence,
    ScenarioEvidenceReadiness,
    ScenarioSuite,
)
from yewo.scenarios.taxonomy import RequirementEvidenceState, ScenarioSurfaceKind


class DiscoveryTestingBridgeError(ValueError):
    """Supplied discovery artifacts do not belong to one evidence chain."""


@dataclass
class _SurfaceAccumulator:
    kind: ScenarioSurfaceKind
    subject: str
    outcomes: list[str] = field(default_factory=list)
    observation_ids: list[RecordIdentifier] = field(default_factory=list)
    state_ids: list[RecordIdentifier] = field(default_factory=list)
    declared_value_names: list[RecordIdentifier] = field(default_factory=list)
    procedure_step_indexes: list[int] = field(default_factory=list)


def build_discovery_testing_bridge(
    journey: ResolvedJourney,
    application: EvidenceApplicationIdentity,
    discovery_run_id: RecordIdentifier,
    result: DiscoveryResult,
    *,
    procedure: ReplayProcedure | None = None,
    evidence: EvidenceDocument | None = None,
    memory: AppMemory | None = None,
) -> ScenarioSuite:
    """Derive one empty scenario suite without inventing application behavior."""
    _validate_artifact_chain(
        application,
        discovery_run_id,
        result,
        procedure=procedure,
        evidence=evidence,
        memory=memory,
    )
    issues = _readiness_issues(journey, result, procedure, evidence, memory)
    requirements = _requirement_evidence(journey, result, procedure, evidence, issues)
    blocking_risk_ids = _blocking_action_risk_ids(procedure, memory)
    risk_reason = (
        "Known application-failure risks block at least one compiled procedure action."
        if blocking_risk_ids
        else None
    )
    all_supported = all(
        requirement.state is RequirementEvidenceState.SUPPORTED for requirement in requirements
    )
    candidate_surfaces = (
        _extract_test_surfaces(journey, result, procedure, evidence)
        if all_supported
        and procedure is not None
        and evidence is not None
        and result.verification is not None
        and result.verification.satisfied
        else ()
    )
    if all_supported and procedure is not None and evidence is not None and not candidate_surfaces:
        issues.append("Discovery did not expose an observable application surface to test.")
    verified_procedure = (
        procedure is not None
        and result.verification is not None
        and result.verification.satisfied
        and all_supported
    )
    executable = verified_procedure and not blocking_risk_ids and not issues
    required_value_names = _required_value_names(procedure)
    procedure_id = procedure.procedure_id if procedure is not None else None
    readiness = ScenarioEvidenceReadiness(
        readiness_id=_derived_identifier(
            "readiness",
            discovery_run_id,
            journey.journey_id,
            procedure.procedure_id if procedure else "none",
        ),
        journey_id=journey.journey_id,
        application=application,
        discovery_run_id=discovery_run_id,
        procedure_id=procedure_id if verified_procedure else None,
        verification_id=(
            result.verification.verification_id
            if verified_procedure and result.verification
            else None
        ),
        requirements=requirements,
        required_declared_value_names=required_value_names,
        required_setup=journey.preconditions,
        blocking_action_risk_ids=blocking_risk_ids,
        readiness_blockers=tuple(dict.fromkeys(issues)),
        executable=executable,
        independently_auditable=executable,
        reason=(
            "Accepted discovery evidence supports executable and independently auditable testing."
            if executable
            else " ".join(dict.fromkeys((*issues, *((risk_reason,) if risk_reason else ()))))
        ),
    )
    surfaces = candidate_surfaces if executable else ()
    return ScenarioSuite(
        suite_id=_derived_identifier("suite", readiness.readiness_id),
        journey_id=journey.journey_id,
        application=application,
        readiness=readiness,
        surfaces=surfaces,
    )


def _validate_artifact_chain(
    application: EvidenceApplicationIdentity,
    discovery_run_id: RecordIdentifier,
    result: DiscoveryResult,
    *,
    procedure: ReplayProcedure | None,
    evidence: EvidenceDocument | None,
    memory: AppMemory | None,
) -> None:
    if procedure is not None and procedure.application != application:
        raise DiscoveryTestingBridgeError("The procedure belongs to a different application build.")
    if evidence is not None and evidence.run.application != application:
        raise DiscoveryTestingBridgeError("The evidence belongs to a different application build.")
    if memory is not None and memory.compatibility.application != application:
        raise DiscoveryTestingBridgeError(
            "The app memory belongs to a different application build."
        )
    if procedure is not None and procedure.source_run_id != discovery_run_id:
        raise DiscoveryTestingBridgeError("The procedure belongs to a different discovery run.")
    if evidence is not None and evidence.run.run_id != discovery_run_id:
        raise DiscoveryTestingBridgeError("The evidence belongs to a different discovery run.")
    if (
        procedure is not None
        and evidence is not None
        and procedure.evidence_id != evidence.evidence_id
    ):
        raise DiscoveryTestingBridgeError("The procedure does not reference the supplied evidence.")
    if procedure is not None and result.compiled_procedure_id != procedure.procedure_id:
        raise DiscoveryTestingBridgeError("The discovery result names a different procedure.")
    if memory is not None and result.memory_id != memory.memory_id:
        raise DiscoveryTestingBridgeError("The discovery result names different app memory.")
    if memory is not None and discovery_run_id not in memory.source_run_ids:
        raise DiscoveryTestingBridgeError("The app memory does not include the discovery run.")


def _readiness_issues(
    journey: ResolvedJourney,
    result: DiscoveryResult,
    procedure: ReplayProcedure | None,
    evidence: EvidenceDocument | None,
    memory: AppMemory | None,
) -> list[str]:
    objective = result.request.objectives[0]
    issues: list[str] = []
    if objective.objective_id != journey.journey_id or objective.statement != journey.objective:
        issues.append("Discovery did not investigate this exact journey objective.")
    coverage = result.coverage.entries[0]
    if coverage.objective_id != objective.objective_id:
        issues.append("Discovery coverage does not describe the requested objective.")
    if coverage.state is not DiscoveryObjectiveState.SUPPORTED:
        issues.append(f"Discovery did not support the journey: {coverage.reason}")
    if result.verification is None or not result.verification.satisfied:
        issues.append("Independent final-state verification did not support the journey.")
    if procedure is None:
        issues.append("Discovery did not produce a compatible compiled procedure.")
    if evidence is None:
        issues.append("Discovery compilation evidence is unavailable.")
    if memory is None:
        issues.append("Accepted app memory is unavailable.")
    if evidence is not None and (
        evidence.task.objective_id != objective.objective_id
        or evidence.task.objective != objective.statement
    ):
        issues.append("Compilation evidence does not describe the discovery objective.")
    if evidence is not None and not _contains_requested_conditions(
        evidence.task.completion_conditions,
        objective.completion_conditions,
    ):
        issues.append("Compilation evidence omitted a requested completion condition.")
    if procedure is not None and result.compiled_procedure_step_count != len(procedure.steps):
        issues.append("The compiled procedure step count does not match discovery.")
    unknown_values = set(_required_value_names(procedure)) - _journey_value_names(journey)
    if unknown_values:
        issues.append(
            "The procedure requires undeclared value name(s): "
            + ", ".join(sorted(unknown_values))
            + "."
        )
    return issues


def _requirement_evidence(
    journey: ResolvedJourney,
    result: DiscoveryResult,
    procedure: ReplayProcedure | None,
    evidence: EvidenceDocument | None,
    issues: list[str],
) -> tuple[JourneyRequirementEvidence, ...]:
    objective = result.request.objectives[0]
    completion_conditions = (
        evidence.task.completion_conditions
        if evidence is not None
        else objective.completion_conditions
    )
    coverage = result.coverage.entries[0]
    if issues or procedure is None or evidence is None:
        reason = " ".join(dict.fromkeys(issues)) or "Discovery evidence is incomplete."
        return tuple(
            _unresolved_requirement(journey.journey_id, index, statement, reason)
            for index, statement in enumerate((journey.objective, *completion_conditions), start=-1)
        )
    requirements = [
        JourneyRequirementEvidence(
            requirement_id=_derived_identifier("requirement", journey.journey_id, "objective"),
            kind="objective",
            statement=journey.objective,
            state=RequirementEvidenceState.SUPPORTED,
            observation_ids=coverage.evidence_observation_ids,
            procedure_step_indexes=tuple(step.procedure_index for step in procedure.steps),
            reason="The compiled procedure reached the independently verified journey state.",
        )
    ]
    evidence_by_source_step = {step.step: step for step in evidence.steps}
    for condition_index, statement in enumerate(completion_conditions):
        procedure_steps = tuple(
            step
            for step in procedure.steps
            if condition_index in step.captures_completion_conditions
        )
        observation_ids = tuple(
            dict.fromkeys(
                destination_id
                for step in procedure_steps
                if (
                    destination_id := evidence_by_source_step[
                        step.source_step
                    ].destination_observation_id
                )
                is not None
            )
        )
        if not procedure_steps or not observation_ids:
            requirements.append(
                _unresolved_requirement(
                    journey.journey_id,
                    condition_index,
                    statement,
                    "No compiled milestone provides observable evidence for this condition.",
                )
            )
            continue
        requirements.append(
            JourneyRequirementEvidence(
                requirement_id=_derived_identifier(
                    "requirement", journey.journey_id, str(condition_index)
                ),
                kind="completion_condition",
                statement=statement,
                state=RequirementEvidenceState.SUPPORTED,
                observation_ids=observation_ids,
                procedure_step_indexes=tuple(step.procedure_index for step in procedure_steps),
                reason="Compiled milestones provide observable evidence for this condition.",
            )
        )
    if any(item.state is RequirementEvidenceState.UNRESOLVED for item in requirements):
        issues.append(
            "At least one journey completion condition lacks compiled milestone evidence."
        )
    return tuple(requirements)


def _contains_requested_conditions(
    compiled_conditions: tuple[str, ...],
    requested_conditions: tuple[str, ...],
) -> bool:
    """Return whether compilation preserved every requested condition in order."""
    compiled_iterator = iter(compiled_conditions)
    return all(
        any(compiled_condition == requested_condition for compiled_condition in compiled_iterator)
        for requested_condition in requested_conditions
    )


def _unresolved_requirement(
    journey_id: RecordIdentifier,
    condition_index: int,
    statement: str,
    reason: str,
) -> JourneyRequirementEvidence:
    is_objective = condition_index == -1
    return JourneyRequirementEvidence(
        requirement_id=_derived_identifier(
            "requirement", journey_id, "objective" if is_objective else str(condition_index)
        ),
        kind="objective" if is_objective else "completion_condition",
        statement=statement,
        state=RequirementEvidenceState.UNRESOLVED,
        reason=reason,
    )


def _extract_test_surfaces(
    journey: ResolvedJourney,
    result: DiscoveryResult,
    procedure: ReplayProcedure,
    evidence: EvidenceDocument,
) -> tuple[DiscoveredTestSurface, ...]:
    evidence_by_source_step = {step.step: step for step in evidence.steps}
    grouped: dict[tuple[ScenarioSurfaceKind, str], _SurfaceAccumulator] = {}
    for procedure_step in procedure.steps:
        evidence_step = evidence_by_source_step[procedure_step.source_step]
        kind = _surface_kind(procedure_step, evidence_step)
        if kind is None:
            continue
        subject = _surface_subject(evidence_step)
        key = (kind, subject.casefold())
        accumulator = grouped.setdefault(key, _SurfaceAccumulator(kind=kind, subject=subject))
        accumulator.outcomes.extend(_observed_outcomes(evidence_step))
        accumulator.observation_ids.extend(
            item
            for item in (
                evidence_step.source_observation_id,
                evidence_step.destination_observation_id,
            )
            if item is not None
        )
        accumulator.state_ids.extend(
            item
            for item in (evidence_step.source_state_id, evidence_step.destination_state_id)
            if item is not None
        )
        if isinstance(procedure_step.proposal.text, DeclaredValueReference):
            accumulator.declared_value_names.append(procedure_step.proposal.text.value_name)
        accumulator.procedure_step_indexes.append(procedure_step.procedure_index)
    verification = result.verification
    if verification is None:
        raise AssertionError("Ready discovery evidence requires final verification.")
    action_surfaces = tuple(
        DiscoveredTestSurface(
            surface_id=_derived_identifier(
                "surface", journey.journey_id, procedure.procedure_id, kind.value, subject
            ),
            journey_id=journey.journey_id,
            kind=kind,
            summary=f"Discovery exercised {subject} as a {kind.value.replace('_', ' ')} surface.",
            observable_outcomes=tuple(dict.fromkeys(accumulator.outcomes)),
            state_ids=tuple(dict.fromkeys(accumulator.state_ids)),
            declared_value_names=tuple(dict.fromkeys(accumulator.declared_value_names)),
            provenance=DiscoveryProvenance(
                discovery_run_id=procedure.source_run_id,
                discovery_objective_id=result.request.objectives[0].objective_id,
                procedure_id=procedure.procedure_id,
                verification_id=verification.verification_id,
                observation_ids=tuple(dict.fromkeys(accumulator.observation_ids)),
                procedure_step_indexes=tuple(dict.fromkeys(accumulator.procedure_step_indexes)),
            ),
        )
        for (kind, _), accumulator in grouped.items()
        for subject in (accumulator.subject,)
        if accumulator.outcomes
    )
    return (*action_surfaces, *_presentation_surfaces(journey, result, procedure, evidence))


def _presentation_surfaces(
    journey: ResolvedJourney,
    result: DiscoveryResult,
    procedure: ReplayProcedure,
    evidence: EvidenceDocument,
) -> tuple[DiscoveredTestSurface, ...]:
    evidence_by_source_step = {step.step: step for step in evidence.steps}
    step_indexes_by_state: dict[RecordIdentifier, list[int]] = {}
    ordered_state_ids: list[RecordIdentifier] = []
    for procedure_step in procedure.steps:
        evidence_step = evidence_by_source_step[procedure_step.source_step]
        for state_id in (evidence_step.source_state_id, evidence_step.destination_state_id):
            if state_id is None:
                continue
            if state_id not in step_indexes_by_state:
                ordered_state_ids.append(state_id)
            step_indexes_by_state.setdefault(state_id, []).append(procedure_step.procedure_index)
    verification = result.verification
    if verification is None:
        raise AssertionError("Ready discovery evidence requires final verification.")
    return tuple(
        _presentation_surface(
            journey,
            result,
            procedure,
            verification.verification_id,
            evidence.states[state_id],
            tuple(dict.fromkeys(step_indexes_by_state[state_id])),
        )
        for state_id in ordered_state_ids
    )


def _presentation_surface(
    journey: ResolvedJourney,
    result: DiscoveryResult,
    procedure: ReplayProcedure,
    verification_id: RecordIdentifier,
    state: EvidenceState,
    procedure_step_indexes: tuple[int, ...],
) -> DiscoveredTestSurface:
    return DiscoveredTestSurface(
        surface_id=_derived_identifier(
            "surface",
            journey.journey_id,
            procedure.procedure_id,
            ScenarioSurfaceKind.PRESENTATION.value,
            state.state_id,
        ),
        journey_id=journey.journey_id,
        kind=ScenarioSurfaceKind.PRESENTATION,
        summary=f"Discovery observed the application presentation for state {state.state_id}.",
        observable_outcomes=(
            "Accepted discovery evidence contains a settled presentation for this state.",
        ),
        state_ids=(state.state_id,),
        provenance=DiscoveryProvenance(
            discovery_run_id=procedure.source_run_id,
            discovery_objective_id=result.request.objectives[0].objective_id,
            procedure_id=procedure.procedure_id,
            verification_id=verification_id,
            observation_ids=state.observation_ids,
            procedure_step_indexes=procedure_step_indexes,
        ),
    )


def _surface_kind(
    procedure_step: ReplayProcedureStep,
    evidence_step: EvidenceStep,
) -> ScenarioSurfaceKind | None:
    action = procedure_step.proposal.action
    if action in {ActionKind.WAIT, ActionKind.DISMISS_KEYBOARD, ActionKind.SCROLL}:
        return None
    if action in {ActionKind.FOCUS, ActionKind.TYPE, ActionKind.CLEAR}:
        return ScenarioSurfaceKind.INPUT
    if evidence_step.control.value_type in {ControlValueType.BOOLEAN, ControlValueType.ENUM}:
        return ScenarioSurfaceKind.SELECTION
    return {
        ProcedureEffect.NAVIGATION: ScenarioSurfaceKind.NAVIGATION,
        ProcedureEffect.STATE_SETTING: ScenarioSurfaceKind.STATE_CHANGE,
        ProcedureEffect.COMMIT: ScenarioSurfaceKind.PERSISTENCE_BOUNDARY,
        ProcedureEffect.ENVIRONMENT: ScenarioSurfaceKind.SYSTEM_INTERACTION,
        ProcedureEffect.UNKNOWN: ScenarioSurfaceKind.OPERATION,
    }[procedure_step.procedure_effect]


def _surface_subject(step: EvidenceStep) -> str:
    return next(
        (
            value.strip()
            for value in (
                step.control.label,
                step.control.identity_key,
                step.proposal.target_expression,
            )
            if value is not None and value.strip()
        ),
        step.proposal.action.value.replace("_", " "),
    )


def _observed_outcomes(step: EvidenceStep) -> tuple[str, ...]:
    outcomes = [
        f"The accepted action produced the observed {signal.value.replace('_', ' ')} signal."
        for signal in sorted(step.signals, key=lambda item: item.value)
    ]
    if step.destination_state_id is not None:
        outcomes.append("The accepted action reached a distinct observed application state.")
    return tuple(outcomes)


def _blocking_action_risk_ids(
    procedure: ReplayProcedure | None,
    memory: AppMemory | None,
) -> tuple[RecordIdentifier, ...]:
    if procedure is None or memory is None:
        return ()
    source_step_ids = set(procedure.source_step_ids)
    return tuple(
        risk.action_risk_id
        for risk in memory.app_knowledge.action_risks
        if any(source.step_id in source_step_ids for source in risk.sources)
    )


def _required_value_names(procedure: ReplayProcedure | None) -> tuple[RecordIdentifier, ...]:
    if procedure is None:
        return ()
    return tuple(
        dict.fromkeys(
            step.proposal.text.value_name
            for step in procedure.steps
            if isinstance(step.proposal.text, DeclaredValueReference)
        )
    )


def _journey_value_names(journey: ResolvedJourney) -> set[str]:
    names = set(journey.declared_values)
    names.update(value_name for actor in journey.actors.values() for value_name in actor)
    return names


def _derived_identifier(prefix: str, *parts: str) -> str:
    digest = sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[: 63 - len(prefix)]}"
