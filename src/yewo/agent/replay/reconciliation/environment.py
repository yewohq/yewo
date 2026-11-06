"""Validate and compile discovery-proven replay environment requirements."""

from yewo.agent.actuation import ActionKind
from yewo.artifacts import EvidenceDeviceIdentity
from yewo.agent.replay.reconciliation.records import (
    EvidenceDocument,
    EvidenceResult,
    EvidenceStep,
    EvidenceStepKind,
    ProcedureSetupContract,
    RuntimePermissionSetupRequirement,
)
from yewo.devices.application_lifecycle import TrustedSystemSurface, trusted_system_surface


def environment_step_refusal_reason(
    evidence: EvidenceDocument,
    step: EvidenceStep,
    previous: EvidenceStep | None,
) -> str | None:
    """Reject every environment action except one proven permission transition."""
    if step.kind is not EvidenceStepKind.ENVIRONMENT:
        return None
    device = evidence.run.device
    source = evidence.states[step.source_state_id]
    destination = (
        evidence.states[step.destination_state_id]
        if step.destination_state_id is not None
        else None
    )
    surface = (
        trusted_system_surface(device.platform, source.foreground_application_id)
        if device is not None
        else None
    )
    if surface is not TrustedSystemSurface.RUNTIME_PERMISSION:
        return "Only a recorded Android runtime-permission transition is reusable."
    if step.action is not ActionKind.TAP or step.result is not EvidenceResult.OBSERVED:
        return "A runtime-permission transition requires one observed tap action."
    if previous is None or previous.destination_state_id != step.source_state_id:
        return "A runtime-permission transition requires its selected application trigger."
    previous_source = evidence.states[previous.source_state_id]
    if previous_source.foreground_application_id != evidence.run.application.application_id:
        return "The permission trigger must originate in the recorded application."
    if (
        destination is None
        or destination.foreground_application_id != evidence.run.application.application_id
    ):
        return "A runtime-permission transition must return to the recorded application."
    return None


def compile_setup_contract(
    evidence: EvidenceDocument,
    selected: tuple[EvidenceStep, ...],
    device: EvidenceDeviceIdentity,
) -> ProcedureSetupContract:
    """Build one canonical setup contract from validated source evidence."""
    runtime_permissions = tuple(
        RuntimePermissionSetupRequirement(
            trigger_source_step=selected[index - 1].step,
            interaction_source_step=step.step,
            prompt_state_id=step.source_state_id,
            return_state_id=step.destination_state_id,
        )
        for index, step in enumerate(selected)
        if step.kind is EvidenceStepKind.ENVIRONMENT
        and index > 0
        and step.destination_state_id is not None
    )
    return ProcedureSetupContract(
        device=device,
        launch_state_id=evidence.run.launch_state_id,
        runtime_permissions=runtime_permissions,
    )
