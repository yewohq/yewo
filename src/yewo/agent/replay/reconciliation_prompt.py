"""Build a bounded semantic projection for replay-reconciliation models."""

from __future__ import annotations

from typing import Any

from yewo.agent.replay.reconciliation_records import EvidenceDocument, EvidenceState, EvidenceStep


def build_reconciler_evidence_projection(evidence: EvidenceDocument) -> dict[str, Any]:
    """Keep model-relevant semantics while the compiler retains lossless evidence."""
    return {
        "projection_version": 1,
        "schema_version": evidence.schema_version,
        "evidence_id": evidence.evidence_id,
        "run": {
            "run_id": evidence.run.run_id,
            "application": evidence.run.application.model_dump(mode="json"),
            "device": (
                evidence.run.device.model_dump(mode="json")
                if evidence.run.device is not None
                else None
            ),
            "step_count": evidence.run.step_count,
            "observation_count": evidence.run.observation_count,
            "launch_state_id": evidence.run.launch_state_id,
        },
        "task": evidence.task.model_dump(mode="json"),
        "states": {
            state_id: _project_state(state) for state_id, state in evidence.states.items()
        },
        "steps": tuple(_project_step(step) for step in evidence.steps),
        "diagnostics": evidence.diagnostics.model_dump(mode="json"),
    }


def _project_state(state: EvidenceState) -> dict[str, Any]:
    return {
        "state_id": state.state_id,
        "foreground_application_id": state.foreground_application_id,
        "platform_context": state.platform_context,
        "keyboard_visible": state.keyboard_visible,
        "orientation": state.orientation.value if state.orientation is not None else None,
        "width_px": state.width_px,
        "height_px": state.height_px,
        "observation_count": len(state.observation_ids),
        "visual_variant_count": len(state.comparison_fingerprints()),
    }


def _project_step(step: EvidenceStep) -> dict[str, Any]:
    proposal = step.proposal
    proposal_json = proposal.model_dump(mode="json")
    return {
        "step": step.step,
        "recorded_step_index": step.recorded_step_index,
        "source_step_id": step.source_step_id,
        "source_state_id": step.source_state_id,
        "destination_state_id": step.destination_state_id,
        "kind": step.kind.value,
        "action": step.action.value,
        "proposal": {
            "target_expression": proposal_json["target_expression"],
            "text": proposal_json["text"],
            "destination_expression": proposal_json["destination_expression"],
            "direction": proposal.direction.value if proposal.direction is not None else None,
            "distance": proposal.distance.value if proposal.distance is not None else None,
            "duration_seconds": proposal.duration_seconds,
            "expected_outcome": proposal.expected_outcome,
            "reason": proposal.reason,
            "screen_summary": proposal.screen_summary,
            "grounding_correction_for_proposal_id": (
                proposal.grounding_correction_for_proposal_id
            ),
            "interaction_refinement_for_proposal_id": (
                proposal.interaction_refinement_for_proposal_id
            ),
        },
        "control": {
            "identity_key": step.control.identity_key,
            "label": step.control.label,
            "role": step.control.role.value,
            "value_type": step.control.value_type.value,
            "input_traits": (
                step.control.input_traits.model_dump(mode="json")
                if step.control.input_traits is not None
                else None
            ),
        },
        "result": step.result.value,
        "outcome_disposition": (
            step.outcome_disposition.value if step.outcome_disposition is not None else None
        ),
        "signals": tuple(sorted(signal.value for signal in step.signals)),
        "text_delivery": (
            step.text_delivery.model_dump(mode="json") if step.text_delivery is not None else None
        ),
        "authorization_side_effect": (
            step.authorization_side_effect.value
            if step.authorization_side_effect is not None
            else None
        ),
        "refused": step.refused,
        "application_failed": step.application_failed,
    }
