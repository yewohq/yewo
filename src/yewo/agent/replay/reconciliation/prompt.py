"""Build a bounded semantic projection for replay-reconciliation models."""

from __future__ import annotations

from typing import Any

from yewo.agent.replay.reconciliation.records import (
    EvidenceDocument,
    EvidenceState,
    EvidenceStep,
    REUSABLE_EVIDENCE_RESULTS,
)


def build_reconciler_evidence_projection(evidence: EvidenceDocument) -> dict[str, Any]:
    """Keep model-relevant semantics while the compiler retains lossless evidence."""
    task = evidence.task.model_dump(mode="json")
    completion_conditions = tuple(
        {"index": index, "condition": condition}
        for index, condition in enumerate(evidence.task.completion_conditions)
    )
    task["completion_conditions"] = completion_conditions
    task["allowed_milestone_indices"] = tuple(range(len(completion_conditions)))
    task["completion_condition_mode"] = (
        "supplied" if completion_conditions else "generate_from_discovery_evidence"
    )
    candidate_steps = tuple(
        step for step in evidence.steps if step.result in REUSABLE_EVIDENCE_RESULTS
    )
    return {
        "projection_version": 3,
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
            "candidate_step_count": len(candidate_steps),
            "observation_count": evidence.run.observation_count,
            "launch_state_id": evidence.run.launch_state_id,
        },
        "task": task,
        "states": {state_id: _project_state(state) for state_id, state in evidence.states.items()},
        "steps": tuple(_project_step(step) for step in candidate_steps),
        "diagnostics": evidence.diagnostics.model_dump(mode="json"),
    }


def _project_state(state: EvidenceState) -> dict[str, Any]:
    presentation = state.representative_presentation
    return {
        "state_id": state.state_id,
        "foreground_application_id": presentation.foreground_application_id,
        "platform_context": presentation.platform_context,
        "keyboard_visible": presentation.keyboard_visible,
        "orientation": (
            presentation.orientation.value if presentation.orientation is not None else None
        ),
        "width_px": presentation.width_px,
        "height_px": presentation.height_px,
        "observation_count": len(state.observation_ids),
        "visual_variant_count": len(state.comparison_fingerprints()),
    }


def _project_step(step: EvidenceStep) -> dict[str, Any]:
    proposal = step.proposal
    proposal_json = proposal.model_dump(mode="json")
    return {
        "step": step.step,
        "source_step_id": step.source_step_id,
        "source_state_id": step.source_state_id,
        "destination_state_id": step.destination_state_id,
        "kind": step.kind.value,
        "action": proposal.action.value,
        "proposal": {
            "target_expression": proposal_json["target_expression"],
            "text": proposal_json["text"],
            "destination_expression": proposal_json["destination_expression"],
            "direction": proposal.direction.value if proposal.direction is not None else None,
            "distance": proposal.distance.value if proposal.distance is not None else None,
            "duration_seconds": proposal.duration_seconds,
            "expected_outcome": proposal.expected_outcome,
            "reason": proposal.reason,
            "grounding_correction_for_proposal_id": (proposal.grounding_correction_for_proposal_id),
            "interaction_refinement_for_proposal_id": (
                proposal.interaction_refinement_for_proposal_id
            ),
        },
        "control": {
            "identity_key": step.control.identity_key,
            "label": step.control.label,
            "kind": step.control.kind.model_dump(mode="json"),
            "kind_source": step.control.kind_source,
            "value_type": step.control.value_type.value,
            "input_traits": (
                step.control.input_traits.model_dump(mode="json")
                if step.control.input_traits is not None
                else None
            ),
        },
        "result": step.result.value,
        "refusal_code": step.refusal_code.value if step.refusal_code is not None else None,
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
    }
