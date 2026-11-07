"""Prompt payload projection for independent scenario audit."""

from __future__ import annotations

from yewo.scenarios.auditor_plan import AuditRequestBatch
from yewo.scenarios.records import ScenarioDefinition


def build_audit_payload(
    scenario: ScenarioDefinition,
    batch: AuditRequestBatch,
) -> dict[str, object]:
    """Project one bounded audit request without dropping coverage metadata."""
    return {
        "scenario": {
            "scenario_id": scenario.scenario_id,
            "objective": scenario.objective,
            "variation": scenario.variation,
            "execution_change": scenario.execution_change.value,
        },
        "expected_outcomes": [
            {
                "outcome_id": target.outcome.outcome_id,
                "statement": target.outcome.statement,
                "authority": target.outcome.authority.value,
                "category": target.category.value,
                "required_observation_ids": [
                    observation_id
                    for image in target.evidence.images
                    for observation_id in image.observation_ids
                ],
            }
            for target in batch.targets
        ],
        "audit_scope": {
            "batch_index": batch.batch_index,
            "batch_count": batch.batch_count,
            "universal": batch.universal,
            "visual_episode_id": batch.visual_episode_id,
            "visual_episode_label": batch.visual_episode_label,
            "required_surface_count": len(batch.evidence.required_surface_ids),
            "covered_surface_count": len(batch.evidence.covered_surface_ids),
            "missing_surface_ids": batch.evidence.missing_surface_ids,
        },
        "action_facts": batch.evidence.action_facts,
        "capability_facts": batch.evidence.capability_facts,
    }
