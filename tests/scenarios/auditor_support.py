"""Shared structured findings for scenario-auditor tests."""

from yewo.scenarios.auditor_findings import AuditCandidateFindingOutput


def visual_finding(
    outcome_id: str,
    observation_id: str,
) -> AuditCandidateFindingOutput:
    """Return one located visual-overflow finding."""
    return AuditCandidateFindingOutput.model_validate(
        {
            "outcome_id": outcome_id,
            "claim": "A visible overflow diagnostic is present.",
            "regions": [
                {
                    "observation_id": observation_id,
                    "label": "Visible overflow diagnostic",
                    "left": 100,
                    "top": 100,
                    "right": 900,
                    "bottom": 200,
                }
            ],
            "verification": {"kind": "visual"},
        }
    )
