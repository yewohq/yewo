"""Map generated expected outcomes to exact discovery surfaces."""

from yewo.artifacts import RecordIdentifier
from yewo.project_config import CoverageCategory
from yewo.scenarios.designer import ValidatedTestDesign
from yewo.scenarios.records import JourneyRequirementEvidence, ScenarioSuite


def requirement_surface_ids(
    suite: ScenarioSuite,
    candidate_ids: tuple[RecordIdentifier, ...],
    requirement: JourneyRequirementEvidence,
) -> tuple[RecordIdentifier, ...]:
    """Return candidate surfaces that contain accepted requirement evidence."""
    observations = set(requirement.observation_ids)
    indexes = set(requirement.procedure_step_indexes)
    matching_surface_ids = tuple(
        surface.surface_id
        for surface in suite.surfaces
        if surface.surface_id in candidate_ids
        and (
            observations.intersection(surface.provenance.observation_ids)
            or indexes.intersection(surface.provenance.procedure_step_indexes)
        )
    )
    return matching_surface_ids or candidate_ids


def interpreted_surface_ids(
    design: ValidatedTestDesign,
    category: CoverageCategory,
) -> tuple[RecordIdentifier, ...]:
    """Return unique surfaces selected by accepted category interpretations."""
    return tuple(
        dict.fromkeys(
            surface_id
            for item in design.category_interpretations
            if item.category is category
            for surface_id in item.surface_ids
        )
    )
