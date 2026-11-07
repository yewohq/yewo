"""Resolve concrete execution changes for generated testing obligations."""

from yewo.project_config import CoverageCategory, ScenarioType
from yewo.scenarios.designer import ValidatedTestDesign
from yewo.scenarios.taxonomy import ScenarioExecutionChange


def resolve_automatic_execution_change(
    design: ValidatedTestDesign,
    category: CoverageCategory,
    scenario_type: ScenarioType,
) -> ScenarioExecutionChange:
    """Return the model-selected or deterministic execution change."""
    interpreted = next(
        (
            variation.execution_change
            for item in design.category_interpretations
            if item.category is category
            for variation in item.variations
            if variation.scenario_type is scenario_type
        ),
        None,
    )
    if interpreted is not None:
        return interpreted
    if scenario_type in {ScenarioType.HAPPY_PATH, ScenarioType.EXPLORATORY}:
        return ScenarioExecutionChange.ACCEPTED_ROUTE
    if category is CoverageCategory.LIFECYCLE:
        return ScenarioExecutionChange.LIFECYCLE_INTERRUPTION
    if category is CoverageCategory.PERMISSION:
        return ScenarioExecutionChange.PERMISSION_DECISION
    if category is CoverageCategory.INSTALLATION:
        return ScenarioExecutionChange.INSTALLATION_CHANGE
    if scenario_type is ScenarioType.RECOVERY:
        return ScenarioExecutionChange.RECOVERY_ATTEMPT
    if scenario_type is ScenarioType.ROLE:
        return ScenarioExecutionChange.ACTOR_VARIATION
    if scenario_type is ScenarioType.CONCURRENCY:
        return ScenarioExecutionChange.CONCURRENT_OPERATION
    if category in {CoverageCategory.NETWORK, CoverageCategory.LOCALIZATION}:
        return ScenarioExecutionChange.ENVIRONMENT_VARIATION
    return ScenarioExecutionChange.INPUT_VARIATION
