"""Static strategy and evidence rules for scenario generation."""

from yewo.project_config import CoverageCategory, ScenarioType
from yewo.scenarios.taxonomy import RequiredEvidenceOwner, ScenarioExecutionChange


AUTOMATIC_STRATEGY = {
    CoverageCategory.DATA_CONSISTENCY: ScenarioType.HAPPY_PATH,
    CoverageCategory.STATE: ScenarioType.HAPPY_PATH,
    CoverageCategory.INPUT: ScenarioType.EMPTY,
    CoverageCategory.NAVIGATION: ScenarioType.HAPPY_PATH,
    CoverageCategory.ERROR_HANDLING: ScenarioType.NEGATIVE,
    CoverageCategory.UI_VISUAL: ScenarioType.EXPLORATORY,
    CoverageCategory.SECURITY: ScenarioType.NEGATIVE,
    CoverageCategory.ACCESSIBILITY: ScenarioType.HAPPY_PATH,
    CoverageCategory.PERFORMANCE: ScenarioType.HAPPY_PATH,
    CoverageCategory.NETWORK: ScenarioType.INTERRUPTION,
    CoverageCategory.LIFECYCLE: ScenarioType.INTERRUPTION,
    CoverageCategory.PERMISSION: ScenarioType.NEGATIVE,
    CoverageCategory.COMPATIBILITY: ScenarioType.HAPPY_PATH,
    CoverageCategory.INSTALLATION: ScenarioType.RECOVERY,
    CoverageCategory.LOCALIZATION: ScenarioType.BOUNDARY,
    CoverageCategory.REGRESSION: ScenarioType.HAPPY_PATH,
}
VARIATION_BY_EXECUTION_CHANGE = {
    ScenarioExecutionChange.ACCEPTED_ROUTE: "Replay the accepted discovery route.",
    ScenarioExecutionChange.INPUT_VARIATION: "Change the selected input action or value.",
    ScenarioExecutionChange.ENVIRONMENT_VARIATION: "Change the selected device environment.",
    ScenarioExecutionChange.LIFECYCLE_INTERRUPTION: (
        "Interrupt the application lifecycle and then continue."
    ),
    ScenarioExecutionChange.PERMISSION_DECISION: "Apply the selected native permission decision.",
    ScenarioExecutionChange.RECOVERY_ATTEMPT: "Apply the selected recovery action.",
    ScenarioExecutionChange.ACTOR_VARIATION: "Use the selected alternate actor.",
    ScenarioExecutionChange.CONCURRENT_OPERATION: "Apply the selected concurrent operation.",
    ScenarioExecutionChange.INSTALLATION_CHANGE: "Apply the selected installation change.",
}
INVARIANT_BY_CATEGORY = {
    CoverageCategory.DATA_CONSISTENCY: "Derived values must match their observed source values.",
    CoverageCategory.STATE: "The tested action must produce an observable application state.",
    CoverageCategory.NAVIGATION: "The tested navigation must produce an observable destination.",
    CoverageCategory.ERROR_HANDLING: "A rejected operation must not be reported as successful.",
    CoverageCategory.UI_VISUAL: "Required content must remain observable in the tested state.",
    CoverageCategory.SECURITY: "Sensitive content must not become newly exposed.",
    CoverageCategory.ACCESSIBILITY: (
        "Required interactive content must be available to accessibility inspection."
    ),
    CoverageCategory.NETWORK: "A controlled network change must produce an observable outcome.",
    CoverageCategory.LIFECYCLE: (
        "A controlled lifecycle change must produce an observable outcome."
    ),
    CoverageCategory.PERMISSION: (
        "A controlled permission decision must produce an observable outcome."
    ),
}
EVIDENCE_DESCRIPTIONS = {
    RequiredEvidenceOwner.APPLICATION_OBSERVATION: (
        "A settled application observation must show the expected outcome."
    ),
    RequiredEvidenceOwner.ACTION_OUTCOME: (
        "The action outcome must show whether the intended effect occurred."
    ),
    RequiredEvidenceOwner.DEVICE_RECEIPT: (
        "A device receipt must record the controlled device-state change."
    ),
    RequiredEvidenceOwner.APPLICATION_GROUND_TRUTH: (
        "Application ground truth must show the expected value."
    ),
    RequiredEvidenceOwner.MEASUREMENT: "A recorded measurement must support the result.",
    RequiredEvidenceOwner.ACCESSIBILITY_INSPECTION: (
        "Accessibility inspection must expose the required interactive content."
    ),
}
