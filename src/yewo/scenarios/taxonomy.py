"""Stable testing vocabulary shared by project files and generated scenarios."""

from enum import StrEnum


class ScenarioSource(StrEnum):
    """Authority that requested one scenario."""

    USER_REQUIREMENT = "user_requirement"
    AUTOMATIC_QA = "automatic_qa"
    EXPLORATORY_SUGGESTION = "exploratory_suggestion"


class ScenarioExecutionChange(StrEnum):
    """Concrete execution mechanics that differ from the accepted route."""

    ACCEPTED_ROUTE = "accepted_route"
    INPUT_VARIATION = "input_variation"
    ENVIRONMENT_VARIATION = "environment_variation"
    LIFECYCLE_INTERRUPTION = "lifecycle_interruption"
    PERMISSION_DECISION = "permission_decision"
    RECOVERY_ATTEMPT = "recovery_attempt"
    ACTOR_VARIATION = "actor_variation"
    CONCURRENT_OPERATION = "concurrent_operation"
    INSTALLATION_CHANGE = "installation_change"


class ScenarioSurfaceKind(StrEnum):
    """A discovered application characteristic to which QA lenses may apply."""

    OPERATION = "operation"
    INPUT = "input"
    SELECTION = "selection"
    NAVIGATION = "navigation"
    STATE_CHANGE = "state_change"
    PERSISTENCE_BOUNDARY = "persistence_boundary"
    ERROR_STATE = "error_state"
    RECOVERY_PATH = "recovery_path"
    SYSTEM_INTERACTION = "system_interaction"
    PRESENTATION = "presentation"


class RequirementEvidenceState(StrEnum):
    """Whether discovery supports one journey requirement."""

    SUPPORTED = "supported"
    UNRESOLVED = "unresolved"


class ExpectedOutcomeAuthority(StrEnum):
    """Why an expected outcome is allowed to define correct behavior."""

    USER_REQUIREMENT = "user_requirement"
    DECLARED_RULE = "declared_rule"
    DETERMINISTIC_INVARIANT = "deterministic_invariant"
    ACCEPTED_BASELINE = "accepted_baseline"


class RequiredEvidenceOwner(StrEnum):
    """System that must provide evidence for an expected outcome."""

    APPLICATION_OBSERVATION = "application_observation"
    ACTION_OUTCOME = "action_outcome"
    DEVICE_RECEIPT = "device_receipt"
    APPLICATION_GROUND_TRUTH = "application_ground_truth"
    MEASUREMENT = "measurement"
    ACCESSIBILITY_INSPECTION = "accessibility_inspection"
