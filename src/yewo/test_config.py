"""Validated configuration for one journey and target test run."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from yewo.agent.config import validate_agent_configuration
from yewo.agent.runtime.records import AgentModePolicy
from yewo.artifacts import RecordIdentifier
from yewo.grounding import GroundingProfile
from yewo.models import ModelProfileRegistry
from yewo.observations import RedactionRegion
from yewo.project_config.records import ResolvedJourney
from yewo.runs import RunMode, RunRequest


class YewoTestConfiguration(BaseModel):
    """Everything required to discover and test one journey on one target."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run: RunRequest
    workflow_name: str
    target_id: RecordIdentifier
    journey: ResolvedJourney
    discovery_mode_policy: AgentModePolicy
    scenario_mode_policy: AgentModePolicy
    model_profiles: ModelProfileRegistry
    grounding_profile: GroundingProfile
    planner_profile_id: RecordIdentifier
    auditor_profile_id: RecordIdentifier
    recovery_profile_id: RecordIdentifier
    action_verifier_profile_id: RecordIdentifier
    replay_preparation_profile_id: RecordIdentifier
    screen_analyzer_profile_id: RecordIdentifier
    test_designer_profile_id: RecordIdentifier
    memory_root: Path
    redaction_regions: tuple[RedactionRegion, ...] = ()

    @field_validator("memory_root")
    @classmethod
    def _validate_memory_root(cls, value: Path) -> Path:
        if not value.parts or value.name in {"", ".", ".."}:
            raise ValueError("A Yewo test memory root must name a directory.")
        return value

    @model_validator(mode="after")
    def _validate_composition(self) -> "YewoTestConfiguration":
        if self.run.mode is not RunMode.SCENARIO:
            raise ValueError("A Yewo test configuration requires scenario mode.")
        validate_agent_configuration(
            self.discovery_mode_policy,
            self.model_profiles,
            self.grounding_profile,
            self.planner_profile_id,
            self.auditor_profile_id,
            self.recovery_profile_id,
            self.action_verifier_profile_id,
            self.replay_preparation_profile_id,
            self.screen_analyzer_profile_id,
            self.test_designer_profile_id,
        )
        validate_agent_configuration(
            self.scenario_mode_policy,
            self.model_profiles,
            self.grounding_profile,
            self.planner_profile_id,
            self.auditor_profile_id,
            self.recovery_profile_id,
            self.action_verifier_profile_id,
            self.replay_preparation_profile_id,
            self.screen_analyzer_profile_id,
            self.test_designer_profile_id,
        )
        return self
