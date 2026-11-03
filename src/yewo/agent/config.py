"""Validated configuration for one Yewo task run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    field_validator,
    model_validator,
)

from yewo.agent.actuation import ACTION_DEFINITIONS
from yewo.agent.runtime.records import AgentModePolicy, AgentTask
from yewo.artifacts import RecordIdentifier
from yewo.grounding import (
    GroundingOutputShape,
    GroundingProfile,
    GroundingTargetOperation,
)
from yewo.models import ModelProfileRegistry, ModelRole
from yewo.observations import RedactionRegion
from yewo.runs import RunMode, RunRequest


class YewoRunConfigurationError(ValueError):
    """A Yewo run configuration could not be loaded safely."""


class YewoRunConfiguration(BaseModel):
    """All explicit configuration required for one Yewo device task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run: RunRequest
    task: AgentTask
    mode_policy: AgentModePolicy
    model_profiles: ModelProfileRegistry
    grounding_profile: GroundingProfile
    planner_profile_id: RecordIdentifier
    auditor_profile_id: RecordIdentifier
    recovery_profile_id: RecordIdentifier
    action_verifier_profile_id: RecordIdentifier
    replay_preparation_profile_id: RecordIdentifier
    memory_root: Path | None = None
    redaction_regions: tuple[RedactionRegion, ...] = ()

    @field_validator("memory_root")
    @classmethod
    def _validate_memory_root(cls, value: Path | None) -> Path | None:
        if value is not None and (not value.parts or value.name in {"", ".", ".."}):
            raise ValueError("A Yewo memory root must name a directory.")
        return value

    @model_validator(mode="after")
    def _validate_composition(self) -> Self:
        if self.run.mode is not RunMode.NORMAL:
            raise ValueError("A Yewo run configuration requires normal mode.")
        if self.task.declared_value_names:
            raise ValueError("The normal CLI does not yet accept declared sensitive values.")
        validate_agent_configuration(
            self.mode_policy,
            self.model_profiles,
            self.grounding_profile,
            self.planner_profile_id,
            self.auditor_profile_id,
            self.recovery_profile_id,
            self.action_verifier_profile_id,
            self.replay_preparation_profile_id,
        )
        return self


def validate_agent_configuration(
    mode_policy: AgentModePolicy,
    model_profiles: ModelProfileRegistry,
    grounding_profile: GroundingProfile,
    planner_profile_id: RecordIdentifier,
    auditor_profile_id: RecordIdentifier,
    recovery_profile_id: RecordIdentifier,
    action_verifier_profile_id: RecordIdentifier,
    replay_preparation_profile_id: RecordIdentifier,
) -> None:
    """Validate model roles and grounding coverage shared by agent workflows."""
    try:
        model_profiles.resolve(planner_profile_id, ModelRole.TASK_PLANNER)
        model_profiles.resolve(auditor_profile_id, ModelRole.AUDITOR)
        recovery_profile = model_profiles.resolve(
            recovery_profile_id, ModelRole.RECOVERY_ADVISOR
        )
        model_profiles.resolve(action_verifier_profile_id, ModelRole.ACTION_VERIFIER)
        model_profiles.resolve(replay_preparation_profile_id, ModelRole.REPLAY_PREPARER)
        model_profiles.resolve(grounding_profile.model_profile_id, ModelRole.GROUNDER)
    except LookupError as error:
        raise ValueError("The run references an incompatible model profile.") from error
    if recovery_profile.images.max_images < 2:
        raise ValueError("The recovery profile must allow at least two images.")
    available_actions = (
        mode_policy.action_limits.allowed_actions & mode_policy.action_limits.supported_actions
    )
    for action in available_actions:
        definition = ACTION_DEFINITIONS[action]
        for operation in (
            definition.target_operation,
            definition.destination_operation,
        ):
            if operation is not None and operation not in grounding_profile.supported_operations:
                raise ValueError("The grounding profile does not cover every allowed action.")
            required_shape = (
                GroundingOutputShape.REGION
                if operation is GroundingTargetOperation.SCROLL_REGION
                else GroundingOutputShape.POINT
            )
            if operation is not None and required_shape not in grounding_profile.supported_shapes:
                raise ValueError("The grounding profile does not cover every allowed shape.")


def load_yewo_run_configuration(path: Path) -> YewoRunConfiguration:
    """Load one strict Yewo configuration without exposing its content in errors."""
    try:
        payload = json.loads(path.read_text())
    except OSError as error:
        raise YewoRunConfigurationError("Could not read the Yewo run configuration.") from error
    except json.JSONDecodeError as error:
        raise YewoRunConfigurationError("The Yewo run configuration is not valid JSON.") from error
    try:
        return YewoRunConfiguration.model_validate(payload)
    except ValidationError as error:
        raise YewoRunConfigurationError("The Yewo run configuration is invalid.") from error
