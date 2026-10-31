"""Validated configuration for one normal Device Agent run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from yewo.agent.actuation import ACTION_DEFINITIONS
from yewo.agent.runtime_records import AgentModePolicy, AgentTask
from yewo.artifacts import RecordIdentifier
from yewo.grounding import (
    GroundingOutputShape,
    GroundingProfile,
    GroundingTargetOperation,
)
from yewo.models import ModelProfileRegistry, ModelRole
from yewo.observations import RedactionRegion
from yewo.runs import RunMode, RunRequest


class NormalRunConfigurationError(ValueError):
    """A normal-run configuration could not be loaded safely."""


class NormalRunConfiguration(BaseModel):
    """All explicit configuration required for one normal device task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run: RunRequest
    task: AgentTask
    mode_policy: AgentModePolicy
    model_profiles: ModelProfileRegistry
    grounding_profile: GroundingProfile
    planner_profile_id: RecordIdentifier
    redaction_regions: tuple[RedactionRegion, ...] = ()

    @model_validator(mode="after")
    def _validate_composition(self) -> Self:
        if self.run.mode is not RunMode.NORMAL:
            raise ValueError("A normal-run configuration requires normal mode.")
        if self.task.declared_value_names:
            raise ValueError("The normal CLI does not yet accept declared sensitive values.")
        try:
            self.model_profiles.resolve(self.planner_profile_id, ModelRole.TASK_PLANNER)
            self.model_profiles.resolve(
                self.grounding_profile.model_profile_id,
                ModelRole.GROUNDER,
            )
        except LookupError as error:
            raise ValueError("The normal run references an incompatible model profile.") from error
        self._validate_grounding_coverage()
        return self

    def _validate_grounding_coverage(self) -> None:
        available_actions = (
            self.mode_policy.action_limits.allowed_actions
            & self.mode_policy.action_limits.supported_actions
        )
        for action in available_actions:
            definition = ACTION_DEFINITIONS[action]
            for operation in (
                definition.target_operation,
                definition.destination_operation,
            ):
                if (
                    operation is not None
                    and operation not in self.grounding_profile.supported_operations
                ):
                    raise ValueError("The grounding profile does not cover every allowed action.")
                required_shape = (
                    GroundingOutputShape.REGION
                    if operation is GroundingTargetOperation.SCROLL_REGION
                    else GroundingOutputShape.POINT
                )
                if (
                    operation is not None
                    and required_shape not in self.grounding_profile.supported_shapes
                ):
                    raise ValueError("The grounding profile does not cover every allowed shape.")


def load_normal_run_configuration(path: Path) -> NormalRunConfiguration:
    """Load one strict JSON configuration without exposing its content in errors."""
    try:
        payload = json.loads(path.read_text())
    except OSError as error:
        raise NormalRunConfigurationError("Could not read the normal-run configuration.") from error
    except json.JSONDecodeError as error:
        raise NormalRunConfigurationError(
            "The normal-run configuration is not valid JSON."
        ) from error
    try:
        return NormalRunConfiguration.model_validate(payload)
    except ValidationError as error:
        raise NormalRunConfigurationError("The normal-run configuration is invalid.") from error
