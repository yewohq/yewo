"""Compile a resolved YAML workflow into the existing normal-run contract."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from yewo.agent.config import YewoRunConfiguration
from yewo.agent.actuation import (
    ACTION_DEFINITIONS,
    ActionHandling,
    ActionPolicyLimits,
    SideEffectClass,
)
from yewo.agent.runtime.limits import AgentModePolicy
from yewo.agent.runtime.records import AgentTask
from yewo.applications import ApplicationInstallMode, ApplicationRequest
from yewo.devices import DevicePlatform, DeviceQuery
from yewo.grounding import (
    GroundingCoordinateConvention,
    GroundingOutputShape,
    GroundingProfile,
    GroundingTargetOperation,
)
from yewo.models import (
    MINIMUM_COMPLETION_TOKENS,
    ModelGenerationSettings,
    ModelImageConstraints,
    ModelProfile,
    ModelProfileRegistry,
    ModelRetryPolicy,
    ModelRole,
    StructuredOutputMode,
    minimum_image_count_for_role,
)
from yewo.project_config.records import (
    GroundingProtocol,
    LoadedYewoProject,
    ModelOverridesSpec,
    ResolvedSettings,
)
from yewo.runs import RunMode, RunRequest


_PROFILE_IDS = {
    ModelRole.TASK_PLANNER: "yaml-task-planner",
    ModelRole.GROUNDER: "yaml-grounder",
    ModelRole.AUDITOR: "yaml-auditor",
    ModelRole.RECOVERY_ADVISOR: "yaml-recovery-advisor",
    ModelRole.ACTION_VERIFIER: "yaml-action-verifier",
    ModelRole.SCREEN_ANALYZER: "yaml-screen-analyzer",
    ModelRole.REPLAY_PREPARER: "yaml-replay-preparer",
}
_SEMANTIC_MODELS = (
    (ModelRole.TASK_PLANNER, "planner"),
    (ModelRole.AUDITOR, "auditor"),
    (ModelRole.RECOVERY_ADVISOR, "recovery"),
    (ModelRole.ACTION_VERIFIER, "action_verifier"),
    (ModelRole.SCREEN_ANALYZER, "screen_analyzer"),
    (ModelRole.REPLAY_PREPARER, "replay_preparer"),
)
_PROTOCOL_CONVENTIONS = {
    GroundingProtocol.SOURCE_PIXELS: GroundingCoordinateConvention.SOURCE_PIXELS,
    GroundingProtocol.NORMALIZED: GroundingCoordinateConvention.NORMALIZED,
    GroundingProtocol.NORMALIZED_1000: GroundingCoordinateConvention.NORMALIZED_1000,
}


class YewoProjectCompileError(ValueError):
    """A loaded YAML project is incomplete or unsafe to execute."""


class RunConfigurationOverrides(BaseModel):
    """Supported command-line overrides for one standalone run."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    platform: Literal["android", "ios"] | None = None
    device_name: str | None = Field(default=None, min_length=1)
    app_path: Path | None = None
    model: str | None = Field(default=None, min_length=1)
    max_steps: int | None = Field(default=None, ge=1, le=500)
    settle_time: float | None = Field(default=None, ge=0, le=5)
    output_dir: Path | None = None


def compile_yewo_run_configuration(
    project: LoadedYewoProject,
    objective: str,
    *,
    overrides: RunConfigurationOverrides | None = None,
) -> YewoRunConfiguration:
    """Compile top-level project settings for one plain-English run goal."""
    if not objective.strip():
        raise YewoProjectCompileError("The run objective must not be empty.")
    if len(objective) > 4_000:
        raise YewoProjectCompileError("The run objective cannot exceed 4,000 characters.")
    settings = _apply_overrides(project.settings, overrides)
    _validate_supported_settings(settings)
    platform = _required(settings.platform, "platform")
    device_name = _required(settings.device_name, "device")
    app_path = _required(settings.app_path, "app")
    grounder = settings.grounder
    if grounder is None:
        raise YewoProjectCompileError(
            "The project needs grounder.model and grounder.protocol. "
            "A semantic model cannot be assumed to support visual grounding."
        )

    profiles = _model_profiles(settings)
    maximum_steps = settings.max_steps or 100
    project_root = project.source_path.parent
    output_root = settings.output_dir or project_root / "assets"
    memory_root = _memory_root(settings, project_root)
    device_platform = DevicePlatform(platform)
    profile_ids = _PROFILE_IDS

    return YewoRunConfiguration(
        run=RunRequest(
            mode=RunMode.NORMAL,
            output_root=output_root,
            device=DeviceQuery(platform=device_platform, target_id=device_name),
            application=ApplicationRequest(
                platform=device_platform,
                install_mode=ApplicationInstallMode.INSTALL,
                artifact_path=app_path,
            ),
        ),
        task=AgentTask(objective=objective),
        mode_policy=_mode_policy(maximum_steps, settings.settle_time),
        model_profiles=ModelProfileRegistry(profiles=profiles),
        grounding_profile=GroundingProfile(
            profile_id="yaml-visual-grounder",
            model_profile_id=profile_ids[ModelRole.GROUNDER],
            coordinate_convention=_PROTOCOL_CONVENTIONS[grounder.protocol],
            supported_operations=frozenset(GroundingTargetOperation),
            supported_shapes=frozenset(GroundingOutputShape),
        ),
        planner_profile_id=profile_ids[ModelRole.TASK_PLANNER],
        auditor_profile_id=profile_ids[ModelRole.AUDITOR],
        recovery_profile_id=profile_ids[ModelRole.RECOVERY_ADVISOR],
        action_verifier_profile_id=profile_ids[ModelRole.ACTION_VERIFIER],
        replay_preparation_profile_id=profile_ids[ModelRole.REPLAY_PREPARER],
        screen_analyzer_profile_id=profile_ids[ModelRole.SCREEN_ANALYZER],
        memory_root=memory_root,
    )


def _apply_overrides(
    settings: ResolvedSettings,
    overrides: RunConfigurationOverrides | None,
) -> ResolvedSettings:
    if overrides is None:
        return settings
    changes = overrides.model_dump(exclude_none=True)
    if overrides.model is not None:
        changes["models"] = None
    return ResolvedSettings.model_validate({**settings.model_dump(mode="python"), **changes})


def _required[T](value: T | None, field: str) -> T:
    if value is None:
        raise YewoProjectCompileError(f"The project needs {field}.")
    return value


def _validate_supported_settings(settings: ResolvedSettings) -> None:
    if settings.headless:
        raise YewoProjectCompileError("headless: true is not wired to the rebuilt runtime yet.")
    if settings.record is not None:
        raise YewoProjectCompileError("record is not wired to the rebuilt runtime yet.")
    if settings.allow_secret_exposure:
        raise YewoProjectCompileError(
            "allow_secret_exposure: true is not supported by the rebuilt runtime."
        )


def _model_profiles(settings: ResolvedSettings) -> tuple[ModelProfile, ...]:
    semantic_models = _resolve_semantic_models(settings.model, settings.models)
    grounder = _required(settings.grounder, "grounder")
    profiles = [_model_profile(role, semantic_models[field]) for role, field in _SEMANTIC_MODELS]
    profiles.append(_model_profile(ModelRole.GROUNDER, grounder.model))
    return tuple(profiles)


def _resolve_semantic_models(
    default: str | None,
    overrides: ModelOverridesSpec | None,
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for _, field in _SEMANTIC_MODELS:
        override = getattr(overrides, field) if overrides is not None else None
        selected = override or default
        if selected is None:
            missing.append(field)
        else:
            resolved[field] = selected
    if missing:
        raise YewoProjectCompileError(
            "Configure model or explicit models for: " + ", ".join(missing) + "."
        )
    return resolved


def _model_profile(role: ModelRole, model: str) -> ModelProfile:
    provider = _provider(model)
    minimum_images = minimum_image_count_for_role(role)
    maximum_image_bytes = 8_000_000 if role is ModelRole.GROUNDER else 20_000_000
    return ModelProfile(
        profile_id=_PROFILE_IDS[role],
        role=role,
        provider=provider,
        model=model,
        structured_output_mode=StructuredOutputMode.PROMPT_SCHEMA,
        generation=ModelGenerationSettings(
            temperature=0,
            max_completion_tokens=MINIMUM_COMPLETION_TOKENS,
        ),
        retry=ModelRetryPolicy(
            timeout_seconds=60,
            max_attempts=4,
            backoff_seconds=(1, 2, 4),
            max_validation_attempts=4,
        ),
        images=ModelImageConstraints(
            max_images=minimum_images,
            max_image_bytes=maximum_image_bytes,
            max_width_px=4_096,
            max_height_px=4_096,
        ),
    )


def _provider(model: str) -> str:
    provider, separator, remainder = model.partition("/")
    if not separator or not provider or not remainder:
        raise YewoProjectCompileError(
            f"Model '{model}' must use an explicit provider/model identifier."
        )
    return provider


def _mode_policy(maximum_steps: int, settle_time: float | None) -> AgentModePolicy:
    executable_actions = frozenset(
        action
        for action, definition in ACTION_DEFINITIONS.items()
        if definition.handling is ActionHandling.EXECUTE
    )
    return AgentModePolicy(
        action_limits=ActionPolicyLimits(
            allowed_actions=executable_actions,
            supported_actions=executable_actions,
            allowed_side_effects=frozenset(
                {
                    SideEffectClass.NONE,
                    SideEffectClass.NAVIGATION,
                    SideEffectClass.APPLICATION_MUTATION,
                    SideEffectClass.UNCLASSIFIED_INTERACTION,
                }
            ),
            maximum_actions=maximum_steps,
            allow_exploratory_back=True,
        ),
        maximum_steps=maximum_steps,
        maximum_grounding_requests=min(1_000, maximum_steps * 2),
        maximum_model_requests=min(2_001, maximum_steps * 3),
        settlement_interval_seconds=settle_time if settle_time is not None else 0.35,
    )


def _memory_root(settings: ResolvedSettings, project_root: Path) -> Path | None:
    if settings.durable_replay is False:
        return None
    return settings.memory_root or project_root / ".yewo"
