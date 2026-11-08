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
from yewo.agent.runtime.limits import (
    DEFAULT_MAXIMUM_DURATION_SECONDS,
    AgentModePolicy,
)
from yewo.agent.runtime.records import AgentTask
from yewo.artifacts import DEFAULT_OUTPUT_DIRECTORY
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
    ModelMessageRole,
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
    ModelSelectionSpec,
    ResolvedSettings,
    ResolvedTarget,
    ResolvedWorkflow,
)
from yewo.project_config.loader import select_workflows
from yewo.runs import RunMode, RunRequest
from yewo.test_config import YewoTestConfiguration


_SCREEN_ANALYZER_COMPLETION_TOKENS = 16_384


_PROFILE_IDS = {
    ModelRole.TASK_PLANNER: "yaml-task-planner",
    ModelRole.GROUNDER: "yaml-grounder",
    ModelRole.AUDITOR: "yaml-auditor",
    ModelRole.RECOVERY_ADVISOR: "yaml-recovery-advisor",
    ModelRole.ACTION_VERIFIER: "yaml-action-verifier",
    ModelRole.SCREEN_ANALYZER: "yaml-screen-analyzer",
    ModelRole.REPLAY_PREPARER: "yaml-replay-preparer",
    ModelRole.TEST_DESIGNER: "yaml-test-designer",
}
_SEMANTIC_MODELS = (
    (ModelRole.TASK_PLANNER, "planner"),
    (ModelRole.AUDITOR, "auditor"),
    (ModelRole.RECOVERY_ADVISOR, "recovery"),
    (ModelRole.ACTION_VERIFIER, "action_verifier"),
    (ModelRole.SCREEN_ANALYZER, "screen_analyzer"),
    (ModelRole.REPLAY_PREPARER, "replay_preparer"),
    (ModelRole.TEST_DESIGNER, "test_designer"),
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
    output_root = settings.output_dir or project_root / DEFAULT_OUTPUT_DIRECTORY
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
        mode_policy=_mode_policy(
            maximum_steps,
            settings.settle_time,
            settings.max_duration_seconds,
        ),
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
        test_designer_profile_id=profile_ids[ModelRole.TEST_DESIGNER],
        memory_root=memory_root,
    )


def compile_yewo_test_configurations(
    project: LoadedYewoProject,
    *,
    workflow_selector: str | None = None,
    target_selector: str | None = None,
) -> tuple[YewoTestConfiguration, ...]:
    """Compile every selected journey-target pair into an isolated test run."""
    workflows = select_workflows(project, workflow_selector)
    workflow_target_pairs = tuple(
        (workflow, target) for workflow in workflows for target in _selected_targets(workflow)
    )
    workflow_target_pairs = _filter_targets(workflow_target_pairs, target_selector)
    configurations: list[YewoTestConfiguration] = []
    for workflow, target in workflow_target_pairs:
        target_project = project.model_copy(update={"settings": target.settings})
        for journey in workflow.journeys:
            base = compile_yewo_run_configuration(target_project, journey.objective)
            if base.memory_root is None:
                raise YewoProjectCompileError(
                    "yewo test requires durable replay. Remove durable_replay: false."
                )
            configurations.append(
                YewoTestConfiguration(
                    run=base.run.model_copy(update={"mode": RunMode.SCENARIO}),
                    workflow_name=workflow.name,
                    target_id=target.target_id,
                    journey=journey,
                    discovery_mode_policy=base.mode_policy,
                    scenario_mode_policy=base.mode_policy.model_copy(),
                    model_profiles=base.model_profiles,
                    grounding_profile=base.grounding_profile,
                    planner_profile_id=base.planner_profile_id,
                    auditor_profile_id=base.auditor_profile_id,
                    recovery_profile_id=base.recovery_profile_id,
                    action_verifier_profile_id=base.action_verifier_profile_id,
                    replay_preparation_profile_id=base.replay_preparation_profile_id,
                    screen_analyzer_profile_id=base.screen_analyzer_profile_id,
                    test_designer_profile_id=base.test_designer_profile_id,
                    memory_root=base.memory_root,
                    redaction_regions=base.redaction_regions,
                )
            )
    return tuple(configurations)


def _selected_targets(workflow: ResolvedWorkflow) -> tuple[ResolvedTarget, ...]:
    if workflow.targets:
        return workflow.targets
    return (ResolvedTarget(target_id="default", settings=workflow.settings),)


def _filter_targets(
    workflow_target_pairs: tuple[tuple[ResolvedWorkflow, ResolvedTarget], ...],
    selector: str | None,
) -> tuple[tuple[ResolvedWorkflow, ResolvedTarget], ...]:
    if selector is None:
        return workflow_target_pairs
    names = tuple(name.strip() for name in selector.split(",") if name.strip())
    if not names:
        raise YewoProjectCompileError("The target selector is empty.")
    available_target_ids = {target.target_id for _, target in workflow_target_pairs}
    unknown_target_names = tuple(name for name in names if name not in available_target_ids)
    if unknown_target_names:
        raise YewoProjectCompileError(
            f"Unknown target(s): {', '.join(unknown_target_names)}. Available targets: "
            f"{', '.join(sorted(available_target_ids)) or 'default'}."
        )
    return tuple(item for name in names for item in workflow_target_pairs if item[1].target_id == name)


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
    profiles.append(
        _model_profile(
            ModelRole.GROUNDER,
            ModelSelectionSpec(model=grounder.model),
        )
    )
    return tuple(profiles)


def _resolve_semantic_models(
    default: str | None,
    overrides: ModelOverridesSpec | None,
) -> dict[str, ModelSelectionSpec]:
    models_by_field: dict[str, ModelSelectionSpec] = {}
    unassigned_fields: list[str] = []
    for _, field in _SEMANTIC_MODELS:
        override = getattr(overrides, field) if overrides is not None else None
        model_name = override or default
        if model_name is None:
            unassigned_fields.append(field)
        else:
            models_by_field[field] = _model_selection(model_name)
    if unassigned_fields:
        raise YewoProjectCompileError(
            "Configure model or explicit models for: " + ", ".join(unassigned_fields) + "."
        )
    return models_by_field


def _model_selection(selection: str | ModelSelectionSpec) -> ModelSelectionSpec:
    if isinstance(selection, ModelSelectionSpec):
        return selection
    return ModelSelectionSpec(model=selection)


def _model_profile(role: ModelRole, selection: ModelSelectionSpec) -> ModelProfile:
    provider = _provider(selection.model)
    uses_openrouter = provider == "openrouter"
    minimum_images = minimum_image_count_for_role(role)
    maximum_image_bytes = 8_000_000 if role is ModelRole.GROUNDER else 20_000_000
    return ModelProfile(
        profile_id=_PROFILE_IDS[role],
        role=role,
        provider=provider,
        model=selection.model,
        structured_output_mode=StructuredOutputMode.PROMPT_SCHEMA,
        generation=ModelGenerationSettings(
            temperature=0,
            max_completion_tokens=(
                _SCREEN_ANALYZER_COMPLETION_TOKENS
                if role is ModelRole.SCREEN_ANALYZER
                else MINIMUM_COMPLETION_TOKENS
            ),
            reasoning=selection.reasoning,
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
        cache_control_role=(ModelMessageRole.SYSTEM if uses_openrouter else None),
        router_cache_affinity=uses_openrouter,
    )


def _provider(model: str) -> str:
    provider, separator, remainder = model.partition("/")
    if not separator or not provider or not remainder:
        raise YewoProjectCompileError(
            f"Model '{model}' must use an explicit provider/model identifier."
        )
    return provider


def _mode_policy(
    maximum_steps: int,
    settle_time: float | None,
    maximum_duration_seconds: float | None,
) -> AgentModePolicy:
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
        maximum_duration_seconds=(
            maximum_duration_seconds
            if maximum_duration_seconds is not None
            else DEFAULT_MAXIMUM_DURATION_SECONDS
        ),
        settlement_interval_seconds=settle_time if settle_time is not None else 0.35,
    )


def _memory_root(settings: ResolvedSettings, project_root: Path) -> Path | None:
    if settings.durable_replay is False:
        return None
    return settings.memory_root or project_root / ".yewo"
