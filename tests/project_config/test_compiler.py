"""Compilation tests for the public YAML run interface."""

from pathlib import Path

import pytest

from yewo.devices import DevicePlatform
from yewo.grounding import GroundingCoordinateConvention
from yewo.models import (
    ModelMessageRole,
    ModelReasoningEffort,
    ModelReasoningMode,
    ModelRole,
)
from yewo.project_config import (
    RunConfigurationOverrides,
    YewoProjectCompileError,
    compile_yewo_run_configuration,
    compile_yewo_test_configurations,
    load_yewo_project,
)


def _project(tmp_path: Path, body: str = ""):
    path = tmp_path / "yewo.yaml"
    path.write_text(
        """
device:
  name: Pixel_8
  platform: android
app: build/app.apk
model: openrouter/example/semantic
grounder:
  model: openrouter/example/grounder
  protocol: normalized_1000
max_steps: 12
journey: Open the app.
"""
        + body
    )
    return load_yewo_project(path)


def test_compiles_project_yaml_and_plain_english_goal(tmp_path: Path) -> None:
    project = _project(tmp_path)

    configuration = compile_yewo_run_configuration(
        project,
        "Find the account screen and explain what is available.",
    )

    assert configuration.task.objective == (
        "Find the account screen and explain what is available."
    )
    assert configuration.run.device.target_id == "Pixel_8"
    assert configuration.run.application.artifact_path == tmp_path / "build/app.apk"
    assert configuration.run.output_root == tmp_path / "assets/yewo"
    assert configuration.memory_root == tmp_path / ".yewo"
    assert configuration.mode_policy.maximum_steps == 12
    assert configuration.mode_policy.maximum_duration_seconds == 1_200
    assert configuration.mode_policy.maximum_grounding_requests == 24
    assert configuration.grounding_profile.coordinate_convention is (
        GroundingCoordinateConvention.NORMALIZED_1000
    )


def test_the_committed_example_project_still_compiles() -> None:
    """The shipped example must stay runnable, without this test owning its model choices."""
    path = Path(__file__).parents[2] / "eval" / "controlled_app" / "yewo.yaml"

    configuration = compile_yewo_run_configuration(
        load_yewo_project(path),
        "Open the restaurant list.",
    )

    assert configuration.run.device.target_id


def test_one_model_defaults_each_semantic_role_but_not_grounding(tmp_path: Path) -> None:
    configuration = compile_yewo_run_configuration(_project(tmp_path), "Open settings.")
    profiles = configuration.model_profiles.profiles

    semantic_profiles = tuple(
        profile for profile in profiles if profile.role is not ModelRole.GROUNDER
    )
    grounder = next(profile for profile in profiles if profile.role is ModelRole.GROUNDER)

    assert {profile.model for profile in semantic_profiles} == {"openrouter/example/semantic"}
    assert grounder.model == "openrouter/example/grounder"


def test_compiled_profiles_accept_the_supported_mobile_screenshot_envelope(
    tmp_path: Path,
) -> None:
    configuration = compile_yewo_run_configuration(_project(tmp_path), "Open settings.")
    profiles = configuration.model_profiles.profiles

    for profile in profiles:
        assert profile.images.max_width_px == 4_096
        assert profile.images.max_height_px == 4_096
        assert profile.images.max_image_bytes == (
            8_000_000 if profile.role is ModelRole.GROUNDER else 20_000_000
        )
        assert profile.generation.max_completion_tokens == 32_768
        assert profile.retry.timeout_seconds == 320
        assert profile.cache_control_role is ModelMessageRole.SYSTEM
        assert profile.router_cache_affinity


def test_only_the_grounder_keeps_its_source_resolution(tmp_path: Path) -> None:
    configuration = compile_yewo_run_configuration(_project(tmp_path), "Open settings.")

    for profile in configuration.model_profiles.profiles:
        if profile.role is ModelRole.GROUNDER:
            assert profile.images.max_long_edge_px is None
        else:
            assert profile.images.max_long_edge_px == 1_568


def test_role_overrides_replace_only_the_named_semantic_models(tmp_path: Path) -> None:
    path = tmp_path / "yewo.yaml"
    path.write_text(
        """
device: Pixel_8
platform: android
app: app.apk
model: openrouter/example/default
models:
  planner: anthropic/claude-planner
  auditor: openrouter/example/auditor
  test_designer: openrouter/example/test-designer
  screen_analyzer:
    model: openrouter/google/gemini-2.5-flash
    reasoning: disabled
grounder:
  model: openrouter/example/grounder
  protocol: source_pixels
journey: Open the app.
"""
    )

    configuration = compile_yewo_run_configuration(
        load_yewo_project(path),
        "Open settings.",
    )

    by_role = {profile.role: profile for profile in configuration.model_profiles.profiles}
    assert by_role[ModelRole.TASK_PLANNER].model == "anthropic/claude-planner"
    assert by_role[ModelRole.TASK_PLANNER].provider == "anthropic"
    # Anthropic permits four breakpoints; only router-side affinity is OpenRouter-only.
    assert by_role[ModelRole.TASK_PLANNER].cache_control_role is ModelMessageRole.SYSTEM
    assert not by_role[ModelRole.TASK_PLANNER].router_cache_affinity
    assert by_role[ModelRole.AUDITOR].model == "openrouter/example/auditor"
    assert by_role[ModelRole.AUDITOR].cache_control_role is ModelMessageRole.SYSTEM
    assert by_role[ModelRole.AUDITOR].router_cache_affinity
    assert by_role[ModelRole.RECOVERY_ADVISOR].model == "openrouter/example/default"
    assert by_role[ModelRole.TEST_DESIGNER].model == "openrouter/example/test-designer"
    assert by_role[ModelRole.SCREEN_ANALYZER].model == ("openrouter/google/gemini-2.5-flash")
    assert by_role[ModelRole.SCREEN_ANALYZER].generation.reasoning is ModelReasoningMode.DISABLED
    assert by_role[ModelRole.AUDITOR].generation.reasoning is ModelReasoningMode.PROVIDER_DEFAULT
    assert configuration.test_designer_profile_id == "yaml-test-designer"


def test_requires_explicit_grounder_instead_of_reusing_the_model(tmp_path: Path) -> None:
    path = tmp_path / "yewo.yaml"
    path.write_text(
        """
device: Pixel_8
platform: android
app: app.apk
model: openrouter/example/semantic
journey: Open the app.
"""
    )

    with pytest.raises(YewoProjectCompileError, match="cannot be assumed"):
        compile_yewo_run_configuration(load_yewo_project(path), "Open settings.")


@pytest.mark.parametrize("objective", ("", "   ", "x" * 4_001))
def test_rejects_an_invalid_plain_english_goal(tmp_path: Path, objective: str) -> None:
    with pytest.raises(YewoProjectCompileError, match="objective"):
        compile_yewo_run_configuration(_project(tmp_path), objective)


def test_standalone_run_uses_top_level_settings_not_workflow_targets(
    tmp_path: Path,
) -> None:
    path = tmp_path / "yewo.yaml"
    path.write_text(
        """
model: openrouter/example/semantic
device: Standalone_Pixel
platform: android
app: standalone.apk
grounder:
  model: openrouter/example/grounder
  protocol: source_pixels
workflows:
  matrix:
    targets:
      - id: android_phone
        platform: android
        device: Pixel_8
        app: android.apk
      - id: ios_phone
        platform: ios
        device: iPhone 17
        app: ios.app
    journey: Open the app.
"""
    )
    project = load_yewo_project(path)

    configuration = compile_yewo_run_configuration(project, "Open settings.")

    assert configuration.run.device.platform is DevicePlatform.ANDROID
    assert configuration.run.device.target_id == "Standalone_Pixel"
    assert configuration.run.application.artifact_path == tmp_path / "standalone.apk"
    assert configuration.grounding_profile.coordinate_convention is (
        GroundingCoordinateConvention.SOURCE_PIXELS
    )


def test_compiles_each_selected_journey_and_target_for_yewo_test(tmp_path: Path) -> None:
    path = tmp_path / "yewo.yaml"
    path.write_text(
        """
model: openrouter/example/semantic
grounder:
  model: openrouter/example/grounder
  protocol: source_pixels
workflows:
  checkout:
    targets:
      - id: pixel
        platform: android
        device: Pixel_8
        app: app.apk
      - id: iphone
        platform: ios
        device: iPhone 17
        app: Runner.app
    journeys:
      - id: browse
        journey: Browse the catalogue.
      - id: basket
        journey: Add an item to the basket.
"""
    )

    configurations = compile_yewo_test_configurations(
        load_yewo_project(path),
        workflow_selector="checkout",
        target_selector="iphone",
    )

    assert [(item.journey.journey_id, item.target_id) for item in configurations] == [
        ("browse", "iphone"),
        ("basket", "iphone"),
    ]
    assert all(item.run.mode.value == "scenario" for item in configurations)
    assert all(item.run.device.target_id == "iPhone 17" for item in configurations)
    assert all(item.discovery_mode_policy.maximum_steps == 100 for item in configurations)
    assert all(item.scenario_mode_policy.maximum_steps == 100 for item in configurations)
    assert all(
        item.discovery_mode_policy.maximum_duration_seconds == 1_200 for item in configurations
    )
    assert all(
        item.scenario_mode_policy.maximum_duration_seconds == 1_200 for item in configurations
    )
    assert all(
        item.discovery_mode_policy is not item.scenario_mode_policy for item in configurations
    )


def test_yewo_test_requires_durable_replay(tmp_path: Path) -> None:
    project = _project(tmp_path, "durable_replay: false\n")

    with pytest.raises(YewoProjectCompileError, match="durable replay"):
        compile_yewo_test_configurations(project)


def test_compiles_an_explicit_yaml_duration_budget(tmp_path: Path) -> None:
    project = _project(tmp_path, "max_duration_seconds: 900\n")

    configuration = compile_yewo_run_configuration(project, "Open settings.")

    assert configuration.mode_policy.maximum_duration_seconds == 900


def test_command_line_overrides_replace_supported_top_level_settings(
    tmp_path: Path,
) -> None:
    configuration = compile_yewo_run_configuration(
        _project(tmp_path),
        "Open settings.",
        overrides=RunConfigurationOverrides(
            platform="ios",
            device_name="iPhone 17",
            app_path=tmp_path / "Runner.app",
            model="anthropic/claude-override",
            max_steps=20,
            settle_time=0.5,
            output_dir=tmp_path / "run-output",
        ),
    )

    assert configuration.run.device.platform is DevicePlatform.IOS
    assert configuration.run.device.target_id == "iPhone 17"
    assert configuration.run.application.artifact_path == tmp_path / "Runner.app"
    assert configuration.run.output_root == tmp_path / "run-output"
    assert configuration.mode_policy.maximum_steps == 20
    assert configuration.mode_policy.settlement_interval_seconds == 0.5
    assert {
        profile.model
        for profile in configuration.model_profiles.profiles
        if profile.role is not ModelRole.GROUNDER
    } == {"anthropic/claude-override"}


@pytest.mark.parametrize(
    ("setting", "message"),
    (
        ("headless: true", "headless"),
        ("record: true", "record"),
        ("record: false", "record"),
    ),
)
def test_rejects_yaml_settings_the_runtime_cannot_honour(
    tmp_path: Path,
    setting: str,
    message: str,
) -> None:
    project = _project(tmp_path, setting + "\n")

    with pytest.raises(YewoProjectCompileError, match=message):
        compile_yewo_run_configuration(project, "Open settings.")


def test_a_project_can_bound_how_long_a_role_reasons(tmp_path: Path) -> None:
    # Reasoning and the answer share one completion budget, so a project needs a way to keep a
    # deliberating role from spending the whole budget before its answer starts.
    path = tmp_path / "yewo.yaml"
    path.write_text(
        """
device:
  name: Pixel
  platform: android
app: app.apk
model: openrouter/example/default
models:
  planner:
    model: openrouter/example/planner
    reasoning: minimal
  screen_analyzer:
    model: openrouter/example/analyzer
    reasoning: disabled
grounder:
  model: openrouter/example/grounder
  protocol: source_pixels
  reasoning: disabled
journey: Open the app.
"""
    )

    configuration = compile_yewo_run_configuration(load_yewo_project(path), "Open settings.")

    by_role = {profile.role: profile for profile in configuration.model_profiles.profiles}
    planner = by_role[ModelRole.TASK_PLANNER].generation
    assert planner.reasoning is ModelReasoningMode.PROVIDER_DEFAULT
    assert planner.reasoning_effort is ModelReasoningEffort.MINIMAL
    for role in (ModelRole.SCREEN_ANALYZER, ModelRole.GROUNDER):
        assert by_role[role].generation.reasoning is ModelReasoningMode.DISABLED
        assert by_role[role].generation.reasoning_effort is None
    unset = by_role[ModelRole.AUDITOR].generation
    assert unset.reasoning is ModelReasoningMode.PROVIDER_DEFAULT
    assert unset.reasoning_effort is None
