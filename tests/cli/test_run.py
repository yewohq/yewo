"""Tests for the YAML-backed standalone run command."""

from argparse import Namespace
from pathlib import Path

from yewo.cli.run import configuration
from yewo.devices import DevicePlatform
from yewo.models import ModelRole


def test_loads_the_nearest_yaml_and_applies_supported_overrides(tmp_path: Path) -> None:
    (tmp_path / "yewo.yaml").write_text(
        """
device: Configured_Pixel
platform: android
app: configured.apk
model: openrouter/example/configured
grounder:
  model: openrouter/example/grounder
  protocol: normalized_1000
workflows:
  first:
    journey: Open the application.
  second:
    journey: Close the application.
"""
    )
    nested = tmp_path / "mobile" / "src"
    nested.mkdir(parents=True)
    arguments = Namespace(
        goal="Open settings.",
        platform="ios",
        device="iPhone 17",
        app_path=Path("build/Runner.app"),
        model="anthropic/claude-override",
        output_dir=Path("artifacts"),
        max_steps=20,
        settle_time=0.5,
    )

    compiled = configuration(arguments, nested)

    assert compiled.task.objective == "Open settings."
    assert compiled.run.device.platform is DevicePlatform.IOS
    assert compiled.run.device.target_id == "iPhone 17"
    assert compiled.run.application.artifact_path == nested / "build/Runner.app"
    assert compiled.run.output_root == nested / "artifacts"
    assert compiled.mode_policy.maximum_steps == 20
    assert {
        profile.model
        for profile in compiled.model_profiles.profiles
        if profile.role is not ModelRole.GROUNDER
    } == {"anthropic/claude-override"}


def test_uses_yaml_values_when_no_command_line_override_is_present(tmp_path: Path) -> None:
    (tmp_path / "yewo.yaml").write_text(
        """
device: Pixel_8
platform: android
app: app.apk
model: openrouter/example/semantic
grounder:
  model: openrouter/example/grounder
  protocol: source_pixels
"""
    )
    arguments = Namespace(
        goal="Open settings.",
        platform=None,
        device=None,
        app_path=None,
        model=None,
        output_dir=None,
        max_steps=None,
        settle_time=None,
    )

    compiled = configuration(arguments, tmp_path)

    assert compiled.run.device.target_id == "Pixel_8"
    assert compiled.run.application.artifact_path == tmp_path / "app.apk"
