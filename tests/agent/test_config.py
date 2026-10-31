"""Strict normal-run configuration tests."""

import json
from pathlib import Path

import pytest

from yewo.agent import NormalRunConfigurationError, load_normal_run_configuration


def _configuration() -> dict[str, object]:
    profile = {
        "provider": "openrouter",
        "revision": "configured-revision",
        "structured_output_mode": "prompt_schema",
        "generation": {"temperature": 0, "max_completion_tokens": 128},
        "retry": {
            "timeout_seconds": 20,
            "max_attempts": 1,
            "backoff_seconds": [],
            "max_validation_attempts": 2,
        },
        "images": {
            "max_images": 1,
            "max_image_bytes": 4000000,
            "max_width_px": 2048,
            "max_height_px": 2048,
        },
    }
    return {
        "run": {
            "mode": "normal",
            "output_root": "run-output",
            "device": {"platform": "android", "target_id": "test-device"},
            "application": {
                "platform": "android",
                "install_mode": "reuse",
                "application_id": "com.example.app",
            },
        },
        "task": {
            "objective": "Wait for the application.",
            "completion_conditions": ["The application is ready."],
        },
        "mode_policy": {
            "action_limits": {
                "allowed_actions": ["wait"],
                "supported_actions": ["wait"],
                "allowed_side_effects": ["none"],
                "maximum_actions": 1,
            },
            "maximum_steps": 1,
            "maximum_grounding_requests": 0,
        },
        "model_profiles": {
            "profiles": [
                {
                    **profile,
                    "profile_id": "planner-primary",
                    "role": "task_planner",
                    "model": "openrouter/configured/planner",
                },
                {
                    **profile,
                    "profile_id": "grounder-primary",
                    "role": "grounder",
                    "model": "openrouter/configured/grounder",
                },
            ]
        },
        "grounding_profile": {
            "profile_id": "visual-grounder-primary",
            "model_profile_id": "grounder-primary",
            "coordinate_convention": "source_pixels",
            "supported_operations": ["tap"],
            "supported_shapes": ["point"],
        },
        "planner_profile_id": "planner-primary",
    }


def _write_configuration(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "normal-run.json"
    path.write_text(json.dumps(payload))
    return path


def test_loads_explicit_profiles_without_a_default_model(tmp_path: Path) -> None:
    configuration = load_normal_run_configuration(_write_configuration(tmp_path, _configuration()))

    assert configuration.planner_profile_id == "planner-primary"
    assert [profile.model for profile in configuration.model_profiles.profiles] == [
        "openrouter/configured/planner",
        "openrouter/configured/grounder",
    ]


def test_rejects_declared_values_until_the_cli_has_a_secret_boundary(tmp_path: Path) -> None:
    payload = _configuration()
    task = payload["task"]
    assert isinstance(task, dict)
    task["declared_value_names"] = ["password"]

    with pytest.raises(NormalRunConfigurationError, match="invalid"):
        load_normal_run_configuration(_write_configuration(tmp_path, payload))


def test_rejects_a_profile_with_the_wrong_role(tmp_path: Path) -> None:
    payload = _configuration()
    profiles = payload["model_profiles"]
    assert isinstance(profiles, dict)
    profile_values = profiles["profiles"]
    assert isinstance(profile_values, list)
    planner_profile = profile_values[0]
    assert isinstance(planner_profile, dict)
    planner_profile["role"] = "grounder"

    with pytest.raises(NormalRunConfigurationError, match="invalid"):
        load_normal_run_configuration(_write_configuration(tmp_path, payload))
