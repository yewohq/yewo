"""Strict Yewo-run configuration tests."""

import json
from pathlib import Path

import pytest

from yewo.agent import YewoRunConfigurationError, load_yewo_run_configuration


def _configuration() -> dict[str, object]:
    profile = {
        "provider": "openrouter",
        "revision": "configured-revision",
        "structured_output_mode": "prompt_schema",
        "generation": {"temperature": 0, "max_completion_tokens": 8192},
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
                {
                    **profile,
                    "profile_id": "auditor-primary",
                    "role": "auditor",
                    "model": "openrouter/configured/auditor",
                },
                {
                    **profile,
                    "profile_id": "recovery-primary",
                    "role": "recovery_advisor",
                    "model": "openrouter/configured/recovery",
                    "images": {**profile["images"], "max_images": 2},
                },
                {
                    **profile,
                    "profile_id": "action-verifier-primary",
                    "role": "action_verifier",
                    "model": "openrouter/configured/action-verifier",
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
        "auditor_profile_id": "auditor-primary",
        "recovery_profile_id": "recovery-primary",
        "action_verifier_profile_id": "action-verifier-primary",
    }


def _write_configuration(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "normal-run.json"
    path.write_text(json.dumps(payload))
    return path


def test_loads_explicit_profiles_without_a_default_model(tmp_path: Path) -> None:
    configuration = load_yewo_run_configuration(_write_configuration(tmp_path, _configuration()))

    assert configuration.planner_profile_id == "planner-primary"
    assert configuration.recovery_profile_id == "recovery-primary"
    assert configuration.action_verifier_profile_id == "action-verifier-primary"
    assert [profile.model for profile in configuration.model_profiles.profiles] == [
        "openrouter/configured/planner",
        "openrouter/configured/grounder",
        "openrouter/configured/auditor",
        "openrouter/configured/recovery",
        "openrouter/configured/action-verifier",
    ]


def test_loads_an_optional_reusable_memory_root(tmp_path: Path) -> None:
    payload = _configuration()
    payload["memory_root"] = "shared-app-memory"

    configuration = load_yewo_run_configuration(_write_configuration(tmp_path, payload))

    assert configuration.memory_root == Path("shared-app-memory")


def test_loads_the_recommended_discovery_budgets(tmp_path: Path) -> None:
    payload = _configuration()
    mode_policy = payload["mode_policy"]
    assert isinstance(mode_policy, dict)
    action_limits = mode_policy["action_limits"]
    assert isinstance(action_limits, dict)
    action_limits["maximum_actions"] = 90
    mode_policy.update(
        {
            "maximum_steps": 100,
            "maximum_grounding_requests": 100,
            "maximum_model_requests": 201,
            "maximum_duration_seconds": 3_600,
        }
    )

    configuration = load_yewo_run_configuration(_write_configuration(tmp_path, payload))

    assert configuration.mode_policy.maximum_steps == 100
    assert configuration.mode_policy.action_limits.maximum_actions == 90
    assert configuration.mode_policy.maximum_grounding_requests == 100
    assert configuration.mode_policy.maximum_model_requests == 201


@pytest.mark.parametrize(
    ("field", "value"),
    (("maximum_steps", 101), ("maximum_model_requests", 202)),
)
def test_rejects_agent_budgets_above_the_supported_ceiling(
    tmp_path: Path,
    field: str,
    value: int,
) -> None:
    payload = _configuration()
    mode_policy = payload["mode_policy"]
    assert isinstance(mode_policy, dict)
    mode_policy[field] = value

    with pytest.raises(YewoRunConfigurationError, match="invalid"):
        load_yewo_run_configuration(_write_configuration(tmp_path, payload))


def test_rejects_declared_values_until_the_cli_has_a_secret_boundary(tmp_path: Path) -> None:
    payload = _configuration()
    task = payload["task"]
    assert isinstance(task, dict)
    task["declared_value_names"] = ["password"]

    with pytest.raises(YewoRunConfigurationError, match="invalid"):
        load_yewo_run_configuration(_write_configuration(tmp_path, payload))


def test_rejects_a_profile_with_the_wrong_role(tmp_path: Path) -> None:
    payload = _configuration()
    profiles = payload["model_profiles"]
    assert isinstance(profiles, dict)
    profile_values = profiles["profiles"]
    assert isinstance(profile_values, list)
    planner_profile = profile_values[0]
    assert isinstance(planner_profile, dict)
    planner_profile["role"] = "grounder"

    with pytest.raises(YewoRunConfigurationError, match="invalid"):
        load_yewo_run_configuration(_write_configuration(tmp_path, payload))


def test_rejects_a_recovery_profile_with_the_wrong_role(tmp_path: Path) -> None:
    payload = _configuration()
    profiles = payload["model_profiles"]
    assert isinstance(profiles, dict)
    profile_values = profiles["profiles"]
    assert isinstance(profile_values, list)
    recovery_profile = profile_values[3]
    assert isinstance(recovery_profile, dict)
    recovery_profile["role"] = "task_planner"

    with pytest.raises(YewoRunConfigurationError, match="invalid"):
        load_yewo_run_configuration(_write_configuration(tmp_path, payload))


def test_rejects_a_recovery_profile_that_cannot_receive_both_screenshots(
    tmp_path: Path,
) -> None:
    payload = _configuration()
    profiles = payload["model_profiles"]
    assert isinstance(profiles, dict)
    profile_values = profiles["profiles"]
    assert isinstance(profile_values, list)
    recovery_profile = profile_values[3]
    assert isinstance(recovery_profile, dict)
    images = recovery_profile["images"]
    assert isinstance(images, dict)
    images["max_images"] = 1

    with pytest.raises(YewoRunConfigurationError, match="invalid"):
        load_yewo_run_configuration(_write_configuration(tmp_path, payload))
