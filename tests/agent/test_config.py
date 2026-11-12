"""Strict Yewo-run configuration tests."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from yewo.agent import YewoRunConfiguration


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
                    "images": {**profile["images"], "max_images": 2},
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
                    "profile_id": "proposal-verifier-primary",
                    "role": "proposal_verifier",
                    "model": "openrouter/configured/proposal-verifier",
                },
                {
                    **profile,
                    "profile_id": "outcome-verifier-primary",
                    "role": "outcome_verifier",
                    "model": "openrouter/configured/outcome-verifier",
                    "images": {**profile["images"], "max_images": 2},
                },
                {
                    **profile,
                    "profile_id": "screen-analyzer-primary",
                    "role": "screen_analyzer",
                    "model": "openrouter/configured/screen-analyzer",
                },
                {
                    **profile,
                    "profile_id": "reconciler-primary",
                    "role": "replay_preparer",
                    "model": "openrouter/configured/reconciler",
                    "images": {**profile["images"], "max_images": 4},
                },
                {
                    **profile,
                    "profile_id": "test-designer-primary",
                    "role": "test_designer",
                    "model": "openrouter/configured/test-designer",
                    "images": {**profile["images"], "max_images": 2},
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
        "proposal_verifier_profile_id": "proposal-verifier-primary",
        "outcome_verifier_profile_id": "outcome-verifier-primary",
        "screen_analyzer_profile_id": "screen-analyzer-primary",
        "replay_preparation_profile_id": "reconciler-primary",
        "test_designer_profile_id": "test-designer-primary",
    }


def test_validates_explicit_profiles_without_a_default_model() -> None:
    configuration = YewoRunConfiguration.model_validate(_configuration())

    assert configuration.planner_profile_id == "planner-primary"
    assert configuration.recovery_profile_id == "recovery-primary"
    assert configuration.proposal_verifier_profile_id == "proposal-verifier-primary"
    assert configuration.screen_analyzer_profile_id == "screen-analyzer-primary"
    assert configuration.replay_preparation_profile_id == "reconciler-primary"
    assert configuration.test_designer_profile_id == "test-designer-primary"
    assert [profile.model for profile in configuration.model_profiles.profiles] == [
        "openrouter/configured/planner",
        "openrouter/configured/grounder",
        "openrouter/configured/auditor",
        "openrouter/configured/recovery",
        "openrouter/configured/proposal-verifier",
        "openrouter/configured/outcome-verifier",
        "openrouter/configured/screen-analyzer",
        "openrouter/configured/reconciler",
        "openrouter/configured/test-designer",
    ]


def test_objective_only_task_preserves_absent_discovery_conditions() -> None:
    payload = _configuration()
    payload["task"] = {"objective": "Create and reopen one record."}

    configuration = YewoRunConfiguration.model_validate(payload)

    assert configuration.task.known_facts == ()
    assert configuration.task.open_work == ()
    assert configuration.task.completion_conditions == ()


def test_normal_configuration_requires_the_screen_analyzer() -> None:
    payload = _configuration()
    del payload["screen_analyzer_profile_id"]

    with pytest.raises(ValidationError, match="screen_analyzer_profile_id"):
        YewoRunConfiguration.model_validate(payload)


def test_loads_an_optional_reusable_memory_root() -> None:
    payload = _configuration()
    payload["memory_root"] = "shared-app-memory"

    configuration = YewoRunConfiguration.model_validate(payload)

    assert configuration.memory_root == Path("shared-app-memory")


def test_configuration_accepts_an_optional_memory_root() -> None:
    configured = _configuration()
    configured["memory_root"] = "shared-app-memory"

    configuration = YewoRunConfiguration.model_validate(configured)

    assert configuration.memory_root == Path("shared-app-memory")


def test_loads_the_recommended_discovery_budgets() -> None:
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

    configuration = YewoRunConfiguration.model_validate(payload)

    assert configuration.mode_policy.maximum_steps == 100
    assert configuration.mode_policy.action_limits.maximum_actions == 90
    assert configuration.mode_policy.maximum_grounding_requests == 100
    assert configuration.mode_policy.maximum_model_requests == 201


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("maximum_steps", 501),
        ("maximum_model_requests", 2_002),
        ("maximum_duration_seconds", 14_401),
    ),
)
def test_rejects_agent_budgets_above_the_supported_ceiling(
    field: str,
    value: int,
) -> None:
    payload = _configuration()
    mode_policy = payload["mode_policy"]
    assert isinstance(mode_policy, dict)
    mode_policy[field] = value

    with pytest.raises(ValidationError):
        YewoRunConfiguration.model_validate(payload)


def test_rejects_declared_values_until_the_cli_has_a_secret_boundary() -> None:
    payload = _configuration()
    task = payload["task"]
    assert isinstance(task, dict)
    task["declared_value_names"] = ["password"]

    with pytest.raises(ValidationError):
        YewoRunConfiguration.model_validate(payload)


def test_rejects_a_profile_with_the_wrong_role() -> None:
    payload = _configuration()
    profiles = payload["model_profiles"]
    assert isinstance(profiles, dict)
    profile_values = profiles["profiles"]
    assert isinstance(profile_values, list)
    planner_profile = profile_values[0]
    assert isinstance(planner_profile, dict)
    planner_profile["role"] = "grounder"

    with pytest.raises(ValidationError):
        YewoRunConfiguration.model_validate(payload)


def test_rejects_a_recovery_profile_with_the_wrong_role() -> None:
    payload = _configuration()
    profiles = payload["model_profiles"]
    assert isinstance(profiles, dict)
    profile_values = profiles["profiles"]
    assert isinstance(profile_values, list)
    recovery_profile = profile_values[3]
    assert isinstance(recovery_profile, dict)
    recovery_profile["role"] = "task_planner"

    with pytest.raises(ValidationError):
        YewoRunConfiguration.model_validate(payload)


def test_rejects_a_recovery_profile_that_cannot_receive_both_screenshots() -> None:
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

    with pytest.raises(ValidationError):
        YewoRunConfiguration.model_validate(payload)


def test_rejects_a_grounding_fallback_whose_profile_is_not_a_grounder() -> None:
    payload = _configuration()
    grounding_profile = payload["grounding_profile"]
    assert isinstance(grounding_profile, dict)
    grounding_profile["fallbacks"] = [
        {
            "model_profile_id": "planner-primary",
            "coordinate_convention": "normalized_1000",
        }
    ]

    with pytest.raises(ValidationError):
        YewoRunConfiguration.model_validate(payload)
