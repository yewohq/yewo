"""Contracts for the per-role model selection summary."""

from yewo.models import ModelProfileRegistry, ModelRole, build_model_selections

from models.support import build_model_profile


def test_each_role_reports_its_model() -> None:
    registry = ModelProfileRegistry(
        profiles=(
            build_model_profile(
                profile_id="grounder-primary",
                role=ModelRole.GROUNDER,
                model="openrouter/example/grounder",
            ),
            build_model_profile(
                profile_id="replay-preparer-primary",
                role=ModelRole.REPLAY_PREPARER,
                provider="anthropic",
                model="anthropic/claude-sonnet-5",
            ),
        )
    )

    assert build_model_selections(registry) == {
        "grounder": "openrouter/example/grounder",
        "replay_preparer": "anthropic/claude-sonnet-5",
    }


def test_a_role_with_several_profiles_lists_every_model_once() -> None:
    registry = ModelProfileRegistry(
        profiles=(
            build_model_profile(profile_id="grounder-b", model="openrouter/example/second"),
            build_model_profile(profile_id="grounder-a", model="openrouter/example/first"),
            build_model_profile(profile_id="grounder-c", model="openrouter/example/first"),
        )
    )

    assert build_model_selections(registry) == {
        "grounder": "openrouter/example/first, openrouter/example/second"
    }


def test_an_empty_registry_reports_no_selections() -> None:
    assert build_model_selections(ModelProfileRegistry(profiles=())) == {}
