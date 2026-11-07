"""YAML project loading, inheritance, and sensitive-value tests."""

from pathlib import Path

import pytest

from yewo.project_config import (
    CoverageCategory,
    ScenarioType,
    YewoProjectError,
    find_yewo_project,
    load_nearest_yewo_project,
    load_yewo_project,
    select_workflows,
)
from yewo.project_config.capabilities import (
    AppStateCapabilitySpec,
    MediaInjectCapabilitySpec,
)


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_loads_the_controlled_application_project() -> None:
    path = Path(__file__).parents[2] / "eval" / "controlled_app" / "yewo.yaml"

    project = load_yewo_project(path)

    assert [workflow.name for workflow in project.workflows] == [
        "permissions_flow",
        "restaurant_menu_matrix_smoke",
        "notifications",
        "delivery_checkout",
        "food_delivery_suite",
    ]
    permissions = project.workflows[0]
    assert permissions.settings.platform == "android"
    assert permissions.settings.device_name == "Pixel_8"
    assert (
        permissions.settings.app_path
        == (path.parent / "build/app/outputs/flutter-apk/app-debug.apk").resolve()
    )
    journey = permissions.journeys[0]
    assert journey.declared_values["password"].secret is True
    assert journey.declared_values["password"].value == "secret-password123"
    assert "{{password}}" in journey.objective
    assert journey.coverage[0].category is CoverageCategory.FUNCTIONAL
    assert journey.coverage[0].scenario_type is ScenarioType.HAPPY_PATH


def test_inherits_settings_and_resolves_target_paths(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "project" / "yewo.yaml",
        """
device:
  name: Pixel_8
  platform: android
app: builds/default.apk
output_dir: results
workflows:
  matrix:
    targets:
      - id: iphone
        platform: ios
        device: iPhone 17
        app: builds/Runner.app
    journeys:
      - id: open_app
        journey: Open the application.
""",
    )

    project = load_yewo_project(path)
    workflow = project.workflows[0]

    assert project.settings.platform == "android"
    assert project.settings.device_name == "Pixel_8"
    assert project.settings.app_path == (path.parent / "builds/default.apk").resolve()
    assert workflow.settings.platform == "android"
    assert workflow.settings.device_name == "Pixel_8"
    assert workflow.settings.output_dir == (path.parent / "results").resolve()
    assert workflow.targets[0].settings.platform == "ios"
    assert workflow.targets[0].settings.device_name == "iPhone 17"
    assert workflow.targets[0].settings.app_path == (path.parent / "builds/Runner.app").resolve()


def test_accepts_a_settings_only_project_for_standalone_run(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "yewo.yaml",
        """
device: Pixel_8
platform: android
app: app.apk
model: openrouter/example/semantic
grounder:
  model: openrouter/example/grounder
  protocol: source_pixels
""",
    )

    project = load_yewo_project(path)

    assert project.settings.device_name == "Pixel_8"
    assert project.settings.app_path == tmp_path / "app.apk"
    assert project.workflows == ()


def test_loads_role_specific_model_reasoning(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "yewo.yaml",
        """
device: Pixel_8
platform: android
app: app.apk
model: openrouter/example/default
models:
  screen_analyzer:
    model: openrouter/google/gemini-2.5-flash
    reasoning: disabled
grounder:
  model: openrouter/example/grounder
  protocol: source_pixels
""",
    )

    project = load_yewo_project(path)

    assert project.settings.models is not None
    screen_analyzer = project.settings.models.screen_analyzer
    assert screen_analyzer is not None
    assert not isinstance(screen_analyzer, str)
    assert screen_analyzer.model == "openrouter/google/gemini-2.5-flash"
    assert screen_analyzer.reasoning.value == "disabled"


def test_resolves_environment_values_without_substituting_journey_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("YEWO_TEST_PASSWORD", "not-for-prose")
    path = _write(
        tmp_path / "yewo.yaml",
        """
env:
  email: tester@example.com
  password: ${YEWO_TEST_PASSWORD}
secrets: password
journeys:
  - id: sign_in
    journey: Sign in as {{email}} using {{password}}.
""",
    )

    journey = load_yewo_project(path).workflows[0].journeys[0]

    assert journey.objective == "Sign in as {{email}} using {{password}}."
    assert journey.declared_values["password"].value == "not-for-prose"
    assert journey.declared_values["password"].secret is True
    assert "not-for-prose" not in repr(journey)


def test_inherits_plain_language_checks_without_translating_them(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "yewo.yaml",
        """
checks:
  - The user's work must remain after the app restarts.
workflows:
  editing:
    checks:
      - Repeating save must not create duplicates.
    journeys:
      - id: edit_work
        journey: A user creates and edits some work.
        checks:
          - Cancelling an edit must preserve the previous version.
""",
    )

    journey = load_yewo_project(path).workflows[0].journeys[0]

    assert journey.checks == (
        "The user's work must remain after the app restarts.",
        "Repeating save must not create duplicates.",
        "Cancelling an edit must preserve the previous version.",
    )


def test_rejects_an_undeclared_value_in_a_plain_language_check(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "yewo.yaml",
        """
journey: A user saves some work.
checks:
  - The saved result must contain {{missing_value}}.
""",
    )

    with pytest.raises(YewoProjectError, match="missing_value"):
        load_yewo_project(path)


def test_reports_an_unset_environment_reference_without_its_value(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "yewo.yaml",
        """
env:
  password:
    value: ${YEWO_UNSET_PASSWORD}
    secret: true
journey: Sign in using {{password}}.
""",
    )

    with pytest.raises(YewoProjectError, match="YEWO_UNSET_PASSWORD") as captured:
        load_yewo_project(path)

    assert "password:" not in str(captured.value)


def test_rejects_an_undeclared_journey_value(tmp_path: Path) -> None:
    path = _write(tmp_path / "yewo.yaml", "journey: Use {{missing_value}}.\n")

    with pytest.raises(YewoProjectError, match="missing_value"):
        load_yewo_project(path)


def test_reports_the_location_of_an_unknown_key(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "yewo.yaml",
        """
workflows:
  smoke:
    journyes:
      - journey: Open the app.
""",
    )

    with pytest.raises(YewoProjectError) as captured:
        load_yewo_project(path)

    assert "workflows.smoke.journyes" in str(captured.value)


def test_finds_the_nearest_project_and_selects_workflows(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "yewo.yaml",
        """
workflows:
  first:
    journey: Do the first task.
  second:
    journey: Do the second task.
""",
    )
    nested = tmp_path / "app" / "src"
    nested.mkdir(parents=True)

    assert find_yewo_project(nested) == path
    project = load_nearest_yewo_project(nested)
    assert [item.name for item in select_workflows(project, "second, first")] == [
        "second",
        "first",
    ]


def test_anchors_media_preconditions_to_the_project(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "project" / "yewo.yaml",
        """
preconditions:
  capabilities:
    - kind: media_inject
      local_path: fixtures/photo.png
journey: Select the injected photo.
""",
    )

    capability = load_yewo_project(path).workflows[0].journeys[0].preconditions.capabilities[0]

    assert isinstance(capability, MediaInjectCapabilitySpec)
    assert capability.local_path == str((path.parent / "fixtures/photo.png").resolve())


def test_rejects_duplicate_journey_identifiers(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "yewo.yaml",
        """
journeys:
  - id: duplicate
    journey: First task.
  - id: duplicate
    journey: Second task.
""",
    )

    with pytest.raises(YewoProjectError, match="must be unique"):
        load_yewo_project(path)


def test_rejects_both_inline_journey_aliases_with_a_formatted_error(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "yewo.yaml",
        """
device: {name: Pixel_8, platform: android}
app: app.apk
model: openrouter/example/model
workflows:
  smoke:
    journey: Sign in.
    description: Sign in again.
""",
    )

    with pytest.raises(YewoProjectError, match="workflows.smoke.*exactly one"):
        load_yewo_project(path)


def test_expands_environment_references_in_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("YEWO_TEST_APP", "build/test.apk")
    monkeypatch.setenv("YEWO_TEST_MODEL", "openrouter/example/semantic")
    path = _write(
        tmp_path / "yewo.yaml",
        """
app: ${YEWO_TEST_APP}
model: ${YEWO_TEST_MODEL}
grounder:
  model: ${YEWO_TEST_MODEL}
  protocol: source_pixels
""",
    )

    settings = load_yewo_project(path).settings

    assert settings.app_path == tmp_path / "build/test.apk"
    assert settings.model == "openrouter/example/semantic"
    assert settings.grounder is not None
    assert settings.grounder.model == "openrouter/example/semantic"


def test_secret_name_can_mark_an_actor_value(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "yewo.yaml",
        """
secrets: password
actors:
  admin:
    password: actor-secret
journey: Sign in with {{actors.admin.password}}.
""",
    )

    actor_value = load_yewo_project(path).workflows[0].journeys[0].actors["admin"]["password"]

    assert actor_value.secret is True
    assert actor_value.value == "actor-secret"


def test_background_resume_default_is_part_of_the_frozen_capability_record() -> None:
    capability = AppStateCapabilitySpec(action="background_resume")

    assert capability.duration_seconds == 1.0
    assert "duration_seconds" in capability.model_fields_set


def test_yaml_and_command_line_share_the_same_settlement_limit(tmp_path: Path) -> None:
    path = _write(tmp_path / "yewo.yaml", "settle_time: 6\n")

    with pytest.raises(YewoProjectError, match="settle_time.*less than or equal to 5"):
        load_yewo_project(path)


def test_yaml_rejects_a_duration_above_the_runtime_limit(tmp_path: Path) -> None:
    path = _write(tmp_path / "yewo.yaml", "max_duration_seconds: 14401\n")

    with pytest.raises(
        YewoProjectError,
        match="max_duration_seconds.*less than or equal to 14400",
    ):
        load_yewo_project(path)
