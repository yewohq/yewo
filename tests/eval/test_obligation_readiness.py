from pathlib import Path

from yewo.evaluation.obligation_readiness import measure_locked_journey
from yewo.testing import lock_file
from yewo.testing.models import GeneratedTestCase, ScenarioDraft, UserJourney
from yewo.testing.steps import ActionStep
from yewo.testing.taxonomy import Category, CoverageCell, Priority, ScenarioType


def test_generated_lock_is_measured_before_strict_routing(tmp_path: Path) -> None:
    cell = CoverageCell(Category.FUNCTIONAL, ScenarioType.HAPPY_PATH, Priority.P0)
    journey = UserJourney(id="j", name="Journey", description="A user opens the app.")
    scenario = GeneratedTestCase.of(
        cell,
        ScenarioDraft(
            scenario_name="Open the app",
            steps=[ActionStep(instruction="Open the app.")],
            expected_outcomes=["The app is visible."],
        ),
    )
    path = tmp_path / "yewo.lock"
    lock_file.write(
        path,
        lock_file.Lock(
            entries=[
                lock_file.LockEntry.of(
                    scenario,
                    journey,
                    "",
                    lock_file.contract_sha256(journey, cell),
                )
            ]
        ),
    )

    result = measure_locked_journey(path)

    assert result.scenario_count > 0
    assert result.obligation_count > 0
    assert result.measurement.total >= result.obligation_count
    assert result.measurement.unknown == 0
    assert result.strict_ready
