"""Construct the canonical record before or after a partial startup."""

from yewo.testing import lock_file, lock_resolution, sensitivity
from yewo.testing.report.manifest import RunInputs, RunManifest, RunSubject


def create_manifest(
    path,
    journey,
    scenarios,
    environment,
    suite,
    scrubber,
    run_id=None,
    allow_secret_exposure=False,
) -> RunManifest:
    """Build the same run record at every runner failure boundary."""
    return RunManifest(
        path,
        RunSubject(
            kind="journey",
            id=journey.id,
            name=scrubber.scrub(journey.name),
            description=scrubber.scrub(journey.description),
        ),
        scenarios,
        environment=environment,
        inputs=RunInputs(
            subject=journey.id,
            subject_sha256=lock_file.journey_sha256(journey),
            coverage_declared=[cell.key for cell in journey.coverage],
            lock_sha256=suite.lock_sha256,
            scenarios_from_lock=suite.reused_count,
            scenarios_generated=suite.generated_count,
        ),
        scrubber=scrubber,
        artifacts=sensitivity.declared_for(journey, allow_secret_exposure),
        run_uuid=run_id,
    )


def ensure_interrupted_manifest(
    manifest, runner, journey, scenarios, llm, session, scrubber, calls
):
    """Recover the planned record when device startup died before it existed."""
    if manifest is not None:
        return manifest, scenarios
    suite = runner._suite(journey, llm) if runner.locked else lock_resolution.LockResolution()
    planned = suite.scenarios if runner.locked else scenarios
    recovered = create_manifest(
        session.manifest_path,
        journey,
        planned,
        runner._environment(),
        suite,
        scrubber,
        runner.run_id,
        runner.allow_secret_exposure,
    )
    recovered.record_unattached_calls(calls)
    return recovered, planned
