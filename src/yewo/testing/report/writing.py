"""Writing every report a run leaves, and failing the run when one will not write.

Three readers, three files, one directory. Markdown is for a person scanning a
terminal. JUnit is for CI. The page is for whoever opens the artifact
afterwards. They are written together. A run that produced verdicts and
recorded none of them is not a passing run. Nothing downstream could tell it
from a run that never started.

Each is attempted even when an earlier one failed, and each failure is named.
A page that could not be written is not a reason to also lose the JUnit file.
"""

import logging
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from yewo.agent.device.session import DeviceSession
from yewo.errors import ArtifactWriteFailed
from yewo.testing.models import JourneyReport
from yewo.utils.secrets import SecretScrubber

from . import infrastructure
from .page import RUN_FILES, save_html_report
from .reports import save_junit_report, save_markdown_report

logger = logging.getLogger(__name__)

Failure = Tuple[str, ArtifactWriteFailed]
"""Which report would not write, under the name the record files it by."""


def _writers(
    report: JourneyReport,
    output_dir: Path,
    session: Optional[DeviceSession],
    scrubber: Optional[SecretScrubber],
    usage: Optional[dict],
    label: Optional[str],
) -> Tuple[Tuple[str, Callable[[], Path]], ...]:
    """Each report, named as the record names it when it fails."""
    return (
        (
            "markdown_report",
            lambda: save_markdown_report(report, output_dir, session, scrubber, label),
        ),
        (
            "junit_report",
            lambda: save_junit_report(report, output_dir, scrubber, label),
        ),
        (
            "html_report",
            lambda: save_html_report(
                report,
                output_dir,
                screenshots=session.screenshots_dir if session is not None else None,
                scrubber=scrubber,
                label=label,
                usage=usage,
                artifacts=RUN_FILES if session is not None else (),
            ),
        ),
    )


def _attempt(
    writers: Tuple[Tuple[str, Callable[[], Path]], ...],
) -> Tuple[List[str], List[Failure]]:
    """Runs each writer once. Returns the ones that wrote, and the ones that did not."""
    wrote: List[str] = []
    failures: List[Failure] = []
    for name, write in writers:
        try:
            write()
            wrote.append(name)
        except ArtifactWriteFailed as failure:
            logger.error("Failed to save %s: %s", name, failure)
            failures.append((name, failure))
    return wrote, failures


def write_reports(
    report: JourneyReport,
    assets_dir: Path,
    session: Optional[DeviceSession] = None,
    scrubber: Optional[SecretScrubber] = None,
    usage: Optional[dict] = None,
    label: Optional[str] = None,
    record_failure: Optional[Callable[[str, ArtifactWriteFailed], None]] = None,
) -> List[Failure]:
    """Writes all three into this run's directory, and returns what would not write.

    A report that will not write fails the run. The reports that did write
    said the run passed. So they are written again once the failure is on
    the report. Without the second pass a reader holds a green JUnit file, a
    green page, and a record that says FAILED.

    `record_failure` puts one failure on the report. It is called between the
    two passes, so the rewrite carries what it did.
    """
    output_dir = session.telemetry_dir if session is not None else assets_dir
    writers = _writers(report, output_dir, session, scrubber, usage, label)
    wrote, failures = _attempt(writers)
    if not failures or record_failure is None:
        return failures

    for name, failure in failures:
        record_failure(name, failure)
    _rewrite(writers, wrote)
    return failures


def _rewrite(
    writers: Tuple[Tuple[str, Callable[[], Path]], ...], wrote: Sequence[str]
) -> None:
    """Writes again whichever reports wrote the first time.

    Only those. A writer that just failed is not asked twice. A failure
    here is logged rather than put on the report. The run already reads
    FAILED, which is the fact a reader acts on.
    """
    _attempt(tuple(pair for pair in writers if pair[0] in wrote))


def persist(
    report: JourneyReport,
    assets_dir: Path,
    session: Optional[DeviceSession],
    scrubber: SecretScrubber,
    manifest,
    unevidenced: List[str],
    usage: dict,
) -> None:
    """Writes every report of a finished run, then closes the record naming them.

    Two things fail the run after a report is already on disk. One is a
    report that would not write. The other is the record itself. Both are
    found after the reports that did write said the run passed, so both
    write them again. A reader must never hold a green JUnit file beside a
    record that says FAILED.
    """
    output_dir = session.telemetry_dir if session is not None else assets_dir
    writers = _writers(report, output_dir, session, scrubber, usage, None)

    def failed(name: str, failure: Exception) -> None:
        infrastructure.write_failure(report, name, failure)
        if manifest is not None:
            manifest.add(report.scenarios[-1])

    wrote, failures = _attempt(writers)
    if failures:
        for name, failure in failures:
            failed(name, failure)
        _rewrite(writers, wrote)

    if manifest is None:
        return
    # Last, so the record carries whichever reports failed to write.
    manifest.finish(
        report.status,
        unevidenced=unevidenced,
        usage=usage,
        duration=report.total_duration,
    )
    if manifest.unwritten is not None:
        failed("run_manifest", manifest.unwritten)
        _rewrite(writers, wrote)


def checkpoint(
    report: JourneyReport,
    assets_dir: Path,
    session: Optional[DeviceSession],
    scrubber: SecretScrubber,
) -> None:
    """Writes what a run has learned so far, under one name it keeps.

    A run that dies part way otherwise leaves nothing about the scenarios that
    already passed. A failure here is recorded in the log and nowhere else. The
    run is still going. A checkpoint is not the record it is judged on.
    """
    for name, failure in write_reports(
        report, assets_dir, session, scrubber, label="partial"
    ):
        logger.error("Could not checkpoint the %s: %s", name, failure)
