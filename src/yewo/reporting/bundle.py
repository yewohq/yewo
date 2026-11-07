"""Write every derived report without losing independent successes."""

from collections.abc import Callable
from pathlib import Path

from yewo.artifacts import REPORTS_DIRECTORY, AtomicArtifactWriter, ArtifactWriteError
from yewo.reporting.html import render_html
from yewo.reporting.junit import render_junit
from yewo.reporting.markdown import render_markdown
from yewo.reporting.records import (
    CanonicalReportOutput,
    CanonicalRunRecord,
    ReportOutputDisposition,
    ReportOutputFormat,
)


MARKDOWN_REPORT_PATH = REPORTS_DIRECTORY / "report.md"
HTML_REPORT_PATH = REPORTS_DIRECTORY / "report.html"
JUNIT_REPORT_PATH = REPORTS_DIRECTORY / "junit.xml"


def write_report_bundle(
    run_root: Path,
    record: CanonicalRunRecord,
) -> tuple[CanonicalReportOutput, ...]:
    """Attempt every renderer and return an observed output result for each."""
    writer = AtomicArtifactWriter(run_root)
    outputs: list[CanonicalReportOutput] = []
    renderers: tuple[
        tuple[ReportOutputFormat, Path, Callable[[CanonicalRunRecord], str | bytes]], ...
    ] = (
        (ReportOutputFormat.MARKDOWN, MARKDOWN_REPORT_PATH, render_markdown),
        (ReportOutputFormat.HTML, HTML_REPORT_PATH, render_html),
        (ReportOutputFormat.JUNIT, JUNIT_REPORT_PATH, render_junit),
    )
    for report_format, path, render in renderers:
        try:
            content = render(record)
            writer.write(path, content.encode("utf-8") if isinstance(content, str) else content)
        except (ArtifactWriteError, OSError, TypeError, ValueError) as error:
            outputs.append(
                CanonicalReportOutput(
                    format=report_format,
                    disposition=ReportOutputDisposition.FAILED,
                    path=path,
                    reason=f"Yewo could not write the {report_format.value} report: {type(error).__name__}.",
                )
            )
        else:
            outputs.append(
                CanonicalReportOutput(
                    format=report_format,
                    disposition=ReportOutputDisposition.WRITTEN,
                    path=path,
                )
            )
    return tuple(outputs)
