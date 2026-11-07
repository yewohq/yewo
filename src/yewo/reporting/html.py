"""Render one self-contained, template-driven Yewo report."""

from html import escape
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files
import json
import re
from typing import Any

from yewo.reporting.records import CanonicalRunRecord


_STYLE = files("yewo.reporting").joinpath("report.css").read_text(encoding="utf-8")
_SCRIPT = files("yewo.reporting").joinpath("report.js").read_text(encoding="utf-8")
_MARKS = json.loads(
    files("yewo.reporting").joinpath("report_marks.json").read_text(encoding="utf-8")
)
_PLACEHOLDER_PATTERN = re.compile(r"@@([A-Z_]+)@@")

_HTML_SHELL = "\n".join(
    (
        '<!doctype html><html lang="en"><head>',
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Yewo run report · @@REPORT_TITLE@@</title>",
        "<style>@@REPORT_STYLE@@</style></head><body>",
        '<div id="report-root"><header class="top"><div class="bar">',
        '<div class="brand"><span class="logo">@@LOGO@@',
        '<span class="name">yewo</span></span>',
        '<span class="slash" aria-hidden="true">/</span>',
        '<span class="report-label">Run report</span></div>',
        '<div class="stamp"><span>Generated <span id="generated-at"></span>',
        ' · yewo <span id="yewo-version"></span></span>@@THEME_BUTTON@@</div>',
        "</div></header><main>",
        '<div class="title"><span class="muted">Journey · ',
        '<span class="mono" id="journey-id"></span></span>',
        '<h1 id="journey-title"></h1></div>',
        '<section class="verdict" id="verdict" aria-label="Verdict">',
        '<div class="verdict-body"><div class="line">',
        '<span class="badge" id="verdict-badge"></span>',
        '<span class="muted" id="verdict-count"></span></div>',
        '<h2 id="verdict-headline"></h2><p id="verdict-detail"></p></div>',
        '<a class="jump" id="verdict-link" hidden>View evidence</a></section>',
        '<section class="tiles" aria-label="Run summary">',
        '<div class="tile"><span class="label">Scenarios</span>',
        '<span class="value" id="tile-scenarios"></span></div>',
        '<div class="tile"><span class="label">Steps</span>',
        '<span class="value" id="tile-steps"></span>',
        '<span class="note" id="tile-steps-note"></span></div>',
        '<div class="tile"><span class="label">Duration</span>',
        '<span class="value" id="tile-duration"></span></div>',
        '<div class="tile"><span class="label">Estimated Cost</span>',
        '<span class="value" id="tile-cost"></span>',
        '<span class="note" id="tile-cost-note"></span></div>',
        '<div class="tile"><span class="label">Model</span>',
        '<span class="value" id="tile-model"></span></div>',
        '<div class="tile"><span class="label">Device</span>',
        '<span class="value" id="tile-device"></span></div>',
        '<div class="tile"><span class="label">Branch</span>',
        '<span class="value">Not recorded</span></div>',
        '<div class="tile"><span class="label">Build</span>',
        '<span class="value" id="tile-build"></span></div></section>',
        '<section class="panel" aria-label="Journey"><div class="head">',
        '<h2>Test Flow</h2><span class="muted" id="flow-summary"></span></div>',
        '<details class="flow-specification">',
        "<summary>Show full journey specification</summary>",
        '<p id="flow-description"></p></details></section>',
        '<section class="issues" id="findings" hidden><div class="head"><div>',
        '<h2>Issues Found</h2><span class="muted" id="issue-summary"></span></div>',
        '<span class="badge b-fail">Failed</span></div><div id="issue-list"></div></section>',
        '<div class="split"><nav class="rail" id="scenario-rail"',
        ' aria-label="Scenarios" hidden><h2>Scenarios</h2>',
        '<div id="scenario-links"></div></nav>',
        '<div class="walks" id="scenario-list"></div></div>',
        "<footer><span>Saved with this run: ",
        '<a class="mono" href="run.json">run.json</a> · ',
        '<a class="mono" href="report.md">report.md</a> · ',
        '<a class="mono" href="junit.xml">junit.xml</a></span></footer>',
        "</main></div>",
        '<template id="scenario-link-template"><a><span class="dot"></span>',
        '<span class="scenario-title"></span><span class="count"></span></a></template>',
        '<template id="scenario-template"><section class="walk">',
        '<div class="head"><h2></h2>',
        '<span class="muted scenario-status"></span></div>',
        '<span class="mono muted scenario-identity"></span>',
        '<p class="failure-reason scenario-summary"></p>',
        '<p class="failure-reason t-warn restoration-warning" hidden></p>',
        '<div class="interaction-list"></div>',
        '<p class="muted no-interactions" hidden>',
        "This scenario recorded no interaction steps.</p>",
        '<section class="rows"><div class="head"><h2>Verification Results</h2>',
        '<span class="muted">Final result for each expected outcome and QA rule</span>',
        '</div><div class="unresolved-checks"></div>',
        '<details class="passed-checks" hidden><summary></summary>',
        '<div class="passed-check-list"></div></details></section>',
        "</section></template>",
        '<template id="interaction-template"><div class="step"><details>',
        '<summary class="row"><span class="step-number"></span>',
        '<svg class="mark" width="18" height="18" viewBox="0 0 24 24"',
        ' fill="none" stroke-width="2.5" stroke-linecap="round" aria-hidden="true">',
        '<path d="M5 12l5 5 9-10"/></svg>',
        '<span class="step-title"></span><span class="chip"></span>@@CARET@@</summary>',
        '<div class="inside"><div class="shots"><figure class="shot"><a>',
        '<span class="screen"><img loading="lazy"><span class="box" hidden></span>',
        '<span class="tap"></span></span></a><figcaption></figcaption></figure></div>',
        '<p class="muted screenshot-unavailable" hidden>',
        "The interaction screenshot is unavailable.</p>",
        '<dl class="notes"><div><dt>Expected outcome</dt>',
        '<dd class="expected-outcome"></dd></div>',
        '<div><dt>Dispatch result</dt><dd class="dispatch-result"></dd></div>',
        '<div><dt>Target</dt><dd class="target"></dd></div></dl>',
        "</div></details></div></template>",
        '<template id="check-template"><div class="check">',
        '<span class="badge"></span><span class="rule"></span>',
        '<span class="check-reasoning"></span></div></template>',
        '<template id="issue-template"><article class="issue-card">',
        '<div class="issue-evidence"></div><div class="issue-copy">',
        '<span class="rule-label"></span><h3></h3><p></p><a></a>',
        "</div></article></template>",
        '<template id="issue-shot-template"><figure class="issue-shot"><a>',
        '<span class="screen"><img loading="lazy"></span></a>',
        "<figcaption></figcaption></figure></template>",
        '<template id="empty-scenarios-template"><section class="walk">',
        '<div class="head"><h2>No executable scenarios</h2></div>',
        '<p class="failure-reason t-warn"></p></section></template>',
        "<noscript>JavaScript is required to load the saved run data.</noscript>",
        '<script id="report-data" type="application/json">@@REPORT_DATA@@</script>',
        "<script>@@REPORT_SCRIPT@@</script></body></html>",
    )
)


def render_html(record: CanonicalRunRecord) -> str:
    """Return one self-contained local report from the canonical record."""
    title = _journey_title(record)
    substitutions = {
        "REPORT_TITLE": escape(title),
        "REPORT_STYLE": _STYLE,
        "LOGO": _MARKS["logo"],
        "THEME_BUTTON": _MARKS["theme_button"],
        "CARET": _MARKS["caret"],
        "REPORT_DATA": _serialize_report_data(record, title),
        "REPORT_SCRIPT": _SCRIPT,
    }
    return _PLACEHOLDER_PATTERN.sub(lambda match: substitutions[match.group(1)], _HTML_SHELL)


def _serialize_report_data(record: CanonicalRunRecord, title: str) -> str:
    payload: dict[str, Any] = {
        "record": record.model_dump(mode="json"),
        "report": {
            "title": title,
            "yewo_version": _yewo_version(),
        },
    }
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return (
        serialized.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _journey_title(record: CanonicalRunRecord) -> str:
    return (
        record.scenarios[0].name.split(":", maxsplit=1)[0] if record.scenarios else "User Journey"
    )


def _yewo_version() -> str:
    try:
        return version("yewo")
    except PackageNotFoundError:
        return "development"
