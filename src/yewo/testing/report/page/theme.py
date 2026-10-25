"""The page's stylesheet, its one script, and the marks it draws.

The style and the script are files under `static/`. Each is read once and
written into the page. A report stays one file, and it opens with no network.
Keeping them out of Python also keeps CSS in a `.css` file.

The directory is `static/` rather than `assets/`. The repository ignores every
`assets/`, because that is where a run writes what it produced. A stylesheet
under that name would never be committed, and would never ship.
"""

from functools import lru_cache
from importlib import resources

from yewo.brand import mark_svg


@lru_cache(maxsize=None)
def stylesheet() -> str:
    """Yewo Cloud's palette and layout, as one inline stylesheet."""
    asset = resources.files(__package__).joinpath("static/report.css")
    return asset.read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def execution_stylesheet() -> str:
    """Layout additions used only by the target-execution index."""
    asset = resources.files(__package__).joinpath("static/execution.css")
    return asset.read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def script() -> str:
    """The theme button. It is the only part of the page that needs script."""
    asset = resources.files(__package__).joinpath("static/report.js")
    return asset.read_text(encoding="utf-8")


ICONS = {
    "pass": '<svg class="mark" width="18" height="18" viewBox="0 0 24 24" fill="none" '
    'stroke="var(--pass-text)" stroke-width="2.5" stroke-linecap="round" '
    'stroke-linejoin="round" role="img" aria-label="Passed"><path d="M5 12l5 5 9-10"/></svg>',
    "fail": '<svg class="mark" width="18" height="18" viewBox="0 0 24 24" fill="none" '
    'stroke="var(--fail-text)" stroke-width="2.5" stroke-linecap="round" '
    'role="img" aria-label="Failed"><path d="M6 6l12 12M18 6L6 18"/></svg>',
    "warn": '<svg class="mark" width="18" height="18" viewBox="0 0 24 24" fill="none" '
    'stroke="var(--warn-text)" stroke-width="2.5" stroke-linecap="round" '
    'role="img" aria-label="No verdict"><path d="M12 7v7M12 17.5v.5"/></svg>',
    "muted": '<svg class="mark" width="18" height="18" viewBox="0 0 24 24" fill="none" '
    'stroke="var(--line-strong)" stroke-width="2.5" stroke-linecap="round" '
    'role="img" aria-label="Not reached"><path d="M6 12h12"/></svg>',
}
"""One mark per tone, each labelled, so a reader who cannot see colour reads it."""

MARKS = {
    "logo": mark_svg(
        21,
        19,
        foreground="var(--logo)",
        accent="var(--clay)",
        stroke_width=7,
    ),
    "lock": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V8a4 4 0 0 1 8 0v3"/></svg>',
    "caret": '<svg class="caret" width="16" height="16" viewBox="0 0 24 24" fill="none" '
    'stroke="var(--muted)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
    'aria-hidden="true"><path d="M6 9l6 6 6-6"/></svg>',
    "toggle": '<button type="button" class="theme" '
    'aria-label="Switch between light and dark">'
    '<svg class="sun" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/>'
    '<path d="M12 2v2M12 20v2M2 12h2M20 12h2M5 5l1.5 1.5M17.5 17.5L19 19M5 19l1.5-1.5M17.5 6.5L19 5"/>'
    '</svg>'
    '<svg class="moon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z"/></svg></button>',
}
"""The marks the page draws itself: its logo, its lock, its caret, its button."""
