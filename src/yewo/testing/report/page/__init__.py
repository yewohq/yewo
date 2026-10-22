"""The run report as a page, for whoever has to read a red build."""

from .record import page_of_record, run_of
from .render import document, now, save_html_report, save_page
from .view import RUN_FILES

__all__ = [
    "RUN_FILES",
    "document",
    "now",
    "page_of_record",
    "run_of",
    "save_html_report",
    "save_page",
]
