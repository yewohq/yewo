"""Atomic persistence for the canonical machine-readable run record."""

from pathlib import Path

from yewo.artifacts import REPORTS_DIRECTORY, AtomicArtifactWriter, ArtifactWriteError
from yewo.reporting.errors import CanonicalRunRecordWriteError
from yewo.reporting.records import CanonicalRunRecord


CANONICAL_RUN_RECORD_PATH = REPORTS_DIRECTORY / "run.json"


class CanonicalRunRecordStore:
    """Persist one immutable canonical record beneath a run root."""

    def __init__(self, run_root: Path) -> None:
        self._writer = AtomicArtifactWriter(run_root)

    def persist(self, record: CanonicalRunRecord) -> Path:
        """Persist the canonical record without replacing existing evidence."""
        content = f"{record.model_dump_json(indent=2)}\n".encode()
        try:
            self._writer.write(CANONICAL_RUN_RECORD_PATH, content)
        except ArtifactWriteError as error:
            raise CanonicalRunRecordWriteError(
                "Could not persist the canonical run record."
            ) from error
        return CANONICAL_RUN_RECORD_PATH
