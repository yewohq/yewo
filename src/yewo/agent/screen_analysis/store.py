"""Append-only persistence for current-screen analysis evidence."""

from pathlib import Path

from pydantic import ValidationError

from yewo.agent.screen_analysis.records import CurrentScreenAnalysis
from yewo.artifacts import (
    ARTIFACTS_DIRECTORY,
    AppendOnlyJsonlWriter,
    ArtifactWriteError,
    RecordIdentifier,
)


SCREEN_ANALYSES_PATH = ARTIFACTS_DIRECTORY / "agent" / "screen-analyses.jsonl"


class ScreenAnalysisStoreError(RuntimeError):
    """Screen-analysis evidence could not be persisted or loaded safely."""


class ScreenAnalysisStore:
    """Persist canonical analyses and load selected records by identity."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._writer = AppendOnlyJsonlWriter(root)

    def persist(self, analysis: CurrentScreenAnalysis) -> Path:
        """Append one immutable analysis to the run ledger."""
        try:
            self._writer.append(
                SCREEN_ANALYSES_PATH,
                analysis.model_dump(mode="json"),
                identity_field="analysis_id",
            )
        except ArtifactWriteError as error:
            raise ScreenAnalysisStoreError("Could not persist screen-analysis evidence.") from error
        return SCREEN_ANALYSES_PATH

    def load_selected(
        self,
        analysis_ids: tuple[RecordIdentifier, ...],
    ) -> tuple[CurrentScreenAnalysis, ...]:
        """Load named analyses in requested order and reject missing identities."""
        if not analysis_ids:
            return ()
        requested = tuple(analysis_ids)
        path = self.root / SCREEN_ANALYSES_PATH
        try:
            if path.is_symlink() or not path.is_file():
                raise ScreenAnalysisStoreError("The screen-analysis ledger is unavailable.")
            selected: dict[str, CurrentScreenAnalysis] = {}
            requested_set = set(requested)
            with path.open("rb") as stream:
                for line in stream:
                    if not line.endswith(b"\n"):
                        raise ScreenAnalysisStoreError(
                            "The screen-analysis ledger has an incomplete final line."
                        )
                    analysis = CurrentScreenAnalysis.model_validate_json(line)
                    if analysis.analysis_id in requested_set:
                        if analysis.analysis_id in selected:
                            raise ScreenAnalysisStoreError(
                                "The screen-analysis ledger repeats an analysis identity."
                            )
                        selected[analysis.analysis_id] = analysis
        except ScreenAnalysisStoreError:
            raise
        except (OSError, ValidationError) as error:
            raise ScreenAnalysisStoreError("The screen-analysis ledger is invalid.") from error
        missing = requested_set - selected.keys()
        if missing:
            raise ScreenAnalysisStoreError(
                "The screen-analysis ledger is missing requested evidence."
            )
        return tuple(selected[analysis_id] for analysis_id in requested)
