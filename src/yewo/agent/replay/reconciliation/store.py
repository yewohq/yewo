"""Immutable persistence for reconciliation and compilation artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from yewo.agent.replay.reconciliation.records import (
    CompilationOutcome,
    CompilerRefusal,
    EvidenceDocument,
    ReconcilerResponse,
    ReplayProcedure,
    ProcedureCompilationReport,
)
from yewo.artifacts import ARTIFACTS_DIRECTORY, ArtifactWriteError, AtomicArtifactWriter


RECONCILIATION_DIRECTORY = ARTIFACTS_DIRECTORY / "agent" / "reconciliation"
EVIDENCE_DOCUMENT_PATH = RECONCILIATION_DIRECTORY / "evidence.json"
RECONCILER_RESPONSE_PATH = RECONCILIATION_DIRECTORY / "response.json"
COMPILER_REFUSAL_PATH = RECONCILIATION_DIRECTORY / "refusal.json"
REPLAY_PROCEDURE_PATH = RECONCILIATION_DIRECTORY / "procedure.json"
PROCEDURE_COMPILATION_REPORT_PATH = RECONCILIATION_DIRECTORY / "compilation-report.json"


class ReconciliationPersistenceError(RuntimeError):
    """A reconciliation artifact could not be stored immutably."""


class ReconciliationArtifactStore:
    """Persist one source run's evidence and compiler outcome."""

    def __init__(self, root: Path) -> None:
        self._writer = AtomicArtifactWriter(root)

    def persist_evidence(self, evidence: EvidenceDocument) -> Path:
        """Persist one deterministic evidence document."""
        self._write(EVIDENCE_DOCUMENT_PATH, evidence.model_dump(mode="json"))
        return EVIDENCE_DOCUMENT_PATH

    def persist_response(self, response: ReconcilerResponse) -> Path:
        """Persist the complete untrusted reconciler response."""
        self._write(RECONCILER_RESPONSE_PATH, response.model_dump(mode="json"))
        return RECONCILER_RESPONSE_PATH

    def persist_compilation(self, outcome: CompilationOutcome) -> Path:
        """Persist either a procedure or a refusal, never both for one outcome."""
        if isinstance(outcome, CompilerRefusal):
            self._write(COMPILER_REFUSAL_PATH, outcome.model_dump(mode="json"))
            return COMPILER_REFUSAL_PATH
        if not isinstance(outcome, ReplayProcedure):
            raise TypeError("Compilation persistence requires a procedure or refusal.")
        self._write(REPLAY_PROCEDURE_PATH, outcome.model_dump(mode="json"))
        return REPLAY_PROCEDURE_PATH

    def persist_compilation_report(self, report: ProcedureCompilationReport) -> Path:
        """Persist one non-executing reconciliation summary."""
        self._write(PROCEDURE_COMPILATION_REPORT_PATH, report.model_dump(mode="json"))
        return PROCEDURE_COMPILATION_REPORT_PATH

    def _write(self, path: Path, content: dict[str, object]) -> None:
        try:
            encoded = (
                json.dumps(content, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
                + "\n"
            ).encode("utf-8")
            self._writer.write(path, encoded)
        except (ArtifactWriteError, TypeError, ValueError) as error:
            raise ReconciliationPersistenceError(
                f"Could not persist reconciliation artifact: {path.as_posix()}"
            ) from error
