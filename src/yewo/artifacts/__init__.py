"""Shared contracts and persistence for immutable run artifacts."""

from yewo.artifacts.errors import ArtifactWriteError
from yewo.artifacts.paths import (
    ARTIFACTS_DIRECTORY,
    DEFAULT_OUTPUT_DIRECTORY,
    REPORTS_DIRECTORY,
    RUN_RECORDS_DIRECTORY,
    RUNTIME_DIRECTORY,
)
from yewo.artifacts.records import (
    ArtifactRetention,
    EvidenceApplicationIdentity,
    EvidenceDeviceIdentity,
    RecordIdentifier,
    RelativeArtifactPath,
    Sha256Digest,
    format_relative_artifact_path,
    new_record_identifier,
    validate_record_identifier,
)
from yewo.artifacts.schema_versions import (
    APP_MEMORY_SCHEMA_VERSION,
    CANONICAL_RUN_RECORD_SCHEMA_VERSION,
    COMPILED_PROCEDURE_SCHEMA_VERSION,
    AppMemorySchemaVersion,
    CanonicalRunRecordSchemaVersion,
    CompiledProcedureSchemaVersion,
)
from yewo.artifacts.store import AppendOnlyJsonlWriter, AtomicArtifactWriter

__all__ = [
    "APP_MEMORY_SCHEMA_VERSION",
    "ARTIFACTS_DIRECTORY",
    "CANONICAL_RUN_RECORD_SCHEMA_VERSION",
    "COMPILED_PROCEDURE_SCHEMA_VERSION",
    "DEFAULT_OUTPUT_DIRECTORY",
    "REPORTS_DIRECTORY",
    "RUN_RECORDS_DIRECTORY",
    "RUNTIME_DIRECTORY",
    "ArtifactRetention",
    "ArtifactWriteError",
    "AppMemorySchemaVersion",
    "AppendOnlyJsonlWriter",
    "AtomicArtifactWriter",
    "CanonicalRunRecordSchemaVersion",
    "CompiledProcedureSchemaVersion",
    "EvidenceApplicationIdentity",
    "EvidenceDeviceIdentity",
    "RecordIdentifier",
    "RelativeArtifactPath",
    "format_relative_artifact_path",
    "Sha256Digest",
    "new_record_identifier",
    "validate_record_identifier",
]
