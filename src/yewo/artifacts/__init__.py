"""Shared contracts and persistence for immutable run artifacts."""

from yewo.artifacts.errors import ArtifactWriteError
from yewo.artifacts.paths import (
    ARTIFACTS_DIRECTORY,
    DEFAULT_OUTPUT_DIRECTORY,
    REPORTS_DIRECTORY,
    RUN_RECORDS_DIRECTORY,
    RUNTIME_DIRECTORY,
)
from yewo.artifacts.redaction_mask import (
    YEWO_REDACTION_POLICY_ID,
    draw_yewo_redaction_mask,
)
from yewo.artifacts.records import (
    ArtifactRetention,
    ArtifactSensitivity,
    EvidenceApplicationIdentity,
    EvidenceDeviceIdentity,
    RecordIdentifier,
    Sha256Digest,
    new_record_identifier,
    validate_record_identifier,
)
from yewo.artifacts.store import AppendOnlyJsonlWriter, AtomicArtifactWriter

__all__ = [
    "ARTIFACTS_DIRECTORY",
    "DEFAULT_OUTPUT_DIRECTORY",
    "REPORTS_DIRECTORY",
    "RUN_RECORDS_DIRECTORY",
    "RUNTIME_DIRECTORY",
    "ArtifactRetention",
    "ArtifactSensitivity",
    "ArtifactWriteError",
    "AppendOnlyJsonlWriter",
    "AtomicArtifactWriter",
    "EvidenceApplicationIdentity",
    "EvidenceDeviceIdentity",
    "RecordIdentifier",
    "Sha256Digest",
    "YEWO_REDACTION_POLICY_ID",
    "draw_yewo_redaction_mask",
    "new_record_identifier",
    "validate_record_identifier",
]
