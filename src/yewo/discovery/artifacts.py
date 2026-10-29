"""Compatibility, provenance, integrity, and isolation for discovery artifacts."""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import PurePosixPath
from typing import Literal, Sequence, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yewo import __version__
from yewo.identity import new_uuid

SHA256_PATTERN = r"^[0-9a-f]{64}$"
DISCOVERY_ARTIFACT_SCHEMA_VERSION = 1


class DiscoveryContract(BaseModel):
    """Strict immutable base for persisted discovery contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ArtifactReference(DiscoveryContract):
    """One immutable evidence artifact used to build another artifact."""

    artifact_id: str
    sha256: str = Field(pattern=SHA256_PATTERN)
    path: str | None = None

    @model_validator(mode="after")
    def _validate_relative_path(self) -> "ArtifactReference":
        if self.path is None:
            return self
        path = PurePosixPath(self.path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("An evidence path must be repository-relative.")
        return self


class ArtifactProvenance(DiscoveryContract):
    """Inputs and producer needed to reproduce a derived artifact."""

    producer: str = "yewo"
    producer_version: str = __version__
    source_run_ids: tuple[str, ...] = ()
    sources: tuple[ArtifactReference, ...] = ()


class ArtifactRedaction(DiscoveryContract):
    """Machine-checkable declaration for content allowed in shared knowledge."""

    policy_version: str
    content_scrubbed: Literal[True] = True
    contains_secrets: Literal[False] = False
    contains_declared_values: Literal[False] = False


class ArtifactMigration(DiscoveryContract):
    """One explicit interpretation change applied to an artifact."""

    migration_id: str
    source_schema: str
    source_version: int = Field(ge=1)
    target_schema: str
    target_version: int = Field(ge=1)
    rule_version: str
    source_digest: str = Field(pattern=SHA256_PATTERN)


class PerceptionArtifactIdentity(DiscoveryContract):
    """One detector, OCR, or fusion artifact required to interpret evidence."""

    name: str
    version: str
    sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)


class DiscoveryCompatibilityKey(DiscoveryContract):
    """Exact context in which discovery knowledge may be interpreted."""

    app_id: str
    app_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    platform: str
    viewport_class: str
    orientation: str
    setup_contract_sha256: str = Field(pattern=SHA256_PATTERN)
    discovery_policy_version: str
    screen_identity_version: str
    control_identity_version: str
    perception_artifacts: tuple[PerceptionArtifactIdentity, ...]

    def incompatibilities(self, other: "DiscoveryCompatibilityKey") -> tuple[str, ...]:
        """Return every dimension that requires revalidation."""
        fields = (
            "app_id",
            "app_artifact_sha256",
            "platform",
            "viewport_class",
            "orientation",
            "setup_contract_sha256",
            "discovery_policy_version",
            "screen_identity_version",
            "control_identity_version",
            "perception_artifacts",
        )
        return tuple(name for name in fields if getattr(self, name) != getattr(other, name))

    def is_compatible_with(self, other: "DiscoveryCompatibilityKey") -> bool:
        return not self.incompatibilities(other)


class DiscoveryArtifactHeader(DiscoveryContract):
    """Common envelope carried by every persisted discovery artifact."""

    artifact_id: str = Field(default_factory=new_uuid)
    schema_name: str
    schema_version: Literal[1] = DISCOVERY_ARTIFACT_SCHEMA_VERSION
    compatibility: DiscoveryCompatibilityKey
    provenance: ArtifactProvenance
    redaction: ArtifactRedaction
    migrations: tuple[ArtifactMigration, ...] = ()
    integrity_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)


class DiscoveryArtifact(DiscoveryContract):
    """Base for a versioned artifact whose canonical payload is signed by digest."""

    header: DiscoveryArtifactHeader


DiscoveryArtifactT = TypeVar("DiscoveryArtifactT", bound=DiscoveryArtifact)


SHARED_KNOWLEDGE_FORBIDDEN_FIELDS = frozenset(
    {
        "scenario_uuid",
        "scenario_id",
        "test_id",
        "scenario_values",
        "declared_values",
        "findings",
        "obligations",
        "recovery_state",
        "verdict",
        "expected_outcomes",
    }
)


def _canonical_payload(artifact: DiscoveryArtifact) -> bytes:
    payload = artifact.model_dump(mode="json")
    payload["header"]["integrity_sha256"] = None
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def artifact_sha256(artifact: DiscoveryArtifact) -> str:
    """Digest one canonical spelling with the digest field cleared."""
    return hashlib.sha256(_canonical_payload(artifact)).hexdigest()


def _validate_isolated_payload(value: object, forbidden_values: frozenset[str]) -> None:
    if isinstance(value, dict):
        forbidden_fields = SHARED_KNOWLEDGE_FORBIDDEN_FIELDS.intersection(value)
        if forbidden_fields:
            names = ", ".join(sorted(forbidden_fields))
            raise ValueError(f"Shared app knowledge cannot contain scenario fields: {names}.")
        for nested in value.values():
            _validate_isolated_payload(nested, forbidden_values)
        return
    if isinstance(value, list | tuple):
        for nested in value:
            _validate_isolated_payload(nested, forbidden_values)
        return
    if (
        isinstance(value, str)
        and value
        and any(forbidden in value for forbidden in forbidden_values)
    ):
        raise ValueError("Shared app knowledge contains an isolated scenario value.")


def validate_shared_knowledge_payload(
    value: object, *, forbidden_scenario_values: Sequence[str] = ()
) -> None:
    """Refuse scenario-owned fields and values before shared-schema validation."""
    forbidden_values = frozenset(value for value in forbidden_scenario_values if value)
    _validate_isolated_payload(value, forbidden_values)


def seal_artifact(
    artifact: DiscoveryArtifactT,
    *,
    forbidden_scenario_values: Sequence[str] = (),
) -> DiscoveryArtifactT:
    """Validate isolation and return a copy sealed with its canonical digest."""
    validate_shared_knowledge_payload(
        artifact.model_dump(mode="json"),
        forbidden_scenario_values=forbidden_scenario_values,
    )
    digest = artifact_sha256(artifact)
    header = artifact.header.model_copy(update={"integrity_sha256": digest})
    return artifact.model_copy(update={"header": header})


def verify_artifact(artifact: DiscoveryArtifact) -> bool:
    """Whether the stored digest matches the complete interpreted payload."""
    stored = artifact.header.integrity_sha256
    return stored is not None and hmac.compare_digest(stored, artifact_sha256(artifact))


def serialize_artifact(artifact: DiscoveryArtifact) -> bytes:
    """Serialize only a sealed artifact; drafts are not durable evidence."""
    if not verify_artifact(artifact):
        raise ValueError("A discovery artifact must have a valid integrity digest.")
    payload = artifact.model_dump(mode="json")
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def parse_artifact(
    content: bytes, artifact_type: type[DiscoveryArtifactT]
) -> DiscoveryArtifactT:
    """Validate schema and integrity before exposing a persisted artifact."""
    artifact = artifact_type.model_validate_json(content)
    if not verify_artifact(artifact):
        raise ValueError("The discovery artifact integrity digest does not match its payload.")
    return artifact
