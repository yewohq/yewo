"""Discovery artifacts refuse incompatible or unverifiable interpretations."""

import pytest
from pydantic import ValidationError

from yewo.discovery.artifacts import (
    ArtifactReference,
    ArtifactRedaction,
    DiscoveryCompatibilityKey,
    validate_shared_knowledge_payload,
)


def test_compatibility_reports_every_changed_dimension(compatibility):
    changed = compatibility.model_copy(
        update={
            "app_artifact_sha256": "d" * 64,
            "orientation": "landscape",
            "discovery_policy_version": "discovery-policy-v2",
        }
    )

    assert compatibility.incompatibilities(changed) == (
        "app_artifact_sha256",
        "orientation",
        "discovery_policy_version",
    )
    assert not compatibility.is_compatible_with(changed)


def test_compatibility_is_exact_for_the_same_context(compatibility):
    reread = DiscoveryCompatibilityKey.model_validate(
        compatibility.model_dump(mode="json")
    )

    assert compatibility.is_compatible_with(reread)


@pytest.mark.parametrize(
    "field,value",
    [
        ("content_scrubbed", False),
        ("contains_secrets", True),
        ("contains_declared_values", True),
    ],
)
def test_redaction_cannot_claim_unsafe_shared_content(field, value):
    payload = {"policy_version": "redaction-v1", field: value}

    with pytest.raises(ValidationError):
        ArtifactRedaction.model_validate(payload)


@pytest.mark.parametrize("path", ["/absolute/evidence.png", "../outside.png"])
def test_evidence_paths_cannot_escape_the_artifact(path):
    with pytest.raises(ValidationError, match="repository-relative"):
        ArtifactReference(artifact_id="evidence", sha256="a" * 64, path=path)


def test_evidence_path_may_be_absent_or_relative():
    assert ArtifactReference(artifact_id="one", sha256="a" * 64).path is None
    assert (
        ArtifactReference(
            artifact_id="two", sha256="b" * 64, path="runs/evidence.png"
        ).path
        == "runs/evidence.png"
    )


def test_raw_shared_payload_rejects_scenario_owned_fields():
    with pytest.raises(ValueError, match="scenario fields"):
        validate_shared_knowledge_payload(
            {"route": [{"verdict": "passed"}, {"scenario_uuid": "scenario-1"}]}
        )
