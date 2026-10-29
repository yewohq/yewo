import pytest

from yewo.discovery.artifacts import (
    ArtifactProvenance,
    ArtifactRedaction,
    DiscoveryArtifactHeader,
    DiscoveryCompatibilityKey,
    PerceptionArtifactIdentity,
)
from yewo.discovery.contracts import DiscoveryBudgets


@pytest.fixture
def compatibility():
    return DiscoveryCompatibilityKey(
        app_id="com.example.app",
        app_artifact_sha256="a" * 64,
        platform="android",
        viewport_class="phone-portrait",
        orientation="portrait",
        setup_contract_sha256="b" * 64,
        discovery_policy_version="discovery-policy-v1",
        screen_identity_version="screen-state-v1",
        control_identity_version="control-v1",
        perception_artifacts=(
            PerceptionArtifactIdentity(name="yolo-ui", version="1", sha256="c" * 64),
        ),
    )


@pytest.fixture
def header(compatibility):
    def build(schema_name):
        return DiscoveryArtifactHeader(
            schema_name=schema_name,
            compatibility=compatibility,
            provenance=ArtifactProvenance(source_run_ids=("run-1",)),
            redaction=ArtifactRedaction(policy_version="redaction-v1"),
        )

    return build


@pytest.fixture
def budgets():
    return DiscoveryBudgets(
        observations=10,
        actions=8,
        model_proposals=4,
        resets=2,
        duration_ms=60_000,
        state_depth=6,
        cost_micro_usd=50_000,
        side_effect_actions=8,
        attempts_per_transition=2,
    )
