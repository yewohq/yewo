"""The one path from a journey file to a live discovery run.

Before this, every piece existed and nothing called any of it. These cases
hold the chain together: a cold run gets a factory, the factory builds the
boundary once a device is up, and the context it records is read from that
device rather than assumed.
"""

import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from discovery_device_fakes import DiscoveryDevice

from yewo.testing.discovery_wiring import (
    CONTROL_IDENTITY_VERSION,
    DISCOVERY_POLICY_VERSION,
    NO_SETUP_CONTRACT_SHA256,
    V0_DISCOVERY_BUDGETS,
    LiveDiscoveryProvider,
    classify_viewport,
)
from yewo.discovery.model_proposer import MODEL_CALLS_PER_PROPOSAL
from yewo.testing.models import UserJourney
from yewo.utils.secrets import SecretScrubber

JOURNEY = UserJourney(
    id="checkout", name="Checkout", description="A user places an order."
)


def _provider(tmp_path, app_path, device=None):
    return LiveDiscoveryProvider(
        device or DiscoveryDevice(),
        SimpleNamespace(current_frame=None),
        model="openrouter/m",
        perception_model="yewo-ui@v1",
        app_path=str(app_path),
        platform="android",
        setup_contract_sha256=NO_SETUP_CONTRACT_SHA256,
        knowledge_root=tmp_path / "knowledge",
        scrubber=SecretScrubber(),
    )


@pytest.fixture
def app_file(tmp_path):
    path = tmp_path / "app.apk"
    path.write_bytes(b"an application artifact")
    return path


def test_the_context_is_read_from_the_device_not_assumed(app_file, tmp_path):
    compatibility = _provider(tmp_path, app_file)._compatibility()

    assert compatibility.app_id == "com.example.app"
    assert compatibility.platform == "android"
    assert compatibility.orientation == "portrait"
    assert compatibility.viewport_class == "phone-portrait"


def test_the_app_artifact_is_fingerprinted_so_a_rebuild_invalidates_knowledge(
    app_file, tmp_path
):
    first = _provider(tmp_path, app_file)._compatibility()
    app_file.write_bytes(b"a different build")
    second = _provider(tmp_path, app_file)._compatibility()

    assert first.app_artifact_sha256 != second.app_artifact_sha256
    assert not first.is_compatible_with(second)


def test_an_app_that_cannot_be_fingerprinted_is_refused(tmp_path):
    provider = _provider(tmp_path, tmp_path / "missing.apk")

    with pytest.raises(ValueError, match="cannot fingerprint"):
        provider._compatibility()


def test_a_device_naming_no_app_is_refused(app_file, tmp_path):
    nameless = DiscoveryDevice()
    nameless.app_id = None

    with pytest.raises(ValueError, match="identifier of the app"):
        _provider(tmp_path, app_file, nameless)._compatibility()


def test_the_recorded_versions_say_how_this_run_explored_and_named_controls(
    app_file, tmp_path
):
    """Knowledge from a later policy or control vocabulary is revalidated."""
    compatibility = _provider(tmp_path, app_file)._compatibility()

    assert compatibility.discovery_policy_version == DISCOVERY_POLICY_VERSION
    assert compatibility.control_identity_version == CONTROL_IDENTITY_VERSION


def test_having_prepared_nothing_is_recorded_as_its_own_contract():
    """So a run that seeds an account cannot reuse knowledge from one that did not."""
    assert hashlib.sha256(
        b"yewo-v0-no-setup-contract"
    ).hexdigest() == NO_SETUP_CONTRACT_SHA256


def test_a_tablet_and_a_phone_are_different_viewport_classes():
    assert classify_viewport((1080, 2400), "portrait") == "phone-portrait"
    assert classify_viewport((1600, 2560), "portrait") == "tablet-portrait"
    assert classify_viewport((2400, 1080), "landscape") == "phone-landscape"


def test_a_screen_of_no_measured_size_is_not_called_a_tablet():
    assert classify_viewport((0, 0), "portrait") == "phone-portrait"


def test_every_v0_budget_is_a_declared_ceiling():
    """A dimension left at zero would end the run before it began."""
    assert V0_DISCOVERY_BUDGETS.observations > 0
    assert V0_DISCOVERY_BUDGETS.actions > 0
    assert V0_DISCOVERY_BUDGETS.resets > 0
    assert (
        V0_DISCOVERY_BUDGETS.model_proposals
        == MODEL_CALLS_PER_PROPOSAL * V0_DISCOVERY_BUDGETS.actions
    )


def test_discovery_runs_with_the_journey_goal_and_the_v0_budgets(app_file, tmp_path):
    """The whole chain, with the controller's work stubbed at its own boundary."""
    provider = _provider(tmp_path, app_file)
    seen = {}

    class _Knowledge:
        def __init__(self, store, compatibility, run_factory, runtime, proposer):
            seen["run"] = run_factory()
            seen["runtime"] = runtime
            seen["proposer"] = proposer

        def discover(self, journey):
            seen["journey"] = journey
            return "compiled-knowledge"

    with patch(
        "yewo.testing.discovery_wiring.DiscoveryStartupKnowledgeProvider", _Knowledge
    ):
        result = provider.discover(JOURNEY)

    assert result == "compiled-knowledge"
    assert seen["journey"] is JOURNEY
    assert seen["run"].budgets == V0_DISCOVERY_BUDGETS
    assert [goal.statement for goal in seen["run"].goals] == ["A user places an order."]
    assert seen["runtime"].goal_judge is seen["proposer"]


def test_the_discovery_lane_reports_its_spend_to_the_run_as_well(app_file, tmp_path):
    """Discovery is a separate lane. Its cost still reaches the run's record."""
    recorded = []
    provider = LiveDiscoveryProvider(
        DiscoveryDevice(),
        SimpleNamespace(current_frame=None),
        model="openrouter/m",
        perception_model="yewo-ui@v1",
        app_path=str(app_file),
        platform="android",
        setup_contract_sha256=NO_SETUP_CONTRACT_SHA256,
        knowledge_root=tmp_path / "knowledge",
        scrubber=SecretScrubber(),
        usage_callback=recorded.append,
    )
    captured = {}

    class _Knowledge:
        def __init__(self, store, compatibility, run_factory, runtime, proposer):
            captured["proposer"] = proposer

        def discover(self, journey):
            return None

    with patch(
        "yewo.testing.discovery_wiring.DiscoveryStartupKnowledgeProvider", _Knowledge
    ):
        provider.discover(JOURNEY)

    call = SimpleNamespace(cost=0.002)
    captured["proposer"].llm.usage_callback(call)

    assert recorded == [call]
    assert captured["proposer"].spend.drain() == (1, 2000)


def test_startup_installs_the_boundary_only_once_a_device_exists():
    """The runner is built before a device. Discovery cannot be wired until one is."""
    from yewo.testing.runner_startup import build_runner_startup_authority

    built = []

    def factory(
        device_control,
        perception,
        llm_client,
        scrubber,
        artifact_writer,
        diagnostic_dir,
    ):
        built.append(
            (
                device_control,
                perception,
                llm_client,
                scrubber,
                artifact_writer,
                diagnostic_dir,
            )
        )
        return SimpleNamespace(discover=Mock(return_value=None))

    authority = build_runner_startup_authority(
        lambda _journey: None,
        None,
        "default",
        False,
        Path("/knowledge"),
        None,
        None,
        None,
        factory,
    )

    assert authority.knowledge_source.discovery_provider is None

    authority.attach_discovery("device", "perception", "llm", "scrubber")

    assert built == [("device", "perception", "llm", "scrubber", None, None)]
    assert authority.knowledge_source.discovery_provider is not None


def test_a_run_given_no_factory_asks_for_no_discovery():
    from yewo.testing.runner_startup import build_runner_startup_authority

    authority = build_runner_startup_authority(
        lambda _journey: None, None, "default", False, None, None, None, None
    )

    authority.attach_discovery("device", "perception", "llm", "scrubber")

    assert authority.knowledge_source.discovery_provider is None


def test_a_cold_run_is_given_a_factory_and_a_warm_one_is_not(tmp_path):
    """Discovery writes its own artifacts, so it needs a knowledge root of its own."""
    from yewo.testing.wiring import _discovery_factory

    assert (
        _discovery_factory(
            None,
            app_path="a.apk",
            platform="android",
            model="m",
            perception_model="yewo-ui@v1",
        )
        is None
    )
    assert (
        _discovery_factory(
            tmp_path / "knowledge",
            app_path="a.apk",
            platform="android",
            model="m",
            perception_model="yewo-ui@v1",
        )
        is not None
    )


def test_a_map_of_the_same_app_on_another_screen_shape_is_known_not_reused(tmp_path):
    """One app, 100 devices, one app map. Its mechanics still do not transfer."""
    from yewo.discovery.artifacts import (
        ArtifactProvenance,
        ArtifactRedaction,
        DiscoveryArtifactHeader,
        DiscoveryCompatibilityKey,
        PerceptionArtifactIdentity,
    )

    def key(viewport, orientation="portrait"):
        return DiscoveryCompatibilityKey(
            app_id="com.example.app",
            app_artifact_sha256="a" * 64,
            platform="android",
            viewport_class=viewport,
            orientation=orientation,
            setup_contract_sha256="b" * 64,
            discovery_policy_version="discovery-policy-v0",
            screen_identity_version="screen-state-v1",
            control_identity_version="control-v1",
            perception_artifacts=(
                PerceptionArtifactIdentity(name="yolo-ui", version="1"),
            ),
        )

    phone = key("phone-portrait")
    tablet = key("tablet-portrait")

    assert phone.describes_same_app_as(tablet)
    assert not phone.is_compatible_with(tablet)
    assert phone.incompatibilities(tablet) == ("viewport_class",)
    assert phone.app_incompatibilities(tablet) == ()
    del ArtifactProvenance, ArtifactRedaction, DiscoveryArtifactHeader, tmp_path


def test_a_different_build_is_a_different_app_whatever_the_screen():
    from yewo.discovery.artifacts import (
        DiscoveryCompatibilityKey,
        PerceptionArtifactIdentity,
    )

    def key(digest):
        return DiscoveryCompatibilityKey(
            app_id="com.example.app",
            app_artifact_sha256=digest,
            platform="android",
            viewport_class="phone-portrait",
            orientation="portrait",
            setup_contract_sha256="b" * 64,
            discovery_policy_version="discovery-policy-v0",
            screen_identity_version="screen-state-v1",
            control_identity_version="control-v1",
            perception_artifacts=(
                PerceptionArtifactIdentity(name="yolo-ui", version="1"),
            ),
        )

    assert not key("a" * 64).describes_same_app_as(key("c" * 64))


def test_a_session_that_never_started_cannot_report_its_orientation(app_file, tmp_path):
    """`DeviceControl.driver` is None until the session starts."""
    unstarted = DiscoveryDevice()
    unstarted.driver = None

    with pytest.raises(ValueError, match="started device session"):
        _provider(tmp_path, app_file, unstarted)._compatibility()
