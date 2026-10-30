"""Build the bounded discovery boundary once a device is live.

Discovery needs a started device and a working perception engine. The runner
is constructed before either exists. So this is handed to the startup
authority as a factory, and called when the device comes up.

Everything app-specific here is read from the device and the journey. Nothing
in this module names an application, a screen, or a control.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from uuid import uuid4

from yewo.agent.device.driver_commands import DriverCommands
from yewo.discovery.artifacts import (
    ArtifactProvenance,
    ArtifactRedaction,
    DiscoveryArtifactHeader,
    DiscoveryCompatibilityKey,
    PerceptionArtifactIdentity,
)
from yewo.discovery.contracts import DiscoveryBudgets, DiscoveryRun
from yewo.discovery.control_identity import CONTROL_IDENTITY_VERSION
from yewo.discovery.device_runtime import SCREEN_IDENTITY_VERSION, DeviceDiscoveryRuntime
from yewo.discovery.model_proposer import (
    MODEL_CALLS_PER_PROPOSAL,
    ModelDiscoveryProposer,
    ProposalSpend,
)
from yewo.discovery.store import DiscoveryStore
from yewo.identity import artifact_sha256
from yewo.llm.llm_client import LLMClient
from yewo.testing.discovery_startup import DiscoveryStartupKnowledgeProvider
from yewo.testing.generation_startup import ScenarioStartupKnowledge
from yewo.testing.journey_goals import build_journey_goals
from yewo.testing.models import UserJourney
from yewo.utils.secrets import SecretScrubber

NO_SETUP_CONTRACT_SHA256 = hashlib.sha256(b"yewo-v0-no-setup-contract").hexdigest()
"""The digest of having prepared nothing.

V0 runs no setup step. Knowledge learned without one is compatible only with
another run that also had none, which is what this records. A run that later
seeds an account carries that contract's own digest instead, and this
knowledge is revalidated rather than reused.
"""

DISCOVERY_POLICY_VERSION = "discovery-policy-v1"
"""The policy this run explored under.

V0 authorizes only reversible actions and judges a goal with a model. Both are
recorded here, so knowledge learned under a later policy is revalidated rather
than reused.
"""

V0_ACTION_LIMIT = 60

V0_DISCOVERY_BUDGETS = DiscoveryBudgets(
    observations=120,
    actions=V0_ACTION_LIMIT,
    model_proposals=V0_ACTION_LIMIT * MODEL_CALLS_PER_PROPOSAL,
    resets=10,
    duration_ms=15 * 60 * 1000,
    state_depth=8,
    cost_micro_usd=1_500_000,
    side_effect_actions=V0_ACTION_LIMIT,
    attempts_per_transition=2,
    proposal_window_size=20,
)
"""What one V0 discovery run may spend.

Sized for one journey on one app, not for exhaustive exploration. Each screen
can require one goal judgement and one control choice. Each structured call
can use two provider attempts. The model-call budget reserves all four attempts
for every allowed action.
"""

TABLET_SHORTEST_EDGE_PX = 1200
"""Shortest-edge pixels at which a viewport is treated as a tablet.

Knowledge is only reused inside one viewport class, so this decides what
counts as the same shape of screen.
"""


def classify_viewport(screen_size: tuple[int, int], orientation: str) -> str:
    """One name for the shape of screen this knowledge was learned on."""
    shortest = min(screen_size) if all(screen_size) else 0
    form = "tablet" if shortest >= TABLET_SHORTEST_EDGE_PX else "phone"
    return f"{form}-{orientation}"


class LiveDiscoveryProvider:
    """One journey's bounded discovery run, over a started device."""

    def __init__(
        self,
        device_control,
        perception,
        *,
        model: str,
        perception_model: str,
        app_path: str,
        platform: str,
        setup_contract_sha256: str,
        knowledge_root: Path,
        scrubber: SecretScrubber,
        usage_callback=None,
        prompt_store=None,
        artifact_writer=None,
        diagnostic_dir: Path | None = None,
        budgets: DiscoveryBudgets = V0_DISCOVERY_BUDGETS,
    ):
        self.device_control = device_control
        self.perception = perception
        self.model = model
        self.perception_model = perception_model
        self.app_path = app_path
        self.platform = platform
        self.setup_contract_sha256 = setup_contract_sha256
        self.knowledge_root = knowledge_root
        self.scrubber = scrubber
        self.usage_callback = usage_callback
        self.prompt_store = prompt_store
        self.artifact_writer = artifact_writer
        self.diagnostic_dir = diagnostic_dir
        self.budgets = budgets

    def discover(self, journey: UserJourney) -> ScenarioStartupKnowledge | None:
        """Explore toward this journey's goal and compile what it reached."""
        goals = build_journey_goals(journey)
        compatibility = self._compatibility()
        spend = ProposalSpend()
        proposer = ModelDiscoveryProposer(
            LLMClient(
                model=self.model,
                cache_affinity=f"discovery-{uuid4()}",
                prompt_store=self.prompt_store,
                usage_callback=_combine_usage_callbacks(
                    spend.record, self.usage_callback
                ),
            ),
            self.perception,
            self.scrubber,
            spend,
        )
        runtime = DeviceDiscoveryRuntime(
            self.device_control,
            self.perception,
            app_id=self._app_id(),
            setup_contract_sha256=self.setup_contract_sha256,
            goals=goals,
            goal_judge=proposer,
            scrubber=self.scrubber,
            artifact_writer=self.artifact_writer,
            diagnostic_dir=self.diagnostic_dir,
            sleep=time.sleep,
        )
        return DiscoveryStartupKnowledgeProvider(
            DiscoveryStore(self.knowledge_root),
            compatibility,
            lambda: _build_discovery_run(compatibility, goals, self.budgets),
            runtime,
            proposer,
        ).discover(journey)

    def _app_id(self) -> str:
        app_id = self.device_control.app_id
        if not app_id:
            raise ValueError("Discovery requires the identifier of the app under test.")
        return app_id

    def _compatibility(self) -> DiscoveryCompatibilityKey:
        """The exact context this knowledge may be interpreted in."""
        digest = artifact_sha256(Path(self.app_path))
        if digest is None:
            raise ValueError(f"Discovery cannot fingerprint the app at {self.app_path}.")
        screen_size = (
            self.device_control.screen_width,
            self.device_control.screen_height,
        )
        orientation = self._orientation()
        return DiscoveryCompatibilityKey(
            app_id=self._app_id(),
            app_artifact_sha256=digest,
            platform=self.platform,
            viewport_class=classify_viewport(screen_size, orientation),
            orientation=orientation,
            setup_contract_sha256=self.setup_contract_sha256,
            discovery_policy_version=DISCOVERY_POLICY_VERSION,
            screen_identity_version=SCREEN_IDENTITY_VERSION,
            control_identity_version=CONTROL_IDENTITY_VERSION,
            perception_artifacts=(
                PerceptionArtifactIdentity(
                    name="perception-model", version=self.perception_model
                ),
            ),
        )

    def _orientation(self) -> str:
        """What the device reports, or portrait when it reports nothing."""
        driver = self.device_control.driver
        if driver is None:
            raise ValueError("Discovery requires a started device session.")
        return DriverCommands(driver).orientation() or "portrait"


def _build_discovery_run(
    compatibility: DiscoveryCompatibilityKey,
    goals: tuple,
    budgets: DiscoveryBudgets,
) -> DiscoveryRun:
    """One open discovery run for this exact context and these goals."""
    return DiscoveryRun(
        header=DiscoveryArtifactHeader(
            schema_name="discovery-run",
            compatibility=compatibility,
            provenance=ArtifactProvenance(source_run_ids=(str(uuid4()),)),
            redaction=ArtifactRedaction(policy_version="redaction-v1"),
        ),
        goals=goals,
        budgets=budgets,
    )


def _combine_usage_callbacks(discovery_callback, run_callback):
    """Report one usage record to the discovery meter and to the run's tracker."""
    if run_callback is None:
        return discovery_callback

    def report(record) -> None:
        discovery_callback(record)
        run_callback(record)

    return report
