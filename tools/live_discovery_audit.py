#!/usr/bin/env python3
"""Prove bounded discovery over the cold trainer navigation flow."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from yewo.agent.actuation.actions import Coordinates, PlannerTapAction
from yewo.agent.actuation.safety import (
    ActionAuthorization,
    execute_authorized,
)
from yewo.agent.device.platform import Platform
from yewo.agent.device.session import DeviceSession
from yewo.agent.memory.control_identity import (
    build_control_identity,
    control_identity_id,
)
from yewo.agent.memory.replay_store import RelativeBounds
from yewo.agent.perception import settle
from yewo.agent.run.settlement import SettlementController, SettlementReceipt
from yewo.discovery.artifacts import (
    ArtifactProvenance,
    ArtifactRedaction,
    DiscoveryArtifactHeader,
    DiscoveryCompatibilityKey,
    PerceptionArtifactIdentity,
)
from yewo.discovery.contracts import (
    CapabilityQuestion,
    DiscoveryBudgets,
    DiscoveryObservation,
    DiscoveryRun,
    ObservedControl,
    SideEffectClass,
)
from yewo.discovery.control_semantics import semantic_control_at
from yewo.discovery.coordinator import DiscoveryCoordinator
from yewo.discovery.runtime import (
    CapabilityEvidenceReceipt,
    DiscoveryActionCandidate,
    DiscoveryDispatchReceipt,
    DiscoveryExecutionReceipt,
    DiscoveryObservationBatch,
    DiscoveryProposalReceipt,
    DiscoveryResetReceipt,
    DiscoverySetupReceipt,
)
from yewo.discovery.store import DiscoveryStore
from yewo.identity import derived_uuid, new_uuid


@dataclass(frozen=True)
class AuditStage:
    name: str
    page_source_fragments: tuple[str, ...]
    next_tap: tuple[int, int] | None


STAGES = (
    AuditStage("schedule-list", ("Trainings Schedule",), (72, 202)),
    AuditStage("drawer", ("Edit Schedule",), (154, 671)),
    AuditStage("schedule-editor", ("Schedule Editor",), (962, 2219)),
    AuditStage("schedule-entry-form", ("Add Trainigns Schedule", "Name"), None),
)


class FirstAuthorizedCandidate:
    """Deterministic audit chooser; production may put a model behind this boundary."""

    model_calls_per_proposal = 0
    maximum_cost_micro_usd = 0

    def select(self, run, candidates):
        del run
        if not candidates:
            raise ValueError("The audit expected one authorized discovery candidate.")
        return DiscoveryProposalReceipt(
            selected_frontier_id=candidates[0].frontier_id,
            model_calls=0,
            cost_micro_usd=0,
        )


class TrainerAuditRuntime:
    """App-specific acceptance fixture beneath the app-agnostic controller."""

    def __init__(self, session: DeviceSession, setup_sha: str, question_ids: tuple[str, ...]):
        self.session = session
        self.device = session.device_control
        if self.device is None:
            raise RuntimeError("The device session has not started.")
        self.setup_sha = setup_sha
        self.question_ids = question_ids
        self.stage_index = 0
        self.capture_index = 0

    def prepare(self, setup_contract_sha256):
        return DiscoverySetupReceipt(
            success=setup_contract_sha256 == self.setup_sha,
            setup_contract_sha256=self.setup_sha,
        )

    def observe(self):
        return self._capture(include_capability=True)

    def execute(self, authorization: ActionAuthorization, candidate):
        stage = STAGES[self.stage_index]
        if stage.next_tap is None:
            raise RuntimeError("The terminal audit stage has no action.")
        settlement = SettlementController()
        settlement.begin_action(
            authorization.control_identity,
            candidate.source_phash,
            candidate.source_observation_id,
        )
        execute_authorized(authorization, self.device)
        destination_phash = settle.settle(self.device, 4.0)
        self.stage_index += 1
        destination = self._capture(include_capability=False)
        receipt = settlement.record_dispatch(
            authorization.control_identity,
            candidate.source_phash,
            destination_phash,
            failed=False,
        )
        if receipt is None:
            settlement.observe(
                destination.observation.phash, f"{destination.observation.observation_id}-stable-1"
            )
            receipt = settlement.observe(
                destination.observation.phash,
                destination.observation.observation_id,
            )
        if not isinstance(receipt, SettlementReceipt):
            raise RuntimeError("The audit action did not produce a settlement receipt.")
        transition_identity = derived_uuid(
            "discovery-transition-v1",
            candidate.source_observation_id,
            candidate.control_identity,
            destination.observation.observation_id,
        )
        return DiscoveryExecutionReceipt(
            dispatch=DiscoveryDispatchReceipt(
                dispatch_id=new_uuid(),
                authorization_id=authorization.authorization_id,
                success=True,
            ),
            settlement=receipt,
            destination=destination,
            capability_evidence=(
                CapabilityEvidenceReceipt(
                    question_id=self.question_ids[self.stage_index],
                    observation_id=destination.observation.observation_id,
                    kind="route_verified",
                    predicate_id=f"trainer-audit-{STAGES[self.stage_index].name}-v1",
                    transition_identity=transition_identity,
                ),
            ),
        )

    def reset(self):
        return DiscoveryResetReceipt(
            success=False,
            reason="The linear acceptance path does not request reset.",
        )

    def _capture(self, *, include_capability: bool) -> DiscoveryObservationBatch:
        stage = STAGES[self.stage_index]
        png = self.device.get_screenshot()
        phash = settle.quick_phash(self.device)
        if phash is None:
            raise RuntimeError("The audit could not fingerprint the current screen.")
        page_source = self.device.driver.page_source
        self.capture_index += 1
        evidence_stem = f"discovery-{self.capture_index:02d}-{stage.name}"
        (self.session.screenshots_dir / f"{evidence_stem}.png").write_bytes(png)
        (self.session.screenshots_dir / f"{evidence_stem}.xml").write_text(
            page_source,
            encoding="utf-8",
        )
        if not any(fragment in page_source for fragment in stage.page_source_fragments):
            expected = ", ".join(stage.page_source_fragments)
            raise RuntimeError(f"The {stage.name} local predicate found none of: {expected}.")
        observation_id = new_uuid()
        screen_state_id = derived_uuid(
            "trainer-phase7-audit-state-v1",
            stage.name,
            phash,
        )
        screen_size = (self.device.screen_width, self.device.screen_height)
        candidates = ()
        controls = ()
        if stage.next_tap is not None:
            semantic_control = semantic_control_at(page_source, stage.next_tap)
            if semantic_control is None:
                raise RuntimeError(f"The {stage.name} audit target has no live semantic identity.")
            action = PlannerTapAction(
                coordinates=Coordinates(x=stage.next_tap[0], y=stage.next_tap[1]),
                target=f"locally verified {stage.name} navigation control",
                reason="advance the controlled Phase 7 discovery audit",
            )
            identity = control_identity_id(build_control_identity(action, phash, screen_size))
            controls = (
                ObservedControl(
                    control_identity=identity,
                    role=semantic_control.role,
                    text_digest=semantic_control.text_digest,
                    bounds=self._relative_bounds(
                        semantic_control.bounds,
                        screen_size,
                    ),
                    actionable=True,
                ),
            )
            candidates = (
                DiscoveryActionCandidate(
                    source_observation_id=observation_id,
                    source_screen_state_id=screen_state_id,
                    control_identity=identity,
                    relevant_question_ids=(self.question_ids[self.stage_index + 1],),
                    action=action,
                    side_effect=SideEffectClass.READ_ONLY,
                    reversible=True,
                    source_phash=phash,
                    screen_size=screen_size,
                ),
            )
        observation = DiscoveryObservation(
            observation_id=observation_id,
            screen_state_id=screen_state_id,
            phash=phash,
            controls=controls,
            foreground_app_id=self.device.app_id,
        )
        evidence = (
            (
                CapabilityEvidenceReceipt(
                    question_id=self.question_ids[self.stage_index],
                    observation_id=observation_id,
                    kind="observed",
                    predicate_id=f"trainer-audit-{stage.name}-v1",
                ),
            )
            if include_capability
            else ()
        )
        return DiscoveryObservationBatch(
            observation=observation,
            candidates=candidates,
            capability_evidence=evidence,
            state_depth=self.stage_index,
        )

    @staticmethod
    def _relative_bounds(
        bounds: tuple[int, int, int, int],
        screen_size: tuple[int, int],
    ) -> RelativeBounds:
        width, height = screen_size
        return RelativeBounds(
            left=round(bounds[0] * 1000 / width),
            top=round(bounds[1] * 1000 / height),
            right=round(bounds[2] * 1000 / width),
            bottom=round(bounds[3] * 1000 / height),
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="Pixel_8")
    parser.add_argument(
        "--app",
        type=Path,
        default=Path("eval/apps/android/trainer-app-1645-buggy.apk"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/runs/phase7-discovery-audit"),
    )
    args = parser.parse_args()
    app_path = args.app.resolve()
    setup_sha = hashlib.sha256(b"trainer-clean-install-v1").hexdigest()
    compatibility = DiscoveryCompatibilityKey(
        app_id="trainer-app",
        app_artifact_sha256=_sha256(app_path),
        platform="android",
        viewport_class="phone-portrait",
        orientation="portrait",
        setup_contract_sha256=setup_sha,
        discovery_policy_version="discovery-policy-v1",
        screen_identity_version="screen-state-v1",
        control_identity_version="control-v1",
        perception_artifacts=(PerceptionArtifactIdentity(name="appium-page-source", version="1"),),
    )
    questions = tuple(
        CapabilityQuestion(intent=f"observe trainer {stage.name}") for stage in STAGES
    )
    run = DiscoveryRun(
        header=DiscoveryArtifactHeader(
            schema_name="discovery-run",
            compatibility=compatibility,
            provenance=ArtifactProvenance(source_run_ids=("phase7-live-audit",)),
            redaction=ArtifactRedaction(policy_version="redaction-v1"),
        ),
        questions=questions,
        budgets=DiscoveryBudgets(
            observations=4,
            actions=3,
            model_proposals=0,
            resets=0,
            duration_ms=120_000,
            state_depth=3,
            cost_micro_usd=0,
            side_effect_actions=0,
            attempts_per_transition=1,
        ),
    )
    session = DeviceSession(
        device_name=args.device,
        app_path=str(app_path),
        platform=Platform.ANDROID,
        session_dir=args.output,
        keep_device=False,
        keep_app=True,
    )
    try:
        session.start()
        runtime = TrainerAuditRuntime(
            session,
            setup_sha,
            tuple(question.question_id for question in questions),
        )
        result = DiscoveryCoordinator(
            DiscoveryStore(session.layout.root / "knowledge")
        ).ensure_knowledge(
            compatibility,
            lambda: run,
            runtime,
            FirstAuthorizedCandidate(),
        )
        if result.discovery is None:
            raise RuntimeError("The cold audit unexpectedly reused an AppMap.")
        manifest = {
            "status": result.discovery.run.terminal.terminal.value,
            "artifact_id": result.discovery.run.header.artifact_id,
            "observed_states": [
                stage.name for stage in STAGES[: len(result.discovery.run.observations)]
            ],
            "observation_count": len(result.discovery.run.observations),
            "transition_count": len(result.discovery.run.transitions),
            "model_calls": result.discovery.run.usage.model_proposals,
            "scenario_verdict_emitted": False,
        }
        session.manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        if manifest["status"] != "discovery_complete":
            raise RuntimeError("The cold trainer discovery did not satisfy the Phase 7 exit gate.")
        print(session.layout.root)
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
