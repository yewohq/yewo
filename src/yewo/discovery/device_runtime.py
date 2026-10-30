"""Device mechanics beneath the deterministic discovery controller.

The controller decides. This observes, acts, and returns home. It holds no
budget, no terminal, and no policy. It reports what one screen offers and what
one dispatched action settled on. The controller accepts or refuses that.

It owns no device lifecycle. The session, the app, and the perception engine
are handed to it already started, and it never creates or tears one down.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from yewo.agent.actuation.action_models import (
    AppStateCapability,
    Coordinates,
    PlannerTapAction,
)
from yewo.agent.actuation.safety import ActionAuthorization, execute_authorized
from yewo.agent.device import resilience
from yewo.agent.device.driver_commands import DriverCommands
from yewo.discovery.control_identity import discovery_control_identity
from yewo.agent.memory.replay_store import BOUNDS_SCALE, RelativeBounds
from yewo.agent.memory.screen_identity import is_same_screen
from yewo.agent.perception import settle
from yewo.agent.perception.elements import element_roles
from yewo.agent.perception.fusion import PerceptionFrame
from yewo.agent.device.platform import Platform
from yewo.agent.run.settlement import SettlementController, SettlementReceipt
from yewo.agent.step import observation as step_observation
from yewo.discovery.contracts import (
    CapabilityCandidate,
    DiscoveryObservation,
    JourneyGoal,
    ObservedControl,
    SideEffectClass,
)
from yewo.discovery.artifacts import ArtifactReference
from yewo.discovery.control_surface import (
    KEYBOARD_ROLES,
    SYSTEM_CHROME_ROLES,
    ControlSurface,
    build_control_surface,
)
from yewo.discovery.runtime import (
    CapabilityEvidenceReceipt,
    DiscoveryActionCandidate,
    DiscoveryDispatchReceipt,
    DiscoveryExecutionReceipt,
    DiscoveryObservationBatch,
    DiscoveryResetReceipt,
    DiscoverySetupReceipt,
    transition_identity_for,
)
from yewo.discovery.system_chrome import (
    android_system_chrome,
    ios_system_chrome,
    is_inside_system_chrome,
)
from yewo.identity import derived_uuid, new_uuid
from yewo.utils.secrets import SecretScrubber

SCREEN_IDENTITY_VERSION = "discovery-screen-state-v1"
"""Namespace for a screen's derived identity.

Change it when the derivation changes. The compatibility key carries a
matching `screen_identity_version`, so stored knowledge is revalidated rather
than silently reinterpreted.
"""

FOREGROUND_APP_STATE = 4
"""The platform's own code for an app running in the foreground."""

_NOT_THE_APPS_OWN = SYSTEM_CHROME_ROLES | KEYBOARD_ROLES
"""What a screen shows that the application did not put there.

Reading matter is deliberately absent: a screen's heading is reading matter.
"""

REACHED_SCREEN_VERSION = "discovery-reached-screen-v1"
"""Namespace for the identity of a screen this run reached.

Derived from the screen and its heading, so one screen reached twice is
recorded once.
"""

EXPLORATION_REASON = "Explore one offered control toward the journey goal."
"""Why every discovery tap is dispatched.

One fixed sentence. A per-control reason would carry app vocabulary into an
artifact that must stay application-independent.
"""


class GoalJudge(Protocol):
    """Decides which journey goals a reached screen satisfies.

    It reads the frame because a goal is stated in prose and a screen answers
    it in prose. The control surface deliberately keeps no label text, so it
    cannot serve this. It is told where the platform's own regions are, so the
    clock does not become part of the question.

    A judgement is not route evidence. The controller still checks the
    observation and the transition the claim names.
    """

    def satisfied_goal_ids(
        self,
        goals: tuple[JourneyGoal, ...],
        frame: PerceptionFrame,
        chrome: tuple[tuple[int, int, int, int], ...],
    ) -> tuple[str, ...]: ...


class DiscoveryArtifactWriter(Protocol):
    def save_step(self, session_dir: Path, step: int, snapshot) -> dict[str, str]: ...


@dataclass(frozen=True)
class _CapturedDiscoveryScreen:
    """One capture, kept whole because the judge needs what the batch drops."""

    batch: DiscoveryObservationBatch
    frame: PerceptionFrame
    chrome: tuple[tuple[int, int, int, int], ...] = ()


class DeviceDiscoveryRuntime:
    """One discovery run's device mechanics, over one app on one device."""

    def __init__(
        self,
        device_control,
        perception,
        *,
        app_id: str,
        setup_contract_sha256: str,
        goals: tuple[JourneyGoal, ...],
        goal_judge: GoalJudge,
        scrubber: SecretScrubber,
        artifact_writer: DiscoveryArtifactWriter | None = None,
        diagnostic_dir: Path | None = None,
        settle_time: float = 4.0,
        sleep: Callable[[float], None],
    ):
        if device_control.driver is None:
            raise ValueError("Discovery requires a started device session.")
        self.device_control = device_control
        self.perception = perception
        self.commands = DriverCommands(device_control.driver)
        self.app_id = app_id
        self.setup_contract_sha256 = setup_contract_sha256
        self.goals = goals
        self.goal_judge = goal_judge
        self.scrubber = scrubber
        self.artifact_writer = artifact_writer
        self.diagnostic_dir = diagnostic_dir
        self.settle_time = settle_time
        self.sleep = sleep
        self._screens: list[tuple[str, str]] = []
        self._state_depth = 0
        self._observation_count = 0
        self._chrome_bounds_by_screen_size: dict[
            tuple[int, int], tuple[tuple[int, int, int, int], ...]
        ] = {}

    def prepare(self, setup_contract_sha256: str) -> DiscoverySetupReceipt:
        """Refuse a run whose setup is not the one this runtime was built for."""
        if setup_contract_sha256 != self.setup_contract_sha256:
            return DiscoverySetupReceipt(
                success=False,
                setup_contract_sha256=self.setup_contract_sha256,
                reason="The requested setup contract is not the prepared one.",
            )
        state = self.commands.app_state(self.app_id)
        if state != FOREGROUND_APP_STATE:
            return DiscoverySetupReceipt(
                success=False,
                setup_contract_sha256=self.setup_contract_sha256,
                reason=f"The app is not in the foreground. Platform state: {state}.",
            )
        return DiscoverySetupReceipt(
            success=True,
            setup_contract_sha256=self.setup_contract_sha256,
        )

    def observe(self) -> DiscoveryObservationBatch:
        """What this screen is, and every control it offers."""
        return self._read().batch

    def execute(
        self,
        authorization: ActionAuthorization,
        candidate: DiscoveryActionCandidate,
    ) -> DiscoveryExecutionReceipt:
        """Dispatch one authorized action and report where it settled."""
        settlement = SettlementController()
        settlement.begin_action(
            authorization.control_identity,
            candidate.source_phash,
            candidate.source_observation_id,
        )
        execute_authorized(authorization, self.device_control)
        settle.settle(self.device_control, self.settle_time)
        self._state_depth += 1
        destination = self._read()
        receipt = self._settled(settlement, authorization, candidate, destination.batch.observation)
        return DiscoveryExecutionReceipt(
            dispatch=DiscoveryDispatchReceipt(
                dispatch_id=new_uuid(),
                authorization_id=authorization.authorization_id,
                success=True,
            ),
            settlement=receipt,
            destination=destination.batch,
            capability_evidence=self._judged(candidate, destination),
        )

    def reset(self) -> DiscoveryResetReceipt:
        """Return to the app's start screen, which is how a dead end is left.

        A relaunch rather than a back action. Back navigates, dismisses an
        overlay, or leaves the app depending on where it is pressed. The number
        of presses needed from an unknown screen is unknown. This is one action
        with a known destination.
        """
        try:
            resilience.relaunch(
                "discovery reset",
                AppStateCapability(action="relaunch"),
                self.app_id,
                self.commands,
                self.sleep,
            )
        except Exception as error:  # noqa: BLE001 - the device is an external boundary
            return DiscoveryResetReceipt(success=False, reason=str(error))
        self._state_depth = 0
        return DiscoveryResetReceipt(success=True, observation=self._read().batch)

    def _read(self) -> _CapturedDiscoveryScreen:
        snapshot = step_observation.capture(self.perception, None, None)
        if snapshot.current_phash is None or snapshot.scan_info is None:
            raise RuntimeError("Discovery requires a fingerprinted observation.")
        if snapshot.perception_frame is None:
            raise RuntimeError("Discovery requires a fused perception frame.")
        screen_size = (
            snapshot.scan_info.screen_width,
            snapshot.scan_info.screen_height,
        )
        chrome = self._system_chrome_bounds(screen_size)
        surface = build_control_surface(
            snapshot.perception_frame,
            screen_size,
            chrome,
        )
        screen_state_id = self._screen_state_id(snapshot.current_phash)
        controls, candidates = self._project(
            surface,
            observation_id=snapshot.observation_id,
            screen_state_id=screen_state_id,
            phash=snapshot.current_phash,
            screen_size=screen_size,
        )
        observation = DiscoveryObservation(
            observation_id=snapshot.observation_id,
            screen_state_id=screen_state_id,
            phash=snapshot.current_phash,
            controls=controls,
            keyboard_visible=self.device_control.is_keyboard_open(),
            foreground_app_id=self.app_id,
            evidence=self._record_diagnostics(snapshot),
        )
        return _CapturedDiscoveryScreen(
            batch=DiscoveryObservationBatch(
                observation=observation,
                candidates=candidates,
                capability_candidates=self._reached(
                    observation, snapshot.perception_frame, chrome
                ),
                state_depth=self._state_depth,
            ),
            frame=snapshot.perception_frame,
            chrome=chrome,
        )

    def _record_diagnostics(self, snapshot) -> tuple[ArtifactReference, ...]:
        if self.artifact_writer is None or self.diagnostic_dir is None:
            return ()
        written = self.artifact_writer.save_step(
            self.diagnostic_dir,
            self._observation_count,
            snapshot,
        )
        self._observation_count += 1
        references = []
        for artifact_kind, filename in sorted(written.items()):
            path = self.diagnostic_dir / filename
            references.append(
                ArtifactReference(
                    artifact_id=derived_uuid(
                        "discovery-observation-artifact-v1",
                        snapshot.observation_id,
                        artifact_kind,
                    ),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    path=f"screenshots/{filename}",
                )
            )
        return tuple(references)

    def _system_chrome_bounds(
        self, screen_size: tuple[int, int]
    ) -> tuple[tuple[int, int, int, int], ...]:
        cached = self._chrome_bounds_by_screen_size.get(screen_size)
        if cached is not None:
            return cached
        if self.device_control.platform is Platform.ANDROID:
            regions = android_system_chrome(
                self.commands.android_system_bars(), screen_size
            )
        else:
            regions = ios_system_chrome(
                self.commands.ios_viewport_rect(), screen_size
            )
        self._chrome_bounds_by_screen_size[screen_size] = regions
        return regions

    def _reached(
        self,
        observation: DiscoveryObservation,
        frame: PerceptionFrame,
        chrome: tuple[tuple[int, int, int, int], ...],
    ) -> tuple[CapabilityCandidate, ...]:
        """What this screen offered, named from what the screen itself reads as.

        A run that misses its goal still reached screens, and this is what says
        so. The name is the screen's own heading, which is observed rather than
        predicted. These carry no requirement authority: a reachable screen is
        not a product rule that it should be reachable.
        """
        heading = _heading(frame, self.scrubber, chrome)
        open_goals = tuple(goal.goal_id for goal in self.goals if not goal.is_terminal)
        if heading is None or not open_goals:
            return ()
        return (
            CapabilityCandidate(
                candidate_id=derived_uuid(
                    REACHED_SCREEN_VERSION,
                    observation.screen_state_id,
                    heading,
                ),
                capability=f"reach the screen headed {heading}",
                goal_ids=open_goals,
                observation_ids=(observation.observation_id,),
                confidence=1.0,
            ),
        )

    def _project(
        self,
        surface: ControlSurface,
        *,
        observation_id: str,
        screen_state_id: str,
        phash: str,
        screen_size: tuple[int, int],
    ) -> tuple[tuple[ObservedControl, ...], tuple[DiscoveryActionCandidate, ...]]:
        """One offered control per perceived control, named on the settled screen.

        Two controls sharing a centre share an identity, because an identity is
        what an action addresses. The first wins, so the frontier counts the
        target once rather than offering the same tap twice.
        """
        goal_ids = tuple(goal.goal_id for goal in self.goals)
        controls: dict[str, ObservedControl] = {}
        candidates: dict[str, DiscoveryActionCandidate] = {}
        for control in surface.controls:
            action = PlannerTapAction(
                coordinates=Coordinates(x=control.center[0], y=control.center[1]),
                reason=EXPLORATION_REASON,
            )
            identity = discovery_control_identity(
                screen_state_id, action, screen_size
            )
            if identity in controls:
                continue
            controls[identity] = ObservedControl(
                control_identity=identity,
                role=control.role,
                bounds=_relative_bounds(control.bounds, screen_size),
                text_digest=control.text_digest,
                actionable=True,
            )
            candidates[identity] = DiscoveryActionCandidate(
                source_observation_id=observation_id,
                source_screen_state_id=screen_state_id,
                control_identity=identity,
                relevant_goal_ids=goal_ids,
                action=action,
                side_effect=SideEffectClass.REVERSIBLE,
                reversible=True,
                source_phash=phash,
                screen_size=screen_size,
                reset_after=False,
            )
        return tuple(controls.values()), tuple(candidates.values())

    def _screen_state_id(self, phash: str) -> str:
        """One identity per visual screen, matched rather than recomputed.

        A transient overlay moves the fingerprint a little. Deriving straight
        from the fingerprint would make the start screen a new screen each time
        it is seen. The controller would then never recognize that it is home.
        """
        for seen_phash, state_id in self._screens:
            if is_same_screen(seen_phash, phash):
                return state_id
        state_id = derived_uuid(SCREEN_IDENTITY_VERSION, self.app_id, phash)
        self._screens.append((phash, state_id))
        return state_id

    def _settled(
        self,
        settlement: SettlementController,
        authorization: ActionAuthorization,
        candidate: DiscoveryActionCandidate,
        destination: DiscoveryObservation,
    ) -> SettlementReceipt:
        """The receipt for one dispatch, whatever it settled on.

        A screen that did not change settles as `NO_VISIBLE_CHANGE`. A screen
        that changed settles as `DURABLE_PROGRESS` on the second matching
        observation. The controller accepts only the second, so an action that
        reached nothing closes its frontier entry rather than disappearing.
        """
        receipt = settlement.record_dispatch(
            authorization.control_identity,
            candidate.source_phash,
            destination.phash,
            failed=False,
        )
        if receipt is None:
            settlement.observe(destination.phash, f"{destination.observation_id}-stable")
            receipt = settlement.observe(destination.phash, destination.observation_id)
        if isinstance(receipt, SettlementReceipt):
            return receipt
        # Unreachable through this sequence. `record_dispatch` returns nothing
        # only when the destination differs from the source, and two matching
        # observations of that destination then close the transaction. Kept as
        # a narrowing guard, because the settlement contract belongs to another
        # module and a silent absence would reach the controller as evidence.
        raise RuntimeError("A dispatched discovery action left no settlement record.")

    def _judged(
        self,
        candidate: DiscoveryActionCandidate,
        destination: _CapturedDiscoveryScreen,
    ) -> tuple[CapabilityEvidenceReceipt, ...]:
        """Ask the judge which goals this screen satisfies, and record who said so."""
        open_goals = tuple(goal for goal in self.goals if not goal.is_terminal)
        if not open_goals:
            return ()
        satisfied = self.goal_judge.satisfied_goal_ids(
            open_goals, destination.frame, destination.chrome
        )
        known = {goal.goal_id for goal in open_goals}
        observation = destination.batch.observation
        return tuple(
            CapabilityEvidenceReceipt(
                evidence_owner="model_proposal",
                goal_id=goal_id,
                observation_id=observation.observation_id,
                kind="route_verified",
                predicate_id=f"{SCREEN_IDENTITY_VERSION}:judged",
                transition_identity=transition_identity_for(
                    candidate.source_observation_id,
                    candidate.control_identity,
                    observation.observation_id,
                ),
            )
            for goal_id in satisfied
            if goal_id in known
        )


def _relative_bounds(
    bounds: tuple[int, int, int, int], screen_size: tuple[int, int]
) -> RelativeBounds:
    """Where a control sat, as thousandths, so a different viewport still reads it."""
    width, height = screen_size
    left, top, right, bottom = bounds

    def scaled(value: int, extent: int) -> int:
        if extent <= 0:
            return 0
        return max(0, min(BOUNDS_SCALE, round(value * BOUNDS_SCALE / extent)))

    return RelativeBounds(
        left=scaled(left, width),
        top=scaled(top, height),
        right=scaled(right, width),
        bottom=scaled(bottom, height),
    )


def _heading(
    frame: PerceptionFrame,
    scrubber: SecretScrubber,
    chrome: tuple[tuple[int, int, int, int], ...],
) -> str | None:
    """What a person reading this screen would call it.

    Taken from the fused elements, where perception has already attached each
    OCR line to the element it sits on. Raw OCR lines carry no role, so the
    topmost of them is the platform's clock on every Android screen.

    The platform's own regions are excluded by where they are, not only by
    what they were classified as. A detector that misses the status bar would
    otherwise name every screen after the time. That name changes every
    minute, so one screen reached twice would never be recorded once.

    Scrubbed, because the name reaches the AppMap, which is shared.
    """
    named = sorted(
        (
            element
            for element in frame.elements
            if element.text.strip()
            and not element_roles(element) & _NOT_THE_APPS_OWN
            and not is_inside_system_chrome(_element_box(element), chrome)
        ),
        key=lambda element: element.bounding_box.y_min,
    )
    for element in named:
        heading = " ".join(scrubber.scrub(element.text).split())
        if heading:
            return heading
    return None


def _element_box(element) -> tuple[float, float, float, float]:
    box = element.bounding_box
    return box.x_min, box.y_min, box.x_max, box.y_max
