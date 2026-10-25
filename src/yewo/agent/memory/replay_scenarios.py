"""One scenario's replay path, from the run that wants it to the file on disk.

The run loop knows a device and a goal. The store knows files. Neither knows
what makes a path valid for a scenario, and this does. It decides the
identity a path is written under, when it may be read, and when it is worth
keeping.

A caller that builds none of this gets no durable replay. That is how the
setting turns the feature off.
"""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

from yewo.agent.memory import replay_files, replay_store, replay_walk
from yewo.agent.memory.replay_store import TraceIdentity, ValueNames
from yewo.agent.memory.replay_learning import DurableReplay
from yewo.agent.run.postconditions import ValueVerificationPolicy
from yewo.agent.grounding import GROUNDING_MARKS

logger = logging.getLogger(__name__)

PASSED = "PASSED"
"""The only verdict a path is kept from.

A path to a failure is a path to a result already known. Walking it again
spends a run arriving where the last one arrived.
"""


class TraceStore:
    """Reads and writes the paths of one journey's scenarios.

    Built per run by whoever knows the journey, because a path is only
    valid for the values and the target it was learned on.
    """

    def __init__(
        self,
        root: Path,
        values: Dict[str, str],
        platform: str,
        target_profile: str,
        detector: str,
        grounding: str,
        key: bytes,
        verification_policies: Optional[Dict[str, ValueVerificationPolicy]] = None,
        artifact: Optional[str] = None,
        allow_path_creation: bool = True,
        approve_path_update: Optional[Callable[[Path], bool]] = None,
    ):
        self._root = root
        self._values = dict(values or {})
        self._names = ValueNames(self._values)
        self._key = key
        self._verification_policies = dict(verification_policies or {})
        self._platform = platform
        self._target_profile = target_profile
        self._detector = detector
        self._grounding = grounding
        self._artifact = artifact
        self._allow_path_creation = allow_path_creation
        self._approve_path_update = approve_path_update or (lambda _path: False)

    def resolve_identity(self, scenario) -> Optional[TraceIdentity]:
        """What this scenario's path would be written under.

        None where the scenario has no uuid to own a file by, because two
        scenarios sharing a path would walk each other's.
        """
        if not scenario.scenario_uuid:
            return None
        return TraceIdentity(
            scenario_uuid=scenario.scenario_uuid,
            scenario_digest=replay_store.scenario_digest(
                scenario,
                self._values,
                self._verification_policies,
            ),
            platform=self._platform,
            target_profile=self._target_profile,
            detector=self._detector,
            grounding=self._grounding,
        )

    def build_replay(self, identity: TraceIdentity) -> DurableReplay:
        """Everything the loop needs to walk this scenario's replay path."""
        trace = replay_files.read_trace(replay_files.trace_path(self._root, identity), identity)
        if trace is not None:
            logger.info(
                "A replay path of %d step(s) is available for this scenario.",
                len(trace.transitions),
            )
        replay = DurableReplay(
            trace=trace,
            values=self._values,
            names=self._names,
            key=self._key,
            verification_policies=self._verification_policies,
            current_artifact=self._artifact,
        )
        return replay

    def save_trace(self, scenario, replay: DurableReplay, status) -> None:
        """Writes the path this scenario walked, where it is worth keeping.

        Only a passing scenario leaves one. A run that replayed part of a
        path and planned the rest retains every successful segment. What it
        writes replaces the file rather than patching the middle of it.

        That is also how a path the app outgrew is replaced. The walk
        refuses it, the planner finds the new route, and the new route is
        what the next run reads.
        """
        if str(status) != PASSED or not replay.is_worth_saving:
            return
        identity = self.resolve_identity(scenario)
        if identity is None:
            return

        steps, screens = replay.select_learned_steps()
        if not screens:
            logger.info("Replay path was not saved: no completed transition was observed.")
            return
        trace = replay_walk.build_trace(
            identity=identity,
            steps=steps[: len(screens)],
            screens=screens,
            names=self._names,
            artifact=self._artifact,
            key=self._key,
            arrival_ms=replay.selected_arrival_ms,
            target_descriptions=replay.selected_target_descriptions,
            screen_descriptions=replay.selected_source_descriptions,
            destination_descriptions=replay.selected_destination_descriptions,
            segment_starts=replay.selected_segment_starts,
            verification_policies=replay.verification_policies,
        )
        if trace is None:
            return

        path = replay_files.trace_path(self._root, identity)
        known = replay_files.read_trace(path, identity)
        if (
            known is not None
            and known.identity == trace.identity
            and known.transitions == trace.transitions
        ):
            return
        if known is not None and len(trace.transitions) < len(known.transitions):
            # A run whose learning stopped early saw a prefix, not a path.
            # Writing it would trade a longer route that worked for a
            # shorter one, and every run after this would replay less.
            logger.info(
                "Keeping the saved path of %d step(s): this run learned %d.",
                len(known.transitions),
                len(trace.transitions),
            )
            return
        if path.exists():
            if not self._approve_path_update(path):
                logger.info("Keeping the existing replay path until its update is approved.")
                return
        elif not self._allow_path_creation:
            logger.info("Replay path was not created: this run is read-only.")
            return
        if replay_files.write_trace(path, trace):
            logger.info("Saved a path of %d step(s) for the next run.", len(trace.transitions))


def build_trace_store(
    root: Path,
    values: Dict[str, str],
    environment,
    grounding: str,
    key: Optional[bytes] = None,
    verification_policies: Optional[Dict[str, ValueVerificationPolicy]] = None,
    allow_path_creation: bool = True,
    approve_path_update: Optional[Callable[[Path], bool]] = None,
) -> Optional[TraceStore]:
    """The store for this run, or None where a path could not be valid.

    A target nothing identified cannot own a path. Guessing one would let
    two different devices read each other's.
    """
    if environment is None or not environment.platform:
        return None
    device = environment.device
    profile = device.model or device.serial or ""
    if not profile:
        return None
    return TraceStore(
        root=root,
        values=values,
        platform=environment.platform,
        target_profile=profile,
        detector=(environment.detector or "") if grounding == GROUNDING_MARKS else "",
        grounding=grounding,
        key=key if key is not None else replay_store.digest_key(),
        verification_policies=verification_policies,
        artifact=environment.app.artifact_sha256,
        allow_path_creation=allow_path_creation,
        approve_path_update=approve_path_update,
    )


__all__: List[str] = ["TraceStore", "build_trace_store"]
