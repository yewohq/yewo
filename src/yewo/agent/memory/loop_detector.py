"""Bounded interaction attempts and screen-stagnation tracking.

An attempt is one resolved action aimed at one control.  Whether targeting later
goes stale or the command reaches the device is a disposition of that attempt,
not a separate kind of loop.  Keeping both dispositions in one ledger prevents
an alternating delivered/stale sequence from evading every bound.
"""

from collections import deque
from dataclasses import dataclass
import logging
from typing import Optional

from yewo.agent.actuation.actions import (
    Action,
    DoubleTapAction,
    KeyEventAction,
    PlannerTapAction,
    TapAction,
)
from yewo.agent.memory.control_identity import ControlIdentity, build_control_identity
from yewo.agent.memory.screen_identity import is_same_screen
from yewo.agent.run.settlement import (
    SettlementController,
    SettlementReceipt,
    SettlementStatus,
)

_QUARANTINE_AT = 3
_QUARANTINE_REJECTIONS_BEFORE_TERMINATION = 2
_STATE_ATTEMPT_LIMIT = 8
_STAGNANT_THRESHOLD = 7
_FOCUS_VERIFICATION_LIMIT = 3
_SEMANTIC_CYCLE_TERMINATES_AT = 2
STALE_REPEATS_BEFORE_WARNING = 3

logger = logging.getLogger(__name__)

_SECOND_ATTEMPT = (
    "This control has already failed to make durable progress. Try it at most "
    "once more, then use a different control or strategy."
)
_QUARANTINED = (
    "This control is quarantined after {count} attempts without durable progress. "
    "Do not select this location again on this screen; use a different control "
    "or strategy."
)
_QUARANTINE_REJECTED = (
    "The selected control is quarantined and was not sent to the device. Choose "
    "a different location or action."
)

_STAGNANT_MESSAGE = (
    "The screen has not changed for several consecutive steps. Check for blocking "
    "input UI, overlays, or permission dialogs. Use an action supported by the "
    "current platform to clear the obstruction before trying another control."
)

_REPEATED_STALE = (
    "The screen under this control moved between planning and acting {count} "
    "times. Every attempt was abandoned before it reached the device. Treat this "
    "control as unreliable and reach the goal another way."
)

_UNCHANGED_TAP = (
    "This visual target received {attempts} delivered taps on the same unchanged "
    "screen. No visible effect was observed. Hidden app state may have changed. "
    "Do not claim that the taps failed. Continue to the next logical action. "
    "Repeat this target only when the goal explicitly requires repeated taps."
)

_SEMANTIC_CYCLE = (
    "Execution returned to an earlier screen while the same scenario instruction "
    "remains active. The intervening route made no durable task progress. Do not "
    "repeat that route. Use a different visible control or strategy."
)


@dataclass
class _ControlAttempts:
    identity: ControlIdentity
    attempts: int = 0
    delivered: int = 0
    stale: int = 0
    dispatch_failed: int = 0
    quarantine_rejections: int = 0
    quarantined: bool = False


@dataclass
class _StateAttempts:
    phash: Optional[str]
    count: int = 0
    exhausted: bool = False


@dataclass
class _SemanticVisit:
    phash: str
    obligation: str
    observation: int


@dataclass
class _SemanticCycle:
    phash: str
    obligation: str
    count: int = 0


class RollingLoopDetector:
    """Track executed, stale, and stagnant interactions in bounded windows."""

    def __init__(self, window_size: int = 20):
        self._window_size = window_size
        self._controls: deque[_ControlAttempts] = deque(maxlen=window_size)
        self._unchanged_taps: deque[_ControlAttempts] = deque(maxlen=window_size)
        self._settlement = SettlementController()
        self._state_attempts: deque[_StateAttempts] = deque(maxlen=window_size)
        self._last_action_blocked = False
        self._termination_reason: Optional[str] = None
        self._last_page_hash: Optional[str] = None
        self._stagnant_count = 0
        self._action_repetition_terminated = False
        self._page_stagnation_terminated = False
        self._focus_verification_failures = 0
        self._focus_verification_terminated = False
        self._semantic_visits: deque[_SemanticVisit] = deque(maxlen=window_size)
        self._semantic_cycles: deque[_SemanticCycle] = deque(maxlen=window_size)
        self._semantic_observation = 0
        self._semantic_cycle_reason: Optional[str] = None
        self._screen_history: deque[str] = deque(maxlen=window_size)

    def record_screen(
        self, page_hash: Optional[str], observation_id: str = ""
    ) -> Optional[str]:
        """Record one observation and report bounded visual stagnation."""
        if page_hash is None:
            return None
        settlement_was_active = self._settlement.active_settlement_id is not None
        receipt = self._settlement.observe(page_hash, observation_id)
        self._apply_settlement(receipt)
        may_enter_navigation_history = not settlement_was_active or (
            receipt is not None
            and receipt.status is SettlementStatus.DURABLE_PROGRESS
        )
        if may_enter_navigation_history and (
            not self._screen_history
            or not is_same_screen(self._screen_history[-1], page_hash)
        ):
            self._screen_history.append(page_hash)
        if self._last_page_hash is not None and is_same_screen(
            page_hash, self._last_page_hash
        ):
            self._stagnant_count += 1
        else:
            self._stagnant_count = 0
            self._page_stagnation_terminated = False
            self._last_page_hash = page_hash

        if self._stagnant_count >= _STAGNANT_THRESHOLD:
            self._page_stagnation_terminated = True
            return None
        if self._stagnant_count > 0 and self._stagnant_count % 3 == 0:
            return _STAGNANT_MESSAGE
        return None

    def record_semantic_state(
        self, page_hash: Optional[str], obligation: Optional[str]
    ) -> Optional[str]:
        """Detect a return through other screens with one obligation unchanged."""
        if page_hash is None or not obligation:
            return None
        self._semantic_observation += 1
        prior = next(
            (
                visit
                for visit in reversed(self._semantic_visits)
                if visit.obligation == obligation
                and is_same_screen(visit.phash, page_hash)
            ),
            None,
        )
        self._semantic_visits.append(
            _SemanticVisit(page_hash, obligation, self._semantic_observation)
        )
        if prior is None or self._semantic_observation - prior.observation < 2:
            return None
        cycle = next(
            (
                item
                for item in self._semantic_cycles
                if item.obligation == obligation and is_same_screen(item.phash, page_hash)
            ),
            None,
        )
        if cycle is None:
            cycle = _SemanticCycle(page_hash, obligation)
            self._semantic_cycles.append(cycle)
        cycle.count += 1
        self._semantic_cycle_reason = _SEMANTIC_CYCLE
        if cycle.count >= _SEMANTIC_CYCLE_TERMINATES_AT:
            self._terminate(
                "A multi-screen route repeated without advancing the active instruction."
            )
        return _SEMANTIC_CYCLE

    def take_semantic_cycle(self) -> Optional[str]:
        """Return one newly detected route cycle to the recovery owner."""
        reason, self._semantic_cycle_reason = self._semantic_cycle_reason, None
        return reason

    def record_action(
        self,
        action: Action | PlannerTapAction,
        screen_phash: Optional[str],
        screen_size: Optional[tuple[int, int]],
        observation_id: str = "",
    ) -> Optional[str]:
        """Record a delivered attempt, or reject a quarantined control.

        This runs immediately before dispatch.  Stale attempts use
        :meth:`record_stale`, but both methods update the same control record.
        """
        identity = build_control_identity(action, screen_phash, screen_size)
        control = self._control(identity)
        self._last_action_blocked = False
        if control.quarantined:
            self._last_action_blocked = True
            return self._reject_quarantined(control)

        self._apply_settlement(
            self._settlement.begin_action(
                identity,
                screen_phash,
                observation_id,
                self.expected_back_destination(screen_phash)
                if isinstance(action, KeyEventAction) and action.key == 4
                else None,
            )
        )
        control.delivered += 1
        return self._record_attempt(control)

    def record_stale(
        self,
        action: Action | PlannerTapAction,
        screen_phash: Optional[str],
        screen_size: Optional[tuple[int, int]],
    ) -> Optional[str]:
        """Record an action abandoned because its observed target moved."""
        identity = build_control_identity(action, screen_phash, screen_size)
        control = self._control(identity)
        if control.quarantined:
            return self._reject_quarantined(control)
        control.stale += 1
        message = self._record_attempt(control)
        if control.stale >= STALE_REPEATS_BEFORE_WARNING and not control.quarantined:
            return _REPEATED_STALE.format(count=control.stale)
        return message

    def record_success(
        self,
        action: Action | PlannerTapAction,
        screen_phash: Optional[str],
        screen_size: Optional[tuple[int, int]],
    ) -> None:
        """Retain evidence until a later observation proves durable progress.

        Kept as an API boundary for dispatch callers.  Device acceptance alone
        is deliberately not treated as semantic success.
        """

    def record_delivery_outcome(
        self,
        action: Action | PlannerTapAction,
        pre_phash: Optional[str],
        post_phash: Optional[str],
        screen_size: Optional[tuple[int, int]],
        *,
        failed: bool = False,
    ) -> Optional[str]:
        """Attach dispatch evidence and defer progress to durable settlement."""
        identity = build_control_identity(action, pre_phash, screen_size)
        control = self._control(identity)
        if failed:
            control.dispatch_failed += 1
        self._apply_settlement(
            self._settlement.record_dispatch(
                identity,
                pre_phash,
                post_phash,
                failed=failed,
            )
        )
        return None

    def record_focus_verification_failure(self) -> None:
        """Count text attempts that could not prove which field held focus."""
        self._focus_verification_failures += 1
        if self._focus_verification_failures >= _FOCUS_VERIFICATION_LIMIT:
            self._focus_verification_terminated = True

    def record_verified_text_entry(self) -> None:
        """A verified text dispatch proves the input path works again."""
        self._focus_verification_failures = 0
        self._focus_verification_terminated = False

    def record_unchanged_tap(
        self,
        action: Action | PlannerTapAction,
        pre_phash: Optional[str],
        post_phash: Optional[str],
        screen_size: Optional[tuple[int, int]],
    ) -> Optional[str]:
        """Report repeated delivered taps whose visual screen stayed equivalent."""
        if not isinstance(action, TapAction | PlannerTapAction | DoubleTapAction):
            return None
        if pre_phash is None or post_phash is None:
            return None
        if not is_same_screen(pre_phash, post_phash):
            return None
        identity = build_control_identity(action, pre_phash, screen_size)
        repeated = next(
            (
                candidate
                for candidate in self._unchanged_taps
                if candidate.identity.matches(identity)
            ),
            None,
        )
        if repeated is None:
            self._unchanged_taps.append(_ControlAttempts(identity, attempts=1))
            return None
        repeated.attempts += 1
        return _UNCHANGED_TAP.format(attempts=repeated.attempts)

    def _control(self, identity: ControlIdentity) -> _ControlAttempts:
        existing = next(
            (item for item in self._controls if item.identity.matches(identity)),
            None,
        )
        if existing is not None:
            return existing
        created = _ControlAttempts(identity)
        self._controls.append(created)
        return created

    def _record_attempt(self, control: _ControlAttempts) -> Optional[str]:
        control.attempts += 1
        state = self._state(control.identity.screen_phash)
        if state.exhausted:
            self._terminate(
                "Too many actions failed to make durable progress on one screen state."
            )
            return None
        state.count += 1
        logger.info("Interaction attempt: %s", self._describe(control))
        if state.count >= _STATE_ATTEMPT_LIMIT:
            state.exhausted = True
            return (
                "This screen state has exhausted its no-progress action budget. "
                "Only durable progress can continue the run."
            )
        if control.attempts >= _QUARANTINE_AT:
            control.quarantined = True
            logger.warning("Quarantined control: %s", self._describe(control))
            return _QUARANTINED.format(count=control.attempts)
        if control.attempts == 2:
            return _SECOND_ATTEMPT
        return None

    def _reject_quarantined(self, control: _ControlAttempts) -> Optional[str]:
        control.quarantine_rejections += 1
        logger.warning("Rejected quarantined control: %s", self._describe(control))
        if (
            control.quarantine_rejections
            >= _QUARANTINE_REJECTIONS_BEFORE_TERMINATION
        ):
            self._terminate(
                "A quarantined control was selected repeatedly without progress."
            )
            return None
        return _QUARANTINE_REJECTED

    def _apply_settlement(self, receipt: SettlementReceipt | None) -> None:
        if receipt is None:
            return
        logger.info("Action settlement: %s — %s", receipt.status.value, receipt.reason)
        if receipt.status is not SettlementStatus.DURABLE_PROGRESS:
            return
        retained = (
            item
            for item in self._controls
            if not item.identity.matches(receipt.control_identity)
        )
        self._controls = deque(retained, maxlen=self._window_size)
        self._state_attempts = deque(
            (
                state
                for state in self._state_attempts
                if not is_same_screen(state.phash, receipt.source_phash)
            ),
            maxlen=self._window_size,
        )
        logger.info("Durable progress cleared evidence for the responsible control.")

    @property
    def settlement_receipts(self) -> tuple[SettlementReceipt, ...]:
        return self._settlement.receipts

    @property
    def active_settlement_id(self) -> str | None:
        return self._settlement.active_settlement_id

    def close_settlement(self, reason: str) -> None:
        """Persist an unfinished transaction when the run cannot observe again."""
        self._apply_settlement(self._settlement.close_unverified(reason))

    def expected_back_destination(self, current_phash: Optional[str]) -> str | None:
        """Return the most recent distinct observed state before the current one."""
        if current_phash is None:
            return None
        skipped_current = False
        for phash in reversed(self._screen_history):
            if not skipped_current and is_same_screen(phash, current_phash):
                skipped_current = True
                continue
            if skipped_current and not is_same_screen(phash, current_phash):
                return phash
        return None

    def _state(self, phash: Optional[str]) -> _StateAttempts:
        existing = next(
            (
                state
                for state in self._state_attempts
                if is_same_screen(state.phash, phash)
            ),
            None,
        )
        if existing is not None:
            return existing
        created = _StateAttempts(phash)
        self._state_attempts.append(created)
        return created

    def _terminate(self, reason: str) -> None:
        self._action_repetition_terminated = True
        self._termination_reason = reason

    @staticmethod
    def _describe(control: _ControlAttempts) -> str:
        return (
            f"type={control.identity.action_type} points={control.identity.points} "
            f"attempts={control.attempts} delivered={control.delivered} "
            f"stale={control.stale} failed={control.dispatch_failed} "
            f"quarantined={control.quarantined}"
        )

    def terminated(self) -> bool:
        return (
            self._action_repetition_terminated
            or self._page_stagnation_terminated
            or self._focus_verification_terminated
        )

    @property
    def action_repetition_terminated(self) -> bool:
        return self._action_repetition_terminated

    @property
    def page_stagnation_terminated(self) -> bool:
        return self._page_stagnation_terminated

    @property
    def focus_verification_terminated(self) -> bool:
        return self._focus_verification_terminated

    @property
    def action_blocked(self) -> bool:
        """Whether the most recently selected action was quarantined."""
        return self._last_action_blocked

    @property
    def termination_reason(self) -> Optional[str]:
        return self._termination_reason
