"""Bounded, auditable recovery without changing a test's meaning.

Recovery is a transaction around one failed original intent.  A model may
propose a bridge action, but policy owns the budget and current-run evidence
owns the verdict.  Nothing in this module contains application, platform, or
language-specific labels.
"""

from enum import Enum
from typing import Callable, Optional

from pydantic import BaseModel, Field

from yewo.identity import new_uuid
from yewo.agent.run.record import StepSource


class RecoveryCause(str, Enum):
    MECHANICAL = "mechanical"
    GROUNDING_DRIFT = "grounding_drift"
    SEMANTIC_OBSTRUCTION = "semantic_obstruction"
    NON_RECOVERABLE = "non_recoverable"


class RecoveryMode(str, Enum):
    DETERMINISTIC = "deterministic"
    MODEL_PROPOSAL = "model_proposal"
    STOP = "stop"


class RecoveryStatus(str, Enum):
    OPEN = "open"
    RESUMED = "resumed"
    RECOVERED = "recovered"
    FAILED = "failed"


def recovery_mode_for(cause: RecoveryCause) -> RecoveryMode:
    """Select the authority boundary from typed, application-neutral evidence."""
    if cause is RecoveryCause.MECHANICAL:
        return RecoveryMode.DETERMINISTIC
    if cause in (RecoveryCause.GROUNDING_DRIFT, RecoveryCause.SEMANTIC_OBSTRUCTION):
        return RecoveryMode.MODEL_PROPOSAL
    return RecoveryMode.STOP


class RecoveryActionRecord(BaseModel):
    """One secret-free bridge proposal and the device's response."""

    step: int
    state_key: str
    action_type: str
    description: str
    success: bool
    observation: str = ""


class RecoverySuggestion(BaseModel):
    """A post-run maintenance proposal; never an automatically applied edit."""

    kind: str
    reason: str
    human_review_required: bool = True
    applied: bool = False


class RecoveryTransaction(BaseModel):
    """The complete record of one attempt to restore an original intent."""

    recovery_id: str = Field(default_factory=new_uuid)
    failed_step: int = Field(ge=1)
    opened_step: int = Field(ge=1)
    original_intent: str
    value_name: Optional[str] = None
    failure_reason: str
    cause: RecoveryCause
    mode: RecoveryMode
    status: RecoveryStatus = RecoveryStatus.OPEN
    bridge_budget: int = Field(default=3, ge=0)
    model_budget: int = Field(default=3, ge=0)
    actions: list[RecoveryActionRecord] = Field(default_factory=list)
    proposed_states: list[str] = Field(default_factory=list)
    active_state_key: Optional[str] = None
    final_evidence: Optional[str] = None
    suggestion: Optional[RecoverySuggestion] = None

    @classmethod
    def open(
        cls,
        *,
        failed_step: int,
        opened_step: int,
        original_intent: str,
        failure_reason: str,
        cause: RecoveryCause,
        value_name: Optional[str] = None,
    ) -> "RecoveryTransaction":
        return cls(
            failed_step=failed_step,
            opened_step=opened_step,
            original_intent=original_intent,
            value_name=value_name,
            failure_reason=failure_reason,
            cause=cause,
            mode=recovery_mode_for(cause),
        )

    @property
    def bridge_actions_remaining(self) -> int:
        return max(0, self.bridge_budget - len(self.actions))

    @property
    def is_terminal(self) -> bool:
        return self.status in (
            RecoveryStatus.RESUMED,
            RecoveryStatus.RECOVERED,
            RecoveryStatus.FAILED,
        )

    def _require_open(self) -> None:
        if self.status is not RecoveryStatus.OPEN:
            raise ValueError("A terminal recovery transaction is immutable.")

    @property
    def model_proposals_remaining(self) -> int:
        return max(0, self.model_budget - len(self.proposed_states))

    def may_propose(self, state_key: str) -> bool:
        """A model gets one proposal for an equivalent observed state."""
        return (
            self.status is RecoveryStatus.OPEN
            and self.mode is RecoveryMode.MODEL_PROPOSAL
            and self.bridge_actions_remaining > 0
            and self.model_proposals_remaining > 0
            and state_key not in self.proposed_states
        )

    def record_proposal(self, state_key: str) -> None:
        if not self.may_propose(state_key):
            raise ValueError("Recovery proposal is not permitted for this state.")
        self.proposed_states.append(state_key)

    def record_action(self, action: RecoveryActionRecord) -> None:
        if self.status is not RecoveryStatus.OPEN:
            raise ValueError("A closed recovery transaction cannot accept actions.")
        if self.bridge_actions_remaining == 0:
            raise ValueError("Recovery bridge-action budget is exhausted.")
        self.actions.append(action)

    def recover(self, evidence: str) -> None:
        self._require_open()
        self.status = RecoveryStatus.RECOVERED
        self.final_evidence = evidence
        if self.cause is RecoveryCause.GROUNDING_DRIFT:
            self.suggestion = RecoverySuggestion(
                kind="review_grounding",
                reason=(
                    "A changed grounding was recovered at runtime. Review the evidence "
                    "before changing a saved locator or path."
                ),
            )

    def resume(self, evidence: str) -> None:
        """Record that execution escaped a cycle without claiming the intent passed."""
        self._require_open()
        self.status = RecoveryStatus.RESUMED
        self.final_evidence = evidence

    def fail(self, evidence: str) -> None:
        self._require_open()
        self.status = RecoveryStatus.FAILED
        self.final_evidence = evidence


def observed_state_key(
    phash: Optional[str],
    keyboard_open: Optional[bool],
    focused_bounds: Optional[tuple[int, int, int, int]] = None,
) -> str:
    """A platform-neutral state key used only to prevent identical proposals."""
    keyboard = "unknown" if keyboard_open is None else "open" if keyboard_open else "closed"
    focus = "none" if focused_bounds is None else ",".join(map(str, focused_bounds))
    return f"{phash or 'unidentified'}:keyboard={keyboard}:focus={focus}"


def recovery_directive(transaction: RecoveryTransaction) -> str:
    """Constrain a semantic planner without asking it to grade itself."""
    return (
        "RECOVERY TRANSACTION: preserve the original test intent exactly. "
        f"The unsatisfied intent is: {transaction.original_intent}. "
        "Choose one reversible bridge action that makes that intent possible. "
        "Do not skip the intent, weaken an assertion, change test data, declare "
        "success, or use a generic Back/Scroll action without visible evidence. "
        f"Bridge actions remaining: {transaction.bridge_actions_remaining}."
    )


def prepare_recovery_proposal(ctx, snapshot, device_control) -> bool:
    """Authorize one proposal for this observed state or close the transaction."""
    transaction = ctx.active_recovery
    if transaction is None:
        return False
    focused_field = device_control.read_focused_field()
    state_key = observed_state_key(
        snapshot.current_phash,
        ctx.keyboard_open,
        focused_field.bounds if focused_field is not None else None,
    )
    if not transaction.may_propose(state_key):
        reason = (
            "Recovery stopped without repeating a proposal. The equivalent state "
            "was already attempted or the bounded recovery budget was exhausted."
        )
        transaction.fail(reason)
        ctx.active_recovery = None
        ctx.postcondition_blocked = True
        return True

    transaction.record_proposal(state_key)
    transaction.active_state_key = state_key
    ctx.recovery_looks += 1
    ctx.recovered.add(snapshot.current_phash or state_key)
    ctx.pending_loop_nudge = "\n".join(
        filter(None, [ctx.pending_loop_nudge, recovery_directive(transaction)])
    )
    if ctx.steps.current is not None:
        ctx.steps.current.source = StepSource.RECOVERY
        ctx.steps.current.recovery_id = transaction.recovery_id
    return False


def record_recovery_action(ctx, snapshot) -> None:
    """Attach the current bridge action and secret-free response to its transaction."""
    transaction = ctx.active_recovery
    step_record = ctx.steps.current
    if (
        transaction is None
        or step_record is None
        or step_record.recovery_id != transaction.recovery_id
        or step_record.action is None
        or step_record.outcome is None
    ):
        return
    transaction.record_action(
        RecoveryActionRecord(
            step=step_record.n,
            state_key=(
                transaction.active_state_key
                or observed_state_key(snapshot.current_phash, ctx.keyboard_open)
            ),
            action_type=step_record.action.type,
            description=step_record.action.description,
            success=step_record.outcome.success,
            observation=step_record.outcome.observation,
        )
    )


def scrub_recovery_transaction(
    transaction: RecoveryTransaction,
    scrub: Callable[[Optional[str]], str],
) -> RecoveryTransaction:
    """Copy a transaction for a durable artifact and scrub all prose."""
    kept = transaction.model_copy(deep=True)
    kept.original_intent = scrub(kept.original_intent)
    kept.failure_reason = scrub(kept.failure_reason)
    kept.final_evidence = scrub(kept.final_evidence) or None
    for action in kept.actions:
        action.description = scrub(action.description)
        action.observation = scrub(action.observation)
    if kept.suggestion is not None:
        kept.suggestion.reason = scrub(kept.suggestion.reason)
    return kept


def fail_active_recovery(ctx, evidence: str) -> None:
    """Close the current transaction without changing the original oracle."""
    transaction = ctx.active_recovery
    if transaction is None:
        return
    transaction.fail(evidence)
    ctx.active_recovery = None


def open_semantic_cycle_recovery(ctx, reason: Optional[str]) -> None:
    """Open one bounded alternate-route attempt for a detected semantic cycle."""
    if not reason or ctx.active_recovery is not None or not ctx.active_instruction:
        return
    current_step = ctx.steps.current.n if ctx.steps.current is not None else 1
    transaction = RecoveryTransaction.open(
        failed_step=max(1, current_step - 1),
        opened_step=current_step,
        original_intent=ctx.active_instruction,
        failure_reason=reason,
        cause=RecoveryCause.SEMANTIC_OBSTRUCTION,
    )
    ctx.recovery_transactions.append(transaction)
    ctx.active_recovery = transaction
