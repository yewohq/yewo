"""A resumable observe-plan-act task for one device goal."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from yewo.agent.device.launch import SetupMode, clear_app_data, restart_app
from yewo.defaults import DEFAULT_INTERVENTION_STEPS
from yewo.agent.memory.history import AgentHistory
from yewo.agent.memory.loop_detector import RollingLoopDetector
from yewo.agent.memory.replay import replay_known_path
from yewo.agent.run import outcomes, record, saved_path
from yewo.agent.run.context import RunContext
from yewo.agent.run.evidence_identity import ExecutionContextIdentity
from yewo.agent.run.instruction_controller import InstructionController
from yewo.agent.run.outcomes import DeviceRunResult, ObservationHookFn, StepHookFn
from yewo.agent.run.postconditions import close_pending_input
from yewo.agent.perception import settle
from yewo.errors import Cause, DeviceUnavailable, cause_of
from yewo.llm.failures import reportable
from yewo.utils.secrets import SecretScrubber

if TYPE_CHECKING:
    from yewo.agent.memory.trail import SessionTrail
    from yewo.agent.perception.screen import ScreenPerception
    from yewo.agent.plan.planner import DevicePlanner
    from yewo.agent.run.agent import DeviceAgent

logger = logging.getLogger(__name__)


class DeviceTask:
    """Advance one goal by one agent turn at a time."""

    def __init__(
        self,
        agent: "DeviceAgent",
        goal: str | list[str],
        device_control,
        perception: "ScreenPerception",
        session_dir: Path,
        session_trail: Optional["SessionTrail"] = None,
        observation_hook: Optional[ObservationHookFn] = None,
        before_action_hook: Optional[outcomes.BeforeActionHookFn] = None,
        step_hook: Optional[StepHookFn] = None,
        planner: Optional["DevicePlanner"] = None,
        max_steps: Optional[int] = None,
        setup: SetupMode = SetupMode.RELAUNCH,
        replay: bool = True,
    ):
        self.agent = agent
        self.goal = goal
        self.device_control = device_control
        self.perception = perception
        self.session_dir = session_dir
        self.session_trail = session_trail
        self.observation_hook = observation_hook
        self.before_action_hook = before_action_hook
        self.step_hook = step_hook
        self.planner = planner or agent._default_fast_planner
        self.step_limit = max_steps or agent.max_steps
        self.intervention_limit = DEFAULT_INTERVENTION_STEPS
        self.step_index = 0
        self.result: Optional[DeviceRunResult] = None
        self._is_closed = False
        self.context = self._build_context(setup)
        self._start(setup, replay)

    def _build_context(self, setup: SetupMode) -> RunContext:
        if isinstance(self.agent.scrubber, SecretScrubber):
            self.agent.scrubber.clear_protected_regions()
        self.session_dir.mkdir(parents=True, exist_ok=True)
        instruction_controller = InstructionController.from_goal(self.goal)
        platform_value = getattr(self.device_control, "platform", "unidentified")
        platform = getattr(platform_value, "value", str(platform_value))
        app_id = getattr(self.device_control, "package_id", None) or "unidentified"
        execution_context = ExecutionContextIdentity.build(
            app_id=app_id,
            platform=platform,
            actor="default",
            setup=setup.value,
            instruction_ids=tuple(
                instruction.instruction_id
                for instruction in instruction_controller.instructions
            ),
        )
        active_instruction = instruction_controller.active_instruction
        return RunContext(
            history=AgentHistory(),
            loop_detector=RollingLoopDetector(),
            steps=record.StepLog(self.agent.scrubber.scrub),
            screen_height=self.device_control.screen_height,
            declared_values=dict(self.agent.declared_values),
            value_verifications=dict(self.agent.value_verifications),
            value_names=self.agent.value_names,
            active_instruction_index=instruction_controller.active_index,
            active_instruction=(
                active_instruction.intent if active_instruction is not None else None
            ),
            instruction_controller=instruction_controller,
            execution_context=execution_context,
        )

    def _start(self, setup: SetupMode, replay: bool) -> None:
        try:
            if setup is SetupMode.CLEAR_DATA and not clear_app_data(self.device_control):
                raise DeviceUnavailable("The requested clean app state was not applied.")
            if setup is not SetupMode.PRESERVE:
                restart_app(self.device_control)
            if replay:
                replay_known_path(
                    ledger=self.agent.replay_ledger,
                    device_control=self.device_control,
                    goal=self.goal,
                    history=self.context.history,
                    quick_phash=settle.quick_phash,
                    settle=lambda control: settle.settle(control, self.agent.settle_time),
                    scrubber=self.agent.scrubber,
                    instruction_controller=self.context.instruction_controller,
                    context_id=(
                        self.context.execution_context.context_id
                        if self.context.execution_context is not None
                        else None
                    ),
                )
            self.agent._step_limit = self.step_limit
        except BaseException as exc:
            if not isinstance(exc, Exception):
                self.agent.interrupted_result = self.interrupted_result(exc)
                self.close()
                raise
            self._handle_exception(exc)

    @property
    def is_finished(self) -> bool:
        return self.result is not None

    def step(self) -> Optional[DeviceRunResult]:
        """Observe, plan, and perform at most one action."""
        if self.result is not None:
            return self.result
        try:
            charged = self.context.turns_spent
            result = self.agent._run_step(
                step=self.step_index,
                goal=self.goal,
                planner=self.planner,
                device_control=self.device_control,
                perception=self.perception,
                session_dir=self.session_dir,
                session_trail=self.session_trail,
                observation_hook=self.observation_hook,
                before_action_hook=self.before_action_hook,
                step_hook=self.step_hook,
                ctx=self.context,
            )
            self.step_index += 1
            if self.context.turns_spent == charged:
                # A turn that reached no charge point still cost an observation.
                self.context.spend_step()
            if result is not None:
                return self._complete_after_terminal(result)
            if self.context.intervention_steps >= self.intervention_limit:
                return self._complete_after_intervention_limit()
            # The instruction's own steps bound the walk. Repair turns spend
            # their own allowance, and the sum bounds a turn that charged
            # neither, so no path can spin.
            if self.context.steps_executed >= self.step_limit:
                return self._complete_after_limit()
            if self.step_index >= self.step_limit + self.intervention_limit:
                return self._complete_after_limit()
            return None
        except BaseException as exc:
            if not isinstance(exc, Exception):
                self.agent.interrupted_result = self.interrupted_result(exc)
                self.close()
                raise
            return self._handle_exception(exc)

    def _record_arrival(self) -> bool:
        return saved_path.record_arrival(
            self.agent,
            self.device_control,
            self.perception,
            self.session_trail,
            self.context,
            self.observation_hook,
        )

    def _complete_after_terminal(self, result: DeviceRunResult) -> DeviceRunResult:
        self._record_arrival()
        if self.context.postcondition_blocked:
            result = outcomes.postcondition_stop(
                self.context, self.session_trail, self.context.current_phash
            )
        return self._complete(result)

    def _complete_after_limit(self) -> DeviceRunResult:
        completed_from_path = self._record_arrival()
        if self.context.postcondition_blocked:
            return self._complete(
                outcomes.postcondition_stop(
                    self.context, self.session_trail, self.context.current_phash
                )
            )
        if completed_from_path:
            cached_result = saved_path.cached_path_result(
                self.agent,
                self.context,
                self.session_trail,
                self.context.current_phash,
            )
            if cached_result is not None:
                return self._complete(cached_result)
        return self._complete(
            outcomes.step_limit(self.step_limit, self.context, self.session_trail)
        )

    def _complete_after_intervention_limit(self) -> DeviceRunResult:
        return self._complete(
            outcomes.intervention_limit(
                self.intervention_limit, self.context, self.session_trail
            )
        )

    def _handle_exception(self, exc: Exception) -> DeviceRunResult:
        if cause_of(exc) is Cause.PROVIDER:
            self.close()
            raise
        safe_error = self.agent.scrubber.scrub(reportable(exc))
        logger.error("Device agent task failed (%s): %s", type(exc).__name__, safe_error)
        return self._complete(
            DeviceRunResult(
                status="exception",
                steps_executed=self.context.steps_executed,
                failure_reason=f"Execution exception: {safe_error}",
                history=self.context.history,
                steps=list(self.context.steps.records),
                cause=cause_of(exc),
                postcondition_verifications=list(
                    self.context.postcondition_verifications
                ),
                recovery_transactions=list(self.context.recovery_transactions),
            )
        )

    def interrupted_result(self, exc: BaseException) -> DeviceRunResult:
        """Snapshot the evidence accumulated before an external interruption."""
        safe_error = self.agent.scrubber.scrub(reportable(exc))
        self._close_pending_contracts("Execution ended before another observation.")
        return DeviceRunResult(
            status="exception",
            steps_executed=self.context.steps_executed,
            failure_reason=f"Execution interrupted: {safe_error}",
            history=self.context.history,
            capability_receipts=list(self.context.capability_receipts),
            steps=list(self.context.steps.records),
            cause=None if isinstance(exc, KeyboardInterrupt) else cause_of(exc),
            postcondition_verifications=list(self.context.postcondition_verifications),
            recovery_transactions=list(self.context.recovery_transactions),
            settlement_receipts=list(self.context.loop_detector.settlement_receipts),
            input_transactions=list(self.context.input_transactions),
        )

    def _complete(self, result: DeviceRunResult) -> DeviceRunResult:
        outcomes.log_recovery_cost(self.context)
        self._close_pending_contracts("The run ended before durable settlement.")
        result.settlement_receipts = list(
            self.context.loop_detector.settlement_receipts
        )
        result.input_transactions = list(self.context.input_transactions)
        self.result = result
        self.close()
        return result

    def _close_pending_contracts(self, reason: str) -> None:
        self.context.loop_detector.close_settlement(reason)
        close_pending_input(self.context, reason)

    def close(self) -> None:
        """End step attribution without changing the device."""
        if self._is_closed:
            return
        self._is_closed = True
        if self.agent.step_opened is not None:
            self.agent.step_opened(None)
