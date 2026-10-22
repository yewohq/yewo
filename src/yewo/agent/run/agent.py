"""
DeviceAgent — core ReAct loop for LLM-driven mobile device control.
No QA concerns. Accepts an optional planner override so the testing layer
can inject QAPlanner without re-implementing the loop.
"""

import logging
import time
from pathlib import Path
from typing import Callable, Optional


from yewo.agent.actuation.actions import ActionType
from yewo.agent.plan.planner import (
    GROUNDING_COORDINATES,
    DevicePlanner,
    DeviceStepResponse,
)
from yewo.agent.memory.history import AgentHistory
from yewo.agent.device.launch import SetupMode, clear_app_data, restart_app
from yewo.agent.run import outcomes, record
from yewo.agent.run.outcomes import DeviceRunResult, StepHookFn
from yewo.agent.device import state
from yewo.agent.perception import settle
from yewo.agent.step import dispatch, observation
from yewo.agent.step.observation import PerceptionSnapshot
from yewo.agent.artifacts import steps as artifacts
from yewo.agent.artifacts.steps import StepArtifacts
from yewo.agent.step import narration
from yewo.agent.memory.loop_detector import RollingLoopDetector
from yewo.agent.run.context import RunContext
from yewo.agent.memory import stall
from yewo.agent.memory.stall import STALL_LIMIT_MESSAGE
from yewo.agent.memory.replay import ReplayLedger, replay_known_path, target_label
from yewo.agent.step import targeting
from yewo.utils.secrets import SecretScrubber
from yewo.agent.grounding import GROUNDING_DESCRIBE
from yewo.agent.perception import keyboard
from yewo.agent.artifacts.redaction import RedactedImages
from yewo.agent.perception.screen import ScreenPerception
from yewo.agent.memory.trail import SessionTrail
from yewo.errors import Cause, cause_of
from yewo.llm.failures import reportable
from yewo.llm.llm_client import LLMClient

logger = logging.getLogger(__name__)


def _elapsed_ms(started: float) -> float:
    """Milliseconds, so a reader of one step sees where its time went."""
    return round((time.perf_counter() - started) * 1000, 1)

"""
Called after each planning step, before action execution.
Return False to request early loop termination (e.g. crash detected, infeasible scenario).
Return None or True to continue normally.
"""


class DeviceAgent:
    # Class-level so an agent built without __init__ still answers sensibly.
    grounding = GROUNDING_COORDINATES

    step_opened: Optional[Callable[[Optional["record.StepRecord"]], None]] = None
    """Told which step is being planned, so a model call can be attributed.

    A call is made several layers below the loop, by a planner that has no
    reason to know which step it is on. Whoever counts the calls sets this.
    Told None when the loop ends, so a later call is not charged to the last
    step.
    """

    """
    LLM-driven ReAct agent for mobile device control.

    Manages the core loop: perceive → plan → execute → observe.
    The testing layer wraps this via QAAgent, injecting a QAPlanner and
    collecting QA outputs through the step_hook callback.
    """

    def __init__(
        self,
        model: str = "openrouter/google/gemini-2.5-flash",
        max_steps: int = 40,
        settle_time: float = 2.0,
        fast_model: str = None,
        planner: Optional[DevicePlanner] = None,
        stop_when_satisfied: bool = False,
        replay_ledger: Optional[ReplayLedger] = None,
        grounding: str = GROUNDING_COORDINATES,
        grounder=None,
    ):
        self.max_steps = max_steps
        self._step_limit = max_steps
        self.settle_time = settle_time
        self.grounding = grounding
        self.stop_when_satisfied = stop_when_satisfied
        self.replay_ledger = replay_ledger
        self.scrubber = SecretScrubber()
        # Built here, so a missing extra or key stops the run before a device
        # opens rather than on the first step that needs a target placed.
        self.grounder = grounder or self._build_grounder()

        self.llm = LLMClient(model=model)
        self.fast_llm = (
            LLMClient(model=fast_model)
            if fast_model and fast_model != model
            else self.llm
        )

        if planner is not None:
            self._default_planner = planner
            self._default_fast_planner = planner
        else:
            self._default_planner = DevicePlanner(
                self.llm, grounding
            )
            self._default_fast_planner = DevicePlanner(
                self.fast_llm, grounding
            )

        self.images = RedactedImages(self.scrubber)
        self.artifacts = StepArtifacts(self.images, self.scrubber)


    def record_prompts(self, store) -> None:
        """Keeps what every call this agent makes asked.

        Set here rather than by the caller: the agent knows which clients it
        built, and whether it built a grounder at all.
        """
        self.llm.prompt_store = store
        self.fast_llm.prompt_store = store
        if self.grounder is not None:
            self.grounder.client.prompt_store = store

    def _build_grounder(self):
        """The grounder describe mode places its targets with, or None."""
        if self.grounding != GROUNDING_DESCRIBE:
            return None
        from yewo.agent.grounding.description import DecisionGrounder
        from yewo.llm.jev_client import JevClient

        return DecisionGrounder(JevClient(usage_callback=self._record_grounding))

    def _record_grounding(self, call) -> None:
        """Grounding calls cost what any other model call costs."""
        if self.llm.usage_callback:
            self.llm.usage_callback(call)

    def run(
        self,
        goal: str | list[str],
        device_control,
        perception: ScreenPerception,
        session_dir: Path,
        session_trail: Optional[SessionTrail] = None,
        step_hook: Optional[StepHookFn] = None,
        planner_override: Optional[DevicePlanner] = None,
        max_steps: Optional[int] = None,
        setup: SetupMode = SetupMode.RELAUNCH,
    ) -> DeviceRunResult:
        """Runs the ReAct loop until something stops it.

        It stops on the goal, the step limit, a detected loop, a stop from
        step_hook, or an exception.

        planner_override replaces the default planner for this run only. QAAgent
        uses it to inject a QAPlanner without building the agent again.
        """
        planner = planner_override or self._default_fast_planner
        session_dir.mkdir(parents=True, exist_ok=True)
        ctx = RunContext(
            history=AgentHistory(),
            loop_detector=RollingLoopDetector(),
            steps=record.StepLog(self.scrubber.scrub),
            screen_height=getattr(device_control, "screen_height", None),
        )

        try:
            # Establish the state the caller asked for, rather than describing
            # a state to the planner and hoping the device agrees.
            if setup is SetupMode.CLEAR_DATA:
                clear_app_data(device_control)
            if setup is not SetupMode.PRESERVE:
                restart_app(device_control)
            replay_known_path(
                ledger=self.replay_ledger,
                device_control=device_control,
                goal=goal,
                history=ctx.history,
                quick_phash=settle.quick_phash,
                settle=lambda control: settle.settle(control, self.settle_time),
                scrubber=self.scrubber,
            )
            self._step_limit = max_steps or self.max_steps
            for step in range(self._step_limit):
                result = self._run_step(
                    step=step,
                    goal=goal,
                    planner=planner,
                    device_control=device_control,
                    perception=perception,
                    session_dir=session_dir,
                    session_trail=session_trail,
                    step_hook=step_hook,
                    ctx=ctx,
                )
                if result is not None:
                    self._log_recovery_cost(ctx)
                    return result

            self._log_recovery_cost(ctx)
            return outcomes.step_limit(self._step_limit, ctx, session_trail)

        except Exception as exc:
            if cause_of(exc) is Cause.PROVIDER:
                # Not a result. The run reached no state worth judging, and
                # the suite decides what a service that stopped answering
                # means. Read by cause rather than by type, so a refused key
                # leaves the same way a spent one does. Device state is
                # restored as this leaves.
                raise
            safe_error = self.scrubber.scrub(reportable(exc))
            logger.error(
                "Device agent run failed (%s): %s", type(exc).__name__, safe_error
            )
            self._log_recovery_cost(ctx)
            return DeviceRunResult(
                status="exception",
                steps_executed=ctx.steps_executed,
                failure_reason=f"Execution exception: {safe_error}",
                history=ctx.history,
                steps=list(ctx.steps.records),
                cause=cause_of(exc),
            )
        finally:
            # A call made after the last step is not charged to it. The
            # auditor and the route judge both answer once the loop is over.
            if self.step_opened is not None:
                self.step_opened(None)

    def _open_record(self, ctx: RunContext, step: int) -> None:
        """Starts this step's record, and says which step a call belongs to.

        The marker is how a model call is attributed. The call itself is made
        several layers down, by a planner that has no reason to know which
        step it is on.
        """
        written = ctx.steps.open(step + 1)
        if self.step_opened is not None:
            self.step_opened(written)

    @staticmethod
    def _log_recovery_cost(ctx: RunContext) -> None:
        """Says what rescuing the run cost, which the step count does not show."""
        if ctx.recovery_looks:
            logger.info(
                "Recovery was used on %d screen(s), costing %d extra model call(s).",
                len(ctx.recovered),
                ctx.recovery_looks,
            )

    def _run_step(
        self,
        step: int,
        goal: str | list[str],
        planner: DevicePlanner,
        device_control,
        perception: ScreenPerception,
        session_dir: Path,
        session_trail: Optional[SessionTrail],
        step_hook: Optional[StepHookFn],
        ctx: RunContext,
    ) -> Optional[DeviceRunResult]:
        """One turn of the loop, as the five things it does."""
        ctx.steps_executed += 1
        logger.info("--- [DeviceAgent] Step %d / %d ---", step + 1, self.max_steps)

        self._open_record(ctx, step)

        started = time.perf_counter()
        snapshot = self._observe(step, device_control, perception, session_dir,
                                 session_trail, ctx)
        ctx.steps.observed(snapshot.current_phash, _elapsed_ms(started))

        ending = outcomes.page_stagnation(ctx, session_trail, snapshot.current_phash)
        if ending is not None:
            return ending

        started = time.perf_counter()
        step_response = self._plan(step, goal, planner, snapshot, session_dir,
                                   session_trail, ctx)
        ctx.steps.planned(step_response, _elapsed_ms(started))

        ending = self._ending_after_planning(
            step_response, snapshot, session_trail, step_hook, ctx
        )
        if ending is not None:
            return ending

        decision = targeting.evaluate(
            step_response.action, snapshot, device_control, self.grounding,
            self.grounder,
        )
        ctx.steps.record_grounding(self.grounding, decision, step_response.action)
        if not decision.ok:
            return self._blocked(step, step_response.action, decision, snapshot,
                                 device_control, session_dir, session_trail, ctx)

        return self._act(step, decision.action or step_response.action, step_response,
                         device_control, snapshot, session_dir, session_trail, ctx)

    def _observe(
        self, step, device_control, perception, session_dir, session_trail, ctx
    ) -> PerceptionSnapshot:
        """Reads the screen, writes it down, and says what the last action did."""
        ctx.drift_warning = state.context_drift(
            device_control, ctx.last_external_nav, self.settle_time
        )

        snapshot = observation.capture(perception, session_trail, ctx.prev_phash)
        ctx.current_phash = snapshot.current_phash
        ctx.steps.files(self.artifacts.save_step(session_dir, step, snapshot))

        # The keyboard resizes the app, so its state is only worth re-reading
        # where something could have changed it.
        moved = snapshot.scan_info and snapshot.scan_info.reason in (
            "navigation", "layout_shift"
        )
        if moved or ctx.last_action_type == ActionType.TYPE.value:
            detected = device_control.is_keyboard_open()
            if detected is not None:
                ctx.keyboard_open = detected

        ctx.history.update_last_reaction(
            narration.reaction(
                snapshot.scan_info, ctx.prev_element_count, ctx.keyboard_open
            )
        )
        ctx.prev_element_count = (
            snapshot.scan_info.element_count if snapshot.scan_info else None
        )
        ctx.pending_loop_nudge = "\n".join(
            filter(None, [
                ctx.pending_loop_nudge,
                ctx.loop_detector.check(None, snapshot.current_phash),
            ])
        ) or None
        return snapshot

    def _plan(
        self, step, goal, planner, snapshot, session_dir, session_trail, ctx
    ) -> DeviceStepResponse:
        """Numbers the targets, asks for an action, and records what came back."""
        # Saved before the model is called, so a planning failure still leaves
        # the image the planner was given.
        observation.number_targets(snapshot, ctx, self.grounding)
        ctx.steps.files(
            {"marked": self.artifacts.save_planner_view(session_dir, step, snapshot)}
        )

        nudge, ctx.pending_loop_nudge = ctx.pending_loop_nudge, None
        step_response = self._plan_step(
            planner=planner, goal=goal, snapshot=snapshot, ctx=ctx,
            session_trail=session_trail, loop_nudge=nudge,
        )
        self._apply_plan_result(step_response, snapshot, session_trail, ctx)
        return step_response

    def _ending_after_planning(
        self, step_response, snapshot, session_trail, step_hook, ctx
    ) -> Optional[DeviceRunResult]:
        """The three ways a plan ends the run instead of reaching the device."""
        if step_hook:
            stopped = outcomes.hook_stop(
                step_hook, step_response, snapshot, session_trail,
                snapshot.current_phash, ctx,
            )
            if stopped is not None:
                return stopped

        done = outcomes.goal_reached(
            self.stop_when_satisfied, step_response, session_trail,
            snapshot.current_phash, ctx,
        )
        if done is not None:
            return done

        ctx.pending_loop_nudge = ctx.loop_detector.check(
            RollingLoopDetector.hash_action(step_response.action), None
        )
        return outcomes.action_repetition(ctx, session_trail, snapshot.current_phash)

    def _blocked(
        self, step, action, decision, snapshot, device_control,
        session_dir, session_trail, ctx
    ) -> Optional[DeviceRunResult]:
        """An action that never reached the device, and what that costs the screen."""
        action_description = narration.describe(action)
        targeting.record_unexecuted(
            ctx.history, step, action_description, decision.observation
        )
        ctx.steps.record_blocked_action(
            action,
            action_description,
            decision.observation,
            target_label(action, snapshot.gui_elements),
        )
        if decision.unmatched:
            artifacts.save_unmatched_screen(
                self.images,
                session_dir
                / f"step{step + 1:03d}_{snapshot.timestamp}_not_listed.png",
                snapshot,
            )
        if stall.reveal_behind_keyboard(device_control, snapshot, ctx):
            return None
        ctx.stall.record_blocked(snapshot.current_phash)
        if ctx.stall.is_exhausted:
            return outcomes.loop_detected(
                ctx, session_trail, snapshot.current_phash,
                targeting.stalled_reason(decision),
            )
        return None

    def _act(
        self, step, action, step_response, device_control, snapshot,
        session_dir, session_trail, ctx
    ) -> Optional[DeviceRunResult]:
        """Touches the device, and decides whether the screen is going anywhere."""
        ctx.steps.files(
            {"action": self.artifacts.save_action(session_dir, step, snapshot, action)}
        )
        started = time.perf_counter()
        ctx.last_external_nav = self._execute_and_record(
            step=step, action=action, step_response=step_response,
            device_control=device_control, snapshot=snapshot,
            session_trail=session_trail, ctx=ctx,
        )
        ctx.steps.record_action_duration(_elapsed_ms(started))
        if ctx.stall.is_blocked and stall.reveal_behind_keyboard(
            device_control, snapshot, ctx
        ):
            return None
        # An action that runs and fails stalls a screen as surely as one refused
        # before dispatch. Burning a mark pushes the planner to try a different
        # one, so a fresh target each time is still a screen going nowhere.
        if ctx.stall.is_exhausted:
            return outcomes.loop_detected(
                ctx, session_trail, snapshot.current_phash, STALL_LIMIT_MESSAGE
            )
        ctx.prev_phash = snapshot.current_phash
        return None

    def _plan_step(
        self,
        planner: DevicePlanner,
        goal: str | list[str],
        snapshot: PerceptionSnapshot,
        ctx: RunContext,
        session_trail: Optional[SessionTrail],
        loop_nudge: Optional[str],
    ) -> DeviceStepResponse:
        # A keyboard commonly resizes the app and moves every field, so plan
        # from the current scan. Pre-keyboard coordinates are stale.
        gui_elements = keyboard.annotate_keyboard_keys(
            snapshot.gui_elements, ctx.keyboard_open, ctx.screen_height
        )
        return planner.plan_step(
            goal=goal,
            gui_elements=gui_elements,
            history=ctx.history.format_for_prompt(),
            trail=session_trail.format_for_prompt() if session_trail else None,
            drift_warning=ctx.drift_warning,
            loop_nudge=loop_nudge,
            current_image_bytes=snapshot.planner_img_bytes,
            marks=snapshot.marks,
            screen_height=ctx.screen_height or 0,
            keyboard_open=ctx.keyboard_open,
        )


    def _apply_plan_result(
        self,
        step_response: DeviceStepResponse,
        snapshot: PerceptionSnapshot,
        session_trail: Optional[SessionTrail],
        ctx: RunContext,
    ) -> None:
        logger.info("Screen: %s", step_response.current_screen)
        logger.info("Reasoning: %s", self.scrubber.scrub(step_response.reasoning))
        if step_response.evaluation_previous_goal:
            logger.info(
                "Previous step eval: %s",
                self.scrubber.scrub(step_response.evaluation_previous_goal),
            )

        ctx.history.update_last_evaluation(step_response.evaluation_previous_goal)

        if session_trail and snapshot.current_phash:
            session_trail.record_screen(
                snapshot.current_phash,
                step_response.current_screen,
                snapshot.gui_elements,
            )





    def _execute_and_record(
        self,
        step: int,
        action,
        step_response: DeviceStepResponse,
        device_control,
        snapshot: PerceptionSnapshot,
        session_trail: Optional[SessionTrail],
        ctx: RunContext,
    ) -> bool:
        return dispatch.execute(
            step=step,
            action=action,
            step_response=step_response,
            device_control=device_control,
            snapshot=snapshot,
            session_trail=session_trail,
            ctx=ctx,
            scrubber=self.scrubber,
            replay_ledger=self.replay_ledger,
            settle_time=self.settle_time,
        )
