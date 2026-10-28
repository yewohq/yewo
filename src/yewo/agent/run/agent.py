"""The observe-plan-act loop for mobile device control."""

import logging
import time
from pathlib import Path
from typing import Callable, Optional

from yewo.agent.actuation.actions import ActionType, ShowKeyboardTargetsAction
from yewo.agent.grounding.runtime import GroundingModels, ground_and_protect_typed_secret
from yewo.agent.plan.planner import (
    GROUNDING_COORDINATES,
    DevicePlanner,
    DeviceStepResponse,
)
from yewo.agent.device.launch import SetupMode
from yewo.agent.run import (
    keyboard_targets,
    observing,
    outcomes,
    postconditions,
    record,
    saved_path,
)
from yewo.agent.run.outcomes import DeviceRunResult, ObservationHookFn, StepHookFn
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
from yewo.agent.memory.replay import ReplayLedger, target_label
from yewo.agent.memory.replay_store import ValueNames
from yewo.agent.step import targeting
from yewo.utils.secrets import SecretScrubber
from yewo.agent.perception import keyboard
from yewo.agent.artifacts.redaction import RedactedImages
from yewo.agent.perception.screen import ScreenPerception
from yewo.agent.memory.trail import SessionTrail
from yewo.agent.run.task import DeviceTask
from yewo.llm.llm_client import LLMClient

logger = logging.getLogger(__name__)


class DeviceAgent:
    """Perceive, plan, execute, and observe until the run ends."""

    # Class-level so an agent built without __init__ still answers sensibly.
    grounding = GROUNDING_COORDINATES

    step_opened: Optional[Callable[[Optional["record.StepRecord"]], None]] = None
    """Receives the active record, then None so post-run calls are uncharged."""

    grounding_usage_callback: Optional[Callable] = None

    durable_replay = None
    """A path an earlier run saved. A class default, so an agent built
    without __init__ still answers for it."""

    declared_values: dict[str, str] = {}
    value_verifications: dict[str, postconditions.ValueVerificationPolicy] = {}
    value_names = ValueNames()

    def __init__(
        self,
        model: str = "openrouter/google/gemini-2.5-flash",
        max_steps: int = 40,
        settle_time: float = 2.0,
        fast_model: str = None,
        planner: Optional[DevicePlanner] = None,
        stop_when_satisfied: bool = False,
        replay_ledger: Optional[ReplayLedger] = None,
        durable_replay=None,
        grounding: str = GROUNDING_COORDINATES,
        grounder=None,
    ):
        self.max_steps = max_steps
        self._step_limit = max_steps
        self.settle_time = settle_time
        self.grounding = grounding
        self.stop_when_satisfied = stop_when_satisfied
        self.replay_ledger = replay_ledger
        self.durable_replay = durable_replay
        self.declared_values = {}
        self.value_verifications = {}
        self.value_names = ValueNames()
        self.scrubber = SecretScrubber()
        # Built here, so a missing extra or key stops the run before a device
        # opens rather than on the first step that needs a target placed.
        self.grounding_models = GroundingModels(grounding, grounder, self._record_grounding)
        self.grounder = self.grounding_models.planner

        self.llm = LLMClient(model=model)
        self.fast_llm = (
            LLMClient(model=fast_model) if fast_model and fast_model != model else self.llm
        )

        if planner is not None:
            self._default_planner = planner
            self._default_fast_planner = planner
        else:
            self._default_planner = DevicePlanner(self.llm, grounding, scrubber=self.scrubber)
            self._default_fast_planner = DevicePlanner(
                self.fast_llm, grounding, scrubber=self.scrubber
            )

        self.images = RedactedImages(self.scrubber)
        self.artifacts = StepArtifacts(self.images, self.scrubber)

    def set_declared_values(
        self,
        values: dict[str, str],
        verification_policies: Optional[
            dict[str, postconditions.ValueVerificationPolicy]
        ] = None,
    ) -> None:
        """Supply current-run values for local action verification."""
        self.declared_values = dict(values)
        self.value_verifications = dict(verification_policies or {})
        self.value_names = ValueNames(self.declared_values)

    def record_prompts(self, store) -> None:
        """Keeps what every call this agent makes asked.

        Set here rather than by the caller: the agent knows which clients it
        built, and whether it built a grounder at all.
        """
        self.llm.prompt_store = store
        self.fast_llm.prompt_store = store
        self.grounding_models.record_prompts(store)

    def _record_grounding(self, call) -> None:
        if self.grounding_usage_callback is not None:
            self.grounding_usage_callback(call)
        elif self.llm.usage_callback:
            self.llm.usage_callback(call)

    def run(
        self,
        goal: str | list[str],
        device_control,
        perception: ScreenPerception,
        session_dir: Path,
        session_trail: Optional[SessionTrail] = None,
        observation_hook: Optional[ObservationHookFn] = None,
        before_action_hook: Optional[outcomes.BeforeActionHookFn] = None,
        step_hook: Optional[StepHookFn] = None,
        planner_override: Optional[DevicePlanner] = None,
        max_steps: Optional[int] = None,
        setup: SetupMode = SetupMode.RELAUNCH,
        replay: bool = True,
    ) -> DeviceRunResult:
        """Run until the goal, a safety boundary, the limit, or an exception."""
        task = self.start_task(
            goal=goal,
            device_control=device_control,
            perception=perception,
            session_dir=session_dir,
            session_trail=session_trail,
            observation_hook=observation_hook,
            before_action_hook=before_action_hook,
            step_hook=step_hook,
            planner_override=planner_override,
            max_steps=max_steps,
            setup=setup,
            replay=replay,
        )
        result = task.result
        while result is None:
            result = task.step()
        return result

    def start_task(
        self,
        goal: str | list[str],
        device_control,
        perception: ScreenPerception,
        session_dir: Path,
        session_trail: Optional[SessionTrail] = None,
        observation_hook: Optional[ObservationHookFn] = None,
        before_action_hook: Optional[outcomes.BeforeActionHookFn] = None,
        step_hook: Optional[StepHookFn] = None,
        planner_override: Optional[DevicePlanner] = None,
        max_steps: Optional[int] = None,
        setup: SetupMode = SetupMode.RELAUNCH,
        replay: bool = True,
    ) -> DeviceTask:
        """Open a goal that a benchmark can advance one turn at a time."""
        return DeviceTask(
            agent=self,
            goal=goal,
            device_control=device_control,
            perception=perception,
            session_dir=session_dir,
            session_trail=session_trail,
            observation_hook=observation_hook,
            before_action_hook=before_action_hook,
            step_hook=step_hook,
            planner=planner_override,
            max_steps=max_steps,
            setup=setup,
            replay=replay,
        )

    def _open_record(self, ctx: RunContext, step: int) -> None:
        """Starts this step's record, and says which step a call belongs to.

        The marker is how a model call is attributed. The call itself is made
        several layers down, by a planner that has no reason to know which
        step it is on.
        """
        written = ctx.steps.open(step + 1)
        if self.step_opened is not None:
            self.step_opened(written)

    def _run_step(
        self,
        step: int,
        goal: str | list[str],
        planner: DevicePlanner,
        device_control,
        perception: ScreenPerception,
        session_dir: Path,
        session_trail: Optional[SessionTrail],
        observation_hook: Optional[ObservationHookFn],
        step_hook: Optional[StepHookFn],
        ctx: RunContext,
        before_action_hook: Optional[outcomes.BeforeActionHookFn] = None,
    ) -> Optional[DeviceRunResult]:
        """One turn of the loop, as the five things it does."""
        logger.info("--- [DeviceAgent] Step %d / %d ---", step + 1, self.max_steps)

        self._open_record(ctx, step)

        started = time.perf_counter()
        snapshot = observing.capture(
            self, step, device_control, perception, session_dir, session_trail, ctx
        )
        ctx.steps.observed(snapshot.current_phash, observing.elapsed_ms(started))

        arrived_as_expected = saved_path.verify_pending_arrival(self, snapshot, ctx, step)
        verification = saved_path.verify_entered_value(
            self, device_control, snapshot, ctx
        )
        requires_action_recovery = (
            verification is not None
            and verification.classification
            is postconditions.VerificationClassification.AGENT_ACTION
        )
        audit_directive = observation_hook(snapshot, step + 1) if observation_hook else None
        if audit_directive is outcomes.ObservationDirective.STOP:
            ctx.steps_executed += 1
            return outcomes.observation_stop(ctx, session_trail, snapshot.current_phash)
        if ctx.postcondition_blocked:
            ctx.steps_executed += 1
            return outcomes.postcondition_stop(ctx, session_trail, snapshot.current_phash)

        # A saved step that fits costs no planning call.
        may_replay = (
            arrived_as_expected
            and audit_directive is not outcomes.ObservationDirective.PLAN
            and not requires_action_recovery
        )
        cached_step_used = may_replay and saved_path.try_cached_step(
            self,
            snapshot,
            device_control,
            ctx,
            settle,
            session_dir,
            step,
            before_action_hook,
        )
        if ctx.observation_blocked:
            ctx.steps_executed += 1
            return outcomes.observation_stop(ctx, session_trail, snapshot.current_phash)
        if cached_step_used:
            return None

        cached_result = saved_path.finish_cached_path(self, snapshot, ctx, session_trail)
        if cached_result is not None:
            return cached_result

        ctx.steps_executed += 1
        ending = outcomes.page_stagnation(ctx, session_trail, snapshot.current_phash)
        if ending is not None:
            return ending

        started = time.perf_counter()
        step_response = self._plan(
            step, goal, planner, perception, snapshot, session_dir, session_trail, ctx
        )
        ctx.steps.planned(step_response, observing.elapsed_ms(started))

        ending = self._ending_after_planning(step_response, snapshot, session_trail, step_hook, ctx)
        if ending is not None:
            saved_path.describe_terminal_screen(self, step_response.current_screen)
            return ending

        if isinstance(step_response.action, ShowKeyboardTargetsAction):
            return keyboard_targets.expand(
                step,
                step_response.action,
                snapshot,
                session_trail,
                ctx,
                self.grounding,
            )

        decision = targeting.evaluate(
            step_response.action,
            snapshot,
            device_control,
            self.grounding,
            self.grounder,
        )
        ctx.steps.record_grounding(self.grounding, decision, step_response.action)
        if not decision.ok:
            return self._blocked(
                step,
                step_response.action,
                decision,
                snapshot,
                device_control,
                session_dir,
                session_trail,
                ctx,
            )
        action = decision.action
        if action is None:
            raise RuntimeError("Targeting approved an action without resolving it.")

        if before_action_hook is not None:
            directive = before_action_hook(action, step + 1)
            if directive is outcomes.ObservationDirective.STOP:
                return outcomes.observation_stop(ctx, session_trail, snapshot.current_phash)

        return self._act(
            step,
            action,
            step_response,
            device_control,
            snapshot,
            session_dir,
            session_trail,
            ctx,
            decision.target_bounds,
        )

    def _plan(
        self, step, goal, planner, perception, snapshot, session_dir, session_trail, ctx
    ) -> DeviceStepResponse:
        """Numbers the targets, asks for an action, and records what came back."""
        # Saved before the model is called, so a planning failure still leaves
        # the image the planner was given.
        observation.number_targets(snapshot, ctx, self.grounding, perception)
        ctx.steps.files({"marked": self.artifacts.save_planner_view(session_dir, step, snapshot)})
        refinement_files = self.artifacts.save_refinement(session_dir, step, snapshot)
        if isinstance(refinement_files, dict):
            ctx.steps.files(refinement_files)

        nudge, ctx.pending_loop_nudge = ctx.pending_loop_nudge, None
        step_response = self._plan_step(
            planner=planner,
            goal=goal,
            snapshot=snapshot,
            ctx=ctx,
            session_trail=session_trail,
            loop_nudge=nudge,
        )
        observing.apply_plan_result(self, step_response, snapshot, session_trail, ctx)
        return step_response

    def _ending_after_planning(
        self, step_response, snapshot, session_trail, step_hook, ctx
    ) -> Optional[DeviceRunResult]:
        """The three ways a plan ends the run instead of reaching the device."""
        if step_hook:
            stopped = outcomes.hook_stop(
                step_hook,
                step_response,
                snapshot,
                session_trail,
                snapshot.current_phash,
                ctx,
            )
            if stopped is not None:
                return stopped

        done = outcomes.goal_reached(
            self.stop_when_satisfied,
            step_response,
            session_trail,
            snapshot.current_phash,
            ctx,
        )
        if done is not None:
            return done

        ctx.pending_loop_nudge = ctx.loop_detector.check(
            RollingLoopDetector.hash_action(step_response.action), None
        )
        return outcomes.action_repetition(ctx, session_trail, snapshot.current_phash)

    def _blocked(
        self, step, action, decision, snapshot, device_control, session_dir, session_trail, ctx
    ) -> Optional[DeviceRunResult]:
        """An action that never reached the device, and what that costs the screen."""
        action_description = narration.describe(action)
        targeting.record_unexecuted(ctx.history, step, action_description, decision.observation)
        ctx.steps.record_blocked_action(
            action,
            action_description,
            decision.observation,
            target_label(action, snapshot.gui_elements),
        )
        if decision.unmatched:
            artifacts.save_unmatched_screen(
                self.images,
                session_dir / f"step{step + 1:03d}_{snapshot.timestamp}_not_listed.png",
                snapshot,
            )
            region = getattr(action, "region", None)
            key = (snapshot.current_phash or "", decision.target or "", region or "")
            if (
                region
                and not ctx.keyboard_open
                and snapshot.current_phash
                and key not in ctx.refined_targets
            ):
                ctx.refined_targets.add(key)
                ctx.pending_visual_refinement = (snapshot.current_phash, region)
                logger.info(
                    "Will inspect the %s region more closely on the next look.", region
                )
        if stall.reveal_behind_keyboard(device_control, snapshot, ctx):
            return None
        ctx.stall.record_blocked(snapshot.current_phash)
        if ctx.stall.is_exhausted:
            return outcomes.loop_detected(
                ctx,
                session_trail,
                snapshot.current_phash,
                targeting.stalled_reason(decision),
            )
        return None

    def _act(
        self,
        step,
        action,
        step_response,
        device_control,
        snapshot,
        session_dir,
        session_trail,
        ctx,
        target_bounds=None,
    ) -> Optional[DeviceRunResult]:
        """Touches the device, and decides whether the screen is going anywhere."""
        if action.type is ActionType.TYPE:
            action, target_bounds = ground_and_protect_typed_secret(
                action,
                snapshot,
                self.scrubber,
                self.images,
                self.grounding_models,
                target_bounds,
            )
        ctx.steps.files({"action": self.artifacts.save_action(session_dir, step, snapshot, action)})
        started = time.perf_counter()
        ctx.last_external_nav = dispatch.execute(
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
            requested_action=step_response.action,
        )
        ctx.steps.record_action_duration(observing.elapsed_ms(started))
        saved_path.record_planned_step(self, action, snapshot, ctx, step_response.current_screen)
        if ctx.stall.is_blocked and stall.reveal_behind_keyboard(device_control, snapshot, ctx):
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
        planner_elements = (
            getattr(snapshot, "planner_elements", None)
            if getattr(snapshot, "planner_elements", None) is not None
            else snapshot.gui_elements
        )
        planner_height = getattr(snapshot, "planner_screen_height", None) or ctx.screen_height
        gui_elements = keyboard.annotate_keyboard_keys(
            planner_elements, ctx.keyboard_open, planner_height
        )
        view_note = getattr(snapshot, "view_note", None)
        if view_note:
            loop_nudge = "\n".join(filter(None, [view_note, loop_nudge]))
        perception_frame = getattr(snapshot, "planner_perception_frame", None) or getattr(
            snapshot, "perception_frame", None
        )
        screen_text = perception_frame.unmatched_text() if perception_frame is not None else ()
        ocr_evidence = perception_frame.ocr_redaction_evidence() if perception_frame is not None else ()
        return planner.plan_step(
            goal=goal,
            gui_elements=gui_elements,
            history=ctx.history.format_for_prompt(),
            trail=session_trail.format_for_prompt() if session_trail else None,
            drift_warning=ctx.drift_warning,
            loop_nudge=loop_nudge,
            current_image_bytes=snapshot.planner_img_bytes,
            marks=snapshot.marks,
            screen_height=planner_height or 0,
            keyboard_open=ctx.keyboard_open,
            screen_text=screen_text,
            ocr_evidence=ocr_evidence,
        )
