"""AndroidWorld agent that drives its emulator through Yewo."""

import hashlib
from pathlib import Path

from accessibility import ensure_forwarder_bound
from yewo.agent.device.launch import SetupMode
from yewo.agent.device.scope import DeviceScope
from yewo.agent.device.session import DeviceSession
from yewo.agent.grounding import GROUNDING_COORDINATES
from yewo.agent.perception.blind import for_grounding
from yewo.agent.run.agent import DeviceAgent
from yewo.agent.run.outcomes import GOAL_REACHED, DeviceRunResult
from yewo.agent.run.task import DeviceTask
from yewo.llm.prompts import PromptStore
from yewo.preflight import require_run
from yewo.testing.report.usage_tracker import LLMUsageTracker

from android_world.agents import base_agent
from android_world.env import adb_utils

AGENT_COMPONENT = "agent"


class YewoAndroidWorldAgent(base_agent.EnvironmentInteractingAgent):
    """Advance Yewo once for each AndroidWorld interaction."""

    def __init__(
        self,
        env,
        device_name: str,
        output_dir: Path,
        model: str,
        fast_model: str | None = None,
        grounding: str = GROUNDING_COORDINATES,
        max_steps: int = 40,
        settle_time: float = 2.0,
    ):
        super().__init__(env, name="yewo", transition_pause=0)
        require_run(model, fast_model, grounding)
        self._session = DeviceSession(
            device_name=device_name,
            app_path=None,
            record_session=False,
            session_dir=output_dir,
            keep_device=True,
            reuse_app=True,
            scope=DeviceScope.DEVICE,
        )
        self._agent = DeviceAgent(
            model=model,
            fast_model=fast_model,
            grounding=grounding,
            max_steps=max_steps,
            settle_time=settle_time,
            stop_when_satisfied=True,
        )
        self._task: DeviceTask | None = None
        self._goal: str | None = None
        self._completed_run: DeviceRunResult | None = None
        self._step_budget: int | None = None
        self._is_closed = False
        try:
            self._device_control = self._session.start()
        except BaseException:
            self._session.close(interrupted=True)
            raise
        self._perception = for_grounding(grounding, self._session)
        self._usage = LLMUsageTracker(
            output_dir=self._session.telemetry_dir,
            scrubber=self._agent.scrubber,
        )
        self._agent.llm.usage_callback = self._record_call
        if self._agent.fast_llm is not self._agent.llm:
            self._agent.fast_llm.usage_callback = self._record_call
        self._agent.step_opened = self._usage.mark
        self._agent.record_prompts(
            PromptStore(self._session.prompts_dir, scrub=self._agent.scrubber.scrub)
        )

    def _record_call(self, call) -> None:
        self._usage.add_call(AGENT_COMPONENT, call)

    def _run_adb(self, arguments) -> str:
        response = adb_utils.issue_generic_request(arguments, self.env.controller)
        output = response.generic.output
        return output.decode() if isinstance(output, bytes) else str(output)

    def set_max_steps(self, max_steps: int) -> None:
        """Adopt AndroidWorld's per-task budget as Yewo's own step limit.

        AndroidWorld allocates the budget from task complexity and ends the
        episode at that number. A lower ceiling inside Yewo stops the task
        before the benchmark would, which records a step limit the budget
        never set.
        """
        super().set_max_steps(max_steps)
        self._step_budget = max_steps

    def reset(self, go_home: bool = False) -> None:
        """Discard Yewo's task memory after AndroidWorld resets the device."""
        if self._task is not None:
            self._task.close()
        self._task = None
        self._goal = None
        self._completed_run = None
        self._perception.reset()
        # Reinstalling the forwarder can leave its component enabled but not
        # bound. Recover it only when needed: toggling a healthy service would
        # unnecessarily interrupt AndroidWorld's active gRPC connection.
        ensure_forwarder_bound(self._run_adb)
        super().reset(go_home=go_home)

    def _task_for(self, goal: str) -> DeviceTask:
        """The open task for this goal, started when the goal is a new one."""
        if self._task is not None and self._goal == goal:
            return self._task
        if self._task is not None:
            self._task.close()
        self._goal = goal
        self._completed_run = None
        task_id = hashlib.sha256(goal.encode("utf-8")).hexdigest()[:16]
        self._task = self._agent.start_task(
            goal=goal,
            device_control=self._device_control,
            perception=self._perception,
            session_dir=self._session.screenshots_dir / task_id,
            max_steps=self._step_budget,
            setup=SetupMode.PRESERVE,
            replay=False,
        )
        return self._task

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        # AndroidWorld enables developer overlays for gesture diagnostics. Its
        # reference agents clear that UI before every observation, including
        # after task resets. Yewo must receive the same uncontaminated pixels.
        self.env.hide_automation_ui()
        task = self._task_for(goal)
        run = task.step()
        if run is not None and self._completed_run is None:
            self._completed_run = run
            ensure_forwarder_bound(self._run_adb)
        self._usage.checkpoint()
        return base_agent.AgentInteractionResult(
            # In AndroidWorld, suite_utils scores a task if and only if interaction_results.done is True:
            #   agent_successful = task_successful if interaction_results.done else 0.0
            # If the agent reached the goal (GOAL_REACHED) or exhausted its step budget ("step_limit")
            # after executing all budgeted actions, done must be reported as True so AndroidWorld's
            # ground-truth evaluator (task.is_successful(env)) evaluates the device state. If the goal
            # was not actually achieved, task.is_successful(env) returns 0.0 anyway.
            # Abortive failures like loop_detected or crashes remain done=False.
            done=run is not None and run.status in (GOAL_REACHED, "step_limit"),
            data={
                "yewo_status": run.status if run else "running",
                "yewo_steps": task.context.steps_executed,
                "yewo_failure_reason": run.failure_reason if run else None,
                "yewo_usage": self._usage.summary(),
            },
        )

    def close(self) -> None:
        """Detach Yewo without resetting AndroidWorld's emulator or apps."""
        if self._is_closed:
            return
        self._is_closed = True
        if self._task is not None:
            self._task.close()
        self._usage.save_report()
        self._session.close()
