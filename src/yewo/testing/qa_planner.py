"""Plans the actions for one QA scenario.

The planner receives the setup state, the scenario instructions, the current
screen and the action history. It does not receive the expected outcomes or
the security rules. A planner that reads an assertion can satisfy the
assertion instead of the scenario, and its own verdicts reach no report.

QAAuditor holds the rules and never reads the plan. Neither side can arrange
its own evidence.
"""

import logging
from typing import List, Optional

from pydantic import Field, model_validator

from yewo.agent.device.launch import SetupMode
from yewo.agent.plan.planner import (
    DRIFT_SECTION,
    DescribeDeviceAction,
    DeviceAction,
    DevicePlanner,
    DeviceStepResponse,
    GROUNDING_COORDINATES,
    GROUNDING_DESCRIBE,
    GROUNDING_MARKS,
    KEYBOARD_HIDDEN,
    KEYBOARD_VISIBLE,
    LOOP_SECTION,
    MarksDeviceAction,
    TRAIL_SECTION,
)
from yewo.llm.llm_client import LLMClient

logger = logging.getLogger(__name__)


PROMPT = """You control a mobile app for one test scenario.
Execute the scenario instructions in order. You do not judge the result.
Read the screen in front of you to find where you are. Do not assume a screen.
"""

SCENARIO_SECTION = """
{setup}

### Scenario Instructions
{instructions}
"""

CURRENT_STATE_SECTION = """
### Current UI Elements Detected {heading}
{targets}
{keyboard}
### Action History (with system-observed outcomes)
{history}
"""

FEASIBILITY = """
### Feasibility
A target that is absent from this screen does not make the scenario impossible.
Navigate to the screen that holds it.

Set `is_scenario_feasible` false only for a scenario this app cannot run. An
instruction contradicts the app, or a target does not exist in the app.

Do not set it false because the app did something unexpected. An app that does
the wrong thing is the finding. If that blocks the remaining instructions, keep
`is_scenario_feasible` true. Select `done`.

Leave the screen as the app left it. Do not navigate after the last
instruction. Do not undo what an instruction did. The final screen is the
evidence.

Do not grade the app. Another component judges every outcome and every rule.
That component does not read your decisions.
"""

RESPONSE_INSTRUCTIONS = """
### Response Instructions
1. Set `current_screen` from the screenshot in front of you.
2. Set `observation` to the visible state only. Do not plan here.
3. Set `evaluation_previous_goal` to what the last action did. Leave it empty
   on the first step.
4. Set `reasoning` to the next instruction that is not complete.
5. Set `action` to one action that advances that instruction. An instruction
   may need several actions.
6. Keep `goal_satisfied` false while any instruction is not complete.
7. Set `goal_satisfied` true after the history shows the last instruction ran.
   Select `done`. Do not read completion from the session trail. Do not read it
   from an earlier scenario.
8. Set `goal_evidence` to the visible state the app ended on.
9. Set `external_navigation` true only when the action leaves the app.
10. Set `infeasibility_reason` only when you set `is_scenario_feasible` false.
"""


class QAStepResponse(DeviceStepResponse):
    """What the planner answers with: an action, and whether it can proceed.

    The verdicts belong to yewo.testing.judge.qa_auditor, which judges the screen
    without reading the plan.

    Four inherited descriptions are overridden here. A field description is
    sent to the model inside the JSON schema, so one that asks the planner to
    judge an outcome undoes a prompt that tells it not to. `yewo run` drives
    a goal and does judge its own, which is why the base keeps that wording.
    """

    goal_satisfied: bool = Field(
        default=False,
        description=(
            "True only after every scenario instruction ran. Do not judge whether "
            "the app behaved correctly. Something else judges that."
        ),
    )
    goal_evidence: str = Field(
        default="",
        description=("The visible state the app ended on. Empty while `goal_satisfied` is false."),
    )
    action: DeviceAction = Field(
        ...,
        description=(
            "The one action to execute next. Select `done` after the last instruction "
            "ran, or when the app blocks the instructions that remain."
        ),
    )
    is_scenario_feasible: bool = Field(
        default=True,
        description=(
            "False only for a scenario this app cannot run. An instruction contradicts "
            "the app, or a target does not exist in the app. A screen you did not reach "
            "yet is not a reason. An app that did something unexpected is not a reason "
            "either. Keep this true and select `done`."
        ),
    )
    infeasibility_reason: Optional[str] = Field(
        default=None,
        description="Detailed reason why the scenario is infeasible. None if feasible.",
    )

    @model_validator(mode="after")
    def _validate_infeasibility_reason(self) -> "QAStepResponse":
        if not self.is_scenario_feasible and not (self.infeasibility_reason or "").strip():
            raise ValueError("infeasibility_reason must explain why the scenario is infeasible.")
        return self


class MarksQAStepResponse(QAStepResponse):
    """The QA step response used in marks mode, where targets have no coordinates."""

    action: MarksDeviceAction = Field(
        ..., description="The single next action, naming its target by number."
    )


class DescribeQAStepResponse(QAStepResponse):
    """The QA step response used in describe mode, where targets are named in words."""

    action: DescribeDeviceAction = Field(
        ..., description="The single next action, naming its target in words."
    )


class QAPlanner(DevicePlanner):
    """
    Plans the actions for one QA scenario. It does not read the assertions.
    Instantiated per-scenario by QAAgent, carrying scenario-specific context.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        setup: SetupMode,
        scrubber=None,
        grounding: str = GROUNDING_COORDINATES,
    ):
        super().__init__(llm_client, grounding, scrubber=scrubber)
        # The setup state, and not the scenario. A GeneratedTestCase carries
        # the assertions, so holding one would put them a prompt edit away.
        self._setup = setup

    def step_response_model(self):
        """The QA schema the model answers in, matched to the grounding mode.

        Every mode needs its own, because the QA fields sit on top of an action
        whose target the mode decides.
        """
        if self.grounding == GROUNDING_DESCRIBE:
            return DescribeQAStepResponse
        if self.grounding == GROUNDING_MARKS:
            return MarksQAStepResponse
        return QAStepResponse

    def plan_step(
        self,
        goal: List[str] | str,
        gui_elements=None,
        history: str = "",
        trail: Optional[str] = None,
        drift_warning: Optional[str] = None,
        loop_nudge: Optional[str] = None,
        current_image_bytes: bytes = b"",
        keyboard_open: Optional[bool] = None,
        marks: Optional[list] = None,
        screen_height: int = 0,
        screen_text: tuple[str, ...] = (),
        ocr_evidence=(),
    ) -> QAStepResponse:
        from yewo.utils.redactor import safe_text

        if isinstance(goal, str):
            goal_lines = [goal]
        else:
            goal_lines = goal or []

        stable_prompt, current_state = self._build_qa_prompt_sections(
            goal_lines=goal_lines,
            gui_elements=gui_elements,
            history=history,
            trail=trail,
            drift_warning=drift_warning,
            loop_nudge=loop_nudge,
            keyboard_open=keyboard_open,
            marks=marks,
            screen_height=screen_height,
            screen_text=screen_text,
        )

        messages = self._model_messages(
            stable_prompt,
            current_state,
            current_image_bytes,
            gui_elements,
            ocr_evidence,
        )

        logger.info("Querying QAPlanner for the next action...")
        response: QAStepResponse = self.llm.generate_response(messages, self.step_response_model())

        def clean(text):
            return safe_text(text, self._scrubber)

        response.reasoning = clean(response.reasoning)
        response.observation = clean(response.observation)
        response.evaluation_previous_goal = clean(response.evaluation_previous_goal)
        response.infeasibility_reason = clean(response.infeasibility_reason)

        return response

    def _build_qa_prompt(
        self,
        goal_lines: List[str],
        gui_elements,
        history: str,
        trail: Optional[str],
        drift_warning: Optional[str],
        loop_nudge: Optional[str],
        keyboard_open: Optional[bool] = None,
        marks: Optional[list] = None,
        screen_height: int = 0,
        screen_text: tuple[str, ...] = (),
    ) -> str:
        stable_prompt, current_state = self._build_qa_prompt_sections(
            goal_lines=goal_lines,
            gui_elements=gui_elements,
            history=history,
            trail=trail,
            drift_warning=drift_warning,
            loop_nudge=loop_nudge,
            keyboard_open=keyboard_open,
            marks=marks,
            screen_height=screen_height,
            screen_text=screen_text,
        )
        return stable_prompt + current_state

    def _build_qa_prompt_sections(
        self,
        goal_lines: List[str],
        gui_elements,
        history: str,
        trail: Optional[str],
        drift_warning: Optional[str],
        loop_nudge: Optional[str],
        keyboard_open: Optional[bool] = None,
        marks: Optional[list] = None,
        screen_height: int = 0,
        screen_text: tuple[str, ...] = (),
    ) -> tuple[str, str]:
        stable_prompt = (
            PROMPT
            + FEASIBILITY
            + self._action_vocabulary()
            + self._action_constraints()
            + RESPONSE_INSTRUCTIONS
            + SCENARIO_SECTION.format(
                setup=self._setup.describe(),
                instructions="\n".join(f"- {step}" for step in goal_lines),
            )
        )
        current_state = CURRENT_STATE_SECTION.format(
            heading=self._target_heading(),
            targets=self.format_targets(gui_elements, marks, screen_height),
            keyboard=(
                ""
                if keyboard_open is None
                else (KEYBOARD_VISIBLE if keyboard_open else KEYBOARD_HIDDEN)
            ),
            history=history or "No actions executed yet.",
        )
        if screen_text:
            current_state += self._format_screen_text(screen_text)
        if drift_warning:
            current_state += DRIFT_SECTION.format(drift=drift_warning)
        if trail:
            current_state += TRAIL_SECTION.format(trail=trail)
        if loop_nudge:
            current_state += LOOP_SECTION.format(nudge=loop_nudge)
        return stable_prompt, current_state
