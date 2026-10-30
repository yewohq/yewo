"""Where the journey's intent is applied to a real screen.

Discovery's controller decides what is legal. This decides what is promising.
It asks a model two questions about the screen in front of it. Does this
screen satisfy an open goal? Which offered control moves toward one?

Neither answer is evidence. The controller re-derives the authorization for
whatever control is named and refuses a name it did not offer. It also refuses
a goal claim whose observation or transition it did not accept. A judgement
travels with `evidence_owner="model_proposal"`, so a reader can separate a
judged capability from a measured one.

One object answers both because both need the same two things: the open goals,
and what this screen reads as. Its spend is reported on the next proposal, so
the model-call and cost budgets account for the judging too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel, Field

from yewo.agent.actuation.action_models import PlannerTapAction, TapAction
from yewo.agent.perception.fusion import PerceptionFrame
from yewo.discovery.contracts import DiscoveryRun, JourneyGoal
from yewo.discovery.runtime import DiscoveryActionCandidate, DiscoveryProposalReceipt
from yewo.discovery.system_chrome import is_inside_system_chrome
from yewo.llm.llm_client import STRUCTURED_ATTEMPTS, LLMClient
from yewo.llm.messages import TextMessage
from yewo.utils.secrets import SecretScrubber

MODEL_CALLS_PER_PROPOSAL = 2 * STRUCTURED_ATTEMPTS
"""One judgement and one choice per screen, each retried once on a bad shape.

The controller reserves this much before every proposal, so the ceiling covers
the judging that preceded it as well as the choice itself. A structured call
reports usage per attempt that reaches the provider, so the retry counts.
"""

MAXIMUM_COST_MICRO_USD = 20_000
"""Two cents a screen. A proposer that exceeds it ends the run on cost."""

AnswerT = TypeVar("AnswerT", bound=BaseModel)

UNNAMED_CONTROL = "(no text)"
"""What an offered control reads as when OCR attached nothing to it."""


class GoalJudgement(BaseModel):
    """Which listed goals this screen satisfies, by their listed numbers.

    Public because the usage record names the discovery lane from the shape a
    call requested, the way it names every other lane.
    """

    satisfied_goals: list[int] = Field(default_factory=list)
    reason: str = ""


class ControlChoice(BaseModel):
    """Which listed control to act on next, by its listed number."""

    control: int | None = None
    reason: str = ""


@dataclass
class ProposalSpend:
    """Provider-reported model spend not yet charged to the discovery run."""

    model_calls: int = 0
    micro_usd: int = 0

    def record(self, call) -> None:
        self.model_calls += 1
        self.micro_usd += round((call.cost or 0.0) * 1_000_000)

    def drain(self) -> tuple[int, int]:
        spent = (self.model_calls, self.micro_usd)
        self.model_calls = 0
        self.micro_usd = 0
        return spent


class ModelDiscoveryProposer:
    """Judges a screen against the open goals, and chooses the next control."""

    model_calls_per_proposal = MODEL_CALLS_PER_PROPOSAL
    maximum_cost_micro_usd = MAXIMUM_COST_MICRO_USD

    def __init__(
        self,
        llm_client: LLMClient,
        perception,
        scrubber: SecretScrubber,
        spend: ProposalSpend,
    ):
        self.llm = llm_client
        self.perception = perception
        self.scrubber = scrubber
        self.spend = spend
        self._provider_failed = False

    def satisfied_goal_ids(
        self,
        goals: tuple[JourneyGoal, ...],
        frame: PerceptionFrame,
        chrome: tuple[tuple[int, int, int, int], ...] = (),
    ) -> tuple[str, ...]:
        """Which of these goals this screen satisfies, as the model reads it."""
        if not goals:
            return ()
        screen = self._screen_text(frame, chrome)
        if not screen:
            return ()
        answer = self._ask(
            _goal_prompt(self._scrubbed(goals), screen),
            GoalJudgement,
        )
        if answer is None:
            return ()
        return tuple(
            goals[number - 1].goal_id
            for number in dict.fromkeys(answer.satisfied_goals)
            if 1 <= number <= len(goals)
        )

    def select(
        self,
        run: DiscoveryRun,
        candidates: tuple[DiscoveryActionCandidate, ...],
    ) -> DiscoveryProposalReceipt:
        """The offered control that moves toward an open goal, or none."""
        goals = tuple(goal for goal in run.goals if not goal.is_terminal)
        labels = self._labels(candidates)
        if not candidates or not goals or labels is None:
            return self._receipt(None)
        answer = self._ask(
            _choice_prompt(self._scrubbed(goals), labels),
            ControlChoice,
        )
        if answer is None:
            return self._receipt(None, provider_failed=self._provider_failed)
        if answer.control is None:
            return self._receipt(None)
        if not 1 <= answer.control <= len(candidates):
            return self._receipt(None)
        return self._receipt(candidates[answer.control - 1].frontier_id)

    def _receipt(
        self, frontier_id: str | None, *, provider_failed: bool = False
    ) -> DiscoveryProposalReceipt:
        """Report the exact pending spend with the selected frontier entry."""
        model_calls, micro_usd = self.spend.drain()
        return DiscoveryProposalReceipt(
            selected_frontier_id=frontier_id,
            model_calls=model_calls,
            cost_micro_usd=micro_usd,
            provider_failed=provider_failed,
        )

    def drain_unreported_spend(self) -> tuple[int, int]:
        """Drain spend accumulated after the final proposal receipt."""
        return self.spend.drain()

    def _ask(self, prompt: str, answer_model: type[AnswerT]) -> AnswerT | None:
        self._provider_failed = False
        try:
            return self.llm.generate_response(
                messages=[TextMessage(content=prompt)],
                response_model=answer_model,
            )
        except Exception:  # noqa: BLE001 - the provider is an external boundary
            self._provider_failed = True
            return None

    def _scrubbed(self, goals: tuple[JourneyGoal, ...]) -> tuple[str, ...]:
        return tuple(self.scrubber.scrub(goal.statement) for goal in goals)

    def _screen_text(
        self,
        frame: PerceptionFrame,
        chrome: tuple[tuple[int, int, int, int], ...],
    ) -> tuple[str, ...]:
        """Every line the application drew, with declared secrets replaced.

        The platform's own regions are left out. A status bar clock otherwise
        changes this prompt every minute. Two readings of one screen are then
        two different questions, nothing is cached between them, and no two
        runs of one journey can be compared.
        """
        return tuple(
            scrubbed
            for line in frame.text_lines
            if not is_inside_system_chrome(line.bounds, chrome)
            and (scrubbed := self.scrubber.scrub(line.text).strip())
        )

    def _labels(
        self, candidates: tuple[DiscoveryActionCandidate, ...]
    ) -> tuple[str, ...] | None:
        """What each offered control reads as, or nothing if the frame moved on.

        A candidate is a point. The frame that named it is the one the runtime
        just captured. A candidate centred on no detection here means this
        frame is not that frame. No choice made against it would address the
        control it names. A candidate carrying an action with no point cannot
        be read off a screen at all, and is refused for the same reason.
        """
        frame = getattr(self.perception, "current_frame", None)
        if frame is None:
            return None
        labels = []
        for candidate in candidates:
            action = candidate.action
            if not isinstance(action, TapAction | PlannerTapAction):
                return None
            text = _text_at(frame, action.coordinates)
            if text is None:
                return None
            labels.append(self.scrubber.scrub(text).strip() or UNNAMED_CONTROL)
        return tuple(labels)


def _text_at(frame: PerceptionFrame, point) -> str | None:
    """What the detection this action addresses reads as.

    The detection centred on this point, because a candidate's point is its
    own control's centre. Reading the smallest detection that merely contains
    the point lets a wide band borrow a name. An app bar then reads as the
    title it wraps. The title answers the goal, so the model picks the bar,
    and the tap lands on a heading that does nothing.

    Smallest among the detections sharing that centre, because detections nest
    and the innermost is the one an action addresses.
    """
    centred = [
        element
        for element in frame.elements
        if _centre(element.bounding_box) == (point.x, point.y)
    ]
    if not centred:
        return None
    return min(centred, key=lambda element: _area(element.bounding_box)).text


def _centre(box) -> tuple[int, int]:
    return (box.x_min + box.x_max) // 2, (box.y_min + box.y_max) // 2


def _area(box) -> int:
    return max(0, box.x_max - box.x_min) * max(0, box.y_max - box.y_min)


def _numbered(items: tuple[str, ...]) -> str:
    return "\n".join(f"{number}. {item}" for number, item in enumerate(items, 1))


def _goal_prompt(goals: tuple[str, ...], screen: tuple[str, ...]) -> str:
    return (
        "You are reading one screen of a mobile application.\n\n"
        "These goals are open:\n"
        f"{_numbered(goals)}\n\n"
        "This screen reads as:\n"
        f"{_numbered(screen)}\n\n"
        "List the number of every open goal this screen shows to be complete. "
        "List a goal only when this screen is evidence that it finished. "
        "A screen on the way to a goal does not complete it. "
        "Answer with an empty list when none is complete."
    )


def _choice_prompt(goals: tuple[str, ...], controls: tuple[str, ...]) -> str:
    return (
        "You are exploring a mobile application to find out whether it can do "
        "what a journey asks.\n\n"
        "These goals are open:\n"
        f"{_numbered(goals)}\n\n"
        "These controls are offered on the current screen:\n"
        f"{_numbered(controls)}\n\n"
        "Answer with the number of the one control most likely to move toward "
        "an open goal. Answer with null when no offered control moves toward "
        "any open goal."
    )
