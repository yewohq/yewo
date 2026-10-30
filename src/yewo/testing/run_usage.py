"""Run component attribution and scenario execution cost accounting."""

from typing import Optional

from yewo.agent.run.record import CallRow
from yewo.discovery.model_proposer import ControlChoice, GoalJudgement
from yewo.testing.models import ScenarioReport
from yewo.testing.warm_execution import ExecutionCost

GENERATION = "generator"
DISCOVERY = "discovery"
PLANNER = "planner"
SCREEN_AUDITOR = "screen_auditor"
TRANSITION_JUDGE = "transition_judge"
SEMANTIC_GROUNDER = "semantic_grounder"
SCENARIO_COMPONENTS = (PLANNER, SCREEN_AUDITOR, TRANSITION_JUDGE, SEMANTIC_GROUNDER)

DISCOVERY_ANSWERS = (ControlChoice.__name__, GoalJudgement.__name__)
"""The shapes only bounded discovery asks for.

Named from the models themselves, so the record cannot drift from the thing it
describes. Discovery runs before any scenario and borrows the generation
callback, so without this its spend is filed under generation.
"""


def component_for(current: str, answering: Optional[str]) -> str:
    """Name a call from the answer shape it actually requested."""
    if answering in DISCOVERY_ANSWERS:
        return DISCOVERY
    if current == GENERATION:
        return GENERATION
    if answering == "ScreenAssessment":
        return SCREEN_AUDITOR
    if answering == "TransitionVerdicts":
        return TRANSITION_JUDGE
    return PLANNER


def record_scenario_execution_cost(
    report: ScenarioReport,
    scenario_level_calls: list[CallRow],
) -> None:
    """Attach semantic execution cost without mixing it into navigation cost."""
    warm_execution = report.warm_execution
    if warm_execution is None:
        return
    calls = [
        *scenario_level_calls,
        *(call for step in report.steps for call in step.calls),
    ]
    known_cost = sum(call.cost for call in calls if call.cost is not None)
    scenario_cost = ExecutionCost(
        duration_ms=max(0, round(report.duration * 1000)),
        actions=sum(step.action is not None for step in report.steps),
        model_calls=len(calls),
        cost_micro_usd=max(0, round(known_cost * 1_000_000)),
        cost_unknown_calls=sum(call.cost is None for call in calls),
    )
    report.warm_execution = warm_execution.model_copy(
        update={"costs": warm_execution.costs.model_copy(update={"scenario": scenario_cost})}
    )
