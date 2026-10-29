"""Stateless helpers used while preparing one QA scenario execution."""

from yewo.defaults import STEP_BUDGET_FLOOR, STEP_BUDGET_PER_INSTRUCTION
from yewo.testing.models import GeneratedTestCase, ReportStatus


def scenario_step_budget(test_case: GeneratedTestCase, ceiling: int) -> int:
    wanted = len(test_case.steps_instructions) * STEP_BUDGET_PER_INSTRUCTION
    return min(ceiling, max(STEP_BUDGET_FLOOR, wanted))


def build_scenario_replay(trace_store, test_case: GeneratedTestCase):
    if trace_store is None:
        return None
    identity = trace_store.resolve_identity(test_case)
    return trace_store.build_replay(identity) if identity is not None else None


def save_scenario_trace(trace_store, test_case, replay, status: ReportStatus) -> None:
    if trace_store is not None and replay is not None:
        trace_store.save_trace(test_case, replay, status)
