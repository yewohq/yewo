"""Deterministic verdicts for measured expected outcomes."""

from collections.abc import Iterable, Sequence

from yewo.metrics import MetricDirection, definition_for
from yewo.testing.judge import rules
from yewo.testing.models import AssertionScope, ExpectedOutcome, Measurement, RuleVerification


def _worst_value(outcome: ExpectedOutcome, samples: Iterable[Measurement]) -> float | None:
    values = [sample.value for sample in samples if sample.metric is outcome.metric]
    if not values or outcome.metric is None:
        return None
    direction = definition_for(outcome.metric).direction
    return max(values) if direction is MetricDirection.LOWER else min(values)


def settle(
    expected_outcomes: Sequence[ExpectedOutcome | str],
    samples: Iterable[Measurement],
) -> list[RuleVerification]:
    """Judge each measured outcome from its worst relevant sample.

    No sample produces no verdict. The ordinary evidence gate then reports the
    required outcome as unverified instead of manufacturing a pass.
    """
    sample_list = list(samples)
    verdicts: list[RuleVerification] = []
    for index, outcome in enumerate(expected_outcomes):
        if not isinstance(outcome, ExpectedOutcome):
            continue
        if outcome.scope is not AssertionScope.MEASURED:
            continue
        worst = _worst_value(outcome, sample_list)
        if worst is None or outcome.metric is None or outcome.threshold is None:
            continue
        definition = definition_for(outcome.metric)
        passed = (
            worst <= outcome.threshold
            if definition.direction is MetricDirection.LOWER
            else worst >= outcome.threshold
        )
        comparison = "at most" if definition.direction is MetricDirection.LOWER else "at least"
        verdicts.append(
            RuleVerification(
                rule=outcome.statement,
                rule_id=rules.rule_id(rules.EXPECTED, index),
                status="passed" if passed else "failed",
                reasoning=(
                    f"Worst sample was {worst:g} {definition.unit.value}; "
                    f"required {comparison} {outcome.threshold:g} {definition.unit.value}."
                ),
            )
        )
    return verdicts


def measured_rule_ids(
    expected_outcomes: Sequence[ExpectedOutcome | str],
) -> set[str]:
    """The rule IDs owned by this deterministic judge."""
    return {
        rules.rule_id(rules.EXPECTED, index)
        for index, outcome in enumerate(expected_outcomes)
        if isinstance(outcome, ExpectedOutcome)
        and outcome.scope is AssertionScope.MEASURED
    }
