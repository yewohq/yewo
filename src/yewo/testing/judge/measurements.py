"""Deterministic verdicts for measured expected outcomes."""

from collections.abc import Iterable, Sequence

from yewo.metrics import MeasurementRelation, MetricDirection, definition_for
from yewo.testing.judge import rules
from yewo.testing.models import AssertionScope, ExpectedOutcome, Measurement, RuleVerification


def _proves(sample: Measurement, threshold: float, direction: MetricDirection) -> bool | None:
    """Return the verdict this exact value or bound proves."""
    if direction is MetricDirection.LOWER:
        if sample.relation is MeasurementRelation.EXACT:
            return sample.value <= threshold
        if sample.relation is MeasurementRelation.AT_MOST and sample.value <= threshold:
            return True
        if sample.relation is MeasurementRelation.AT_LEAST and sample.value > threshold:
            return False
        return None
    if sample.relation is MeasurementRelation.EXACT:
        return sample.value >= threshold
    if sample.relation is MeasurementRelation.AT_LEAST and sample.value >= threshold:
        return True
    if sample.relation is MeasurementRelation.AT_MOST and sample.value < threshold:
        return False
    return None


def _decide_samples(
    outcome: ExpectedOutcome,
    samples: Iterable[Measurement],
) -> tuple[bool, Measurement] | None:
    if outcome.metric is None or outcome.threshold is None:
        return None
    relevant = [sample for sample in samples if sample.metric is outcome.metric]
    if not relevant:
        return None
    direction = definition_for(outcome.metric).direction
    decisions = [(sample, _proves(sample, outcome.threshold, direction)) for sample in relevant]
    failed = [sample for sample, decision in decisions if decision is False]
    if failed:
        selected = (
            max(failed, key=lambda sample: sample.value)
            if direction is MetricDirection.LOWER
            else min(failed, key=lambda sample: sample.value)
        )
        return False, selected
    if any(decision is None for _, decision in decisions):
        return None
    selected = (
        max(relevant, key=lambda sample: sample.value)
        if direction is MetricDirection.LOWER
        else min(relevant, key=lambda sample: sample.value)
    )
    return True, selected


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
        decision = _decide_samples(outcome, sample_list)
        if decision is None or outcome.metric is None or outcome.threshold is None:
            continue
        passed, selected = decision
        definition = definition_for(outcome.metric)
        comparison = "at most" if definition.direction is MetricDirection.LOWER else "at least"
        observed = {
            MeasurementRelation.EXACT: "",
            MeasurementRelation.AT_LEAST: "at least ",
            MeasurementRelation.AT_MOST: "at most ",
        }[selected.relation]
        verdicts.append(
            RuleVerification(
                rule=outcome.statement,
                rule_id=rules.rule_id(rules.EXPECTED, index),
                status="passed" if passed else "failed",
                reasoning=(
                    f"Worst proven sample was {observed}{selected.value:g} "
                    f"{definition.unit.value}; "
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
