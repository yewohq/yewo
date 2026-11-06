"""Accepted-value shapes a focused field reports to the planner."""

from yewo.devices.interaction_records import TextInputConstraint


def test_decimal_field_accepts_decimal_value() -> None:
    constraint = TextInputConstraint(is_numeric=True, allows_decimal=True)

    assert constraint.accepts("1.5")
    assert "a decimal number" in constraint.describe()


def test_integer_field_rejects_decimal_point() -> None:
    constraint = TextInputConstraint(is_numeric=True)

    assert not constraint.accepts("1.5")
    assert "whole digits only" in constraint.describe()
    assert "decimal point is not accepted" in constraint.describe()
