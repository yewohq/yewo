"""Accepted-value shapes a focused field reports to the planner."""

from yewo.devices.interaction_records import TextInputConstraint


def test_a_decimal_field_reports_that_a_decimal_value_is_accepted() -> None:
    constraint = TextInputConstraint(is_numeric=True, allows_decimal=True)

    assert constraint.accepts("1.5")
    assert "a decimal number" in constraint.describe()
    assert "not accepted" not in constraint.describe().split("sign")[0]


def test_an_integer_field_names_the_decimal_point_it_refuses() -> None:
    constraint = TextInputConstraint(is_numeric=True)

    assert not constraint.accepts("1.5")
    assert "whole digits only" in constraint.describe()
    assert "decimal point is not accepted" in constraint.describe()
