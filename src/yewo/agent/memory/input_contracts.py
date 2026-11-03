"""Build reusable input-shape contracts without retaining literal text."""

import re

from yewo.agent.memory.records import InputContract, InputValueShape


def build_literal_input_contract(value: str) -> InputContract:
    """Describe a literal value by shape and separator pattern only."""
    if re.fullmatch(r"[+-]?\d+", value):
        return InputContract(value_shape=InputValueShape.INTEGER)
    if re.fullmatch(r"[+-]?(?:\d+\.\d*|\d*\.\d+)", value):
        return InputContract(value_shape=InputValueShape.DECIMAL)
    if re.fullmatch(r"\d+(?:[^\w\s]+\d+)+", value):
        return InputContract(
            value_shape=InputValueShape.FORMATTED,
            format_pattern=re.sub(r"\d", "#", value),
        )
    return InputContract(value_shape=InputValueShape.TEXT)
