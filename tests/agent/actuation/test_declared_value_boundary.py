"""The literal behind a declared value must not leave the device boundary."""

import json

from yewo.agent.actuation import (
    ActionExecutor,
    ActionKind,
    DeclaredValueReference,
)

from agent.actuation.test_executor import RecordingDevice, authorization


SECRET = "swordfish-7319"


def test_a_declared_literal_reaches_the_device_and_nothing_else() -> None:
    # Yewo logs model traffic verbatim and keeps the literal out at the input boundary instead.
    # That holds only while the resolved value stays inside dispatch: a receipt, an outcome
    # reason or a planner-visible field carrying it would put it in the ledger by another route.
    device = RecordingDevice()
    executor = ActionExecutor(device, declared_value_resolver=lambda name: SECRET)
    action = authorization(
        ActionKind.TYPE,
        text=DeclaredValueReference(value_name="password"),
    )

    receipt = executor.dispatch(action)

    assert ("type_focused_text", (SECRET,), {}) in device.calls
    assert SECRET not in json.dumps(receipt.model_dump(mode="json"))
    assert SECRET not in json.dumps(action.model_dump(mode="json"))


def test_the_authorized_action_names_the_value_rather_than_holding_it() -> None:
    action = authorization(
        ActionKind.TYPE,
        text=DeclaredValueReference(value_name="password"),
    )

    assert "password" in json.dumps(action.model_dump(mode="json"))
