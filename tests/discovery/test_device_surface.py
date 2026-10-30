"""The device discovery reads is the device a run actually has.

A fake carrying a convenience attribute no real `DeviceControl` has makes a
green suite compatible with a run that cannot take one step. That happened: the
runtime read `device_control.commands`, every fake offered it, and the first
cold run ended `DISCOVERY_BLOCKED` before taking a screenshot.
"""

import ast
import inspect
import textwrap

from discovery_device_fakes import DiscoveryDevice

from yewo.agent.device.control import DeviceControl

READ_FROM_THE_DEVICE = (
    "driver",
    "app_id",
    "screen_width",
    "screen_height",
    "is_keyboard_open",
    "get_screenshot",
    "tap",
)
"""Every member discovery reads off a started device session."""


def _members_of(cls) -> set[str]:
    """Class members plus the instance attributes its `__init__` assigns.

    Read from the syntax tree rather than from an instance, because
    constructing a real `DeviceControl` needs a device.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(cls.__init__)))
    assigned = {
        target.attr
        for node in ast.walk(tree)
        for target in (
            [node.target] if isinstance(node, ast.AnnAssign) else getattr(node, "targets", [])
        )
        if isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    }
    return set(dir(cls)) | assigned


def test_every_member_discovery_reads_exists_on_the_real_device():
    missing = sorted(set(READ_FROM_THE_DEVICE) - _members_of(DeviceControl))

    assert missing == [], f"discovery reads what DeviceControl does not have: {missing}"


def test_the_device_fake_offers_nothing_the_real_device_lacks():
    """A fake wider than the real thing is how the cold run broke."""
    invented = sorted(
        name
        for name in _members_of(DiscoveryDevice)
        if not name.startswith("_")
        and name not in _members_of(DeviceControl)
        and name != "dispatched"
    )

    assert invented == [], f"the fake invents members no device has: {invented}"


def test_a_command_surface_is_not_one_of_them():
    """`DeviceControl` exposes a driver. Callers wrap it themselves."""
    assert "commands" not in _members_of(DeviceControl)
