"""Asking the platform what is at a point the screenshot could not explain.

A screenshot that missed a control cannot be asked again about the same
pixels. The driver can: it knows which element holds focus, and that answer
comes from the app rather than from the picture of it. One tap buys it.

The tap is still a tap. A control nobody detected could be a button. Pressing
a button cannot be taken back. A caller must therefore already need an editable
field at this point before it probes for one.
"""

from yewo.agent.device.focus import FocusProbe, bounds_of


def focus_at(control, x: int, y: int) -> FocusProbe:
    """Tap one screenshot pixel and report which element took focus.

    Nothing is typed. The focus query uses the driver's coordinate space. The
    control's bounds come back in the screenshot's space. A caller never has
    to know the two differ.
    """
    before = control.focus.rect()
    before_bounds = bounds_of(before)
    keyboard_before = control.is_keyboard_open()
    point = control.driver_point(x, y)
    control.tap(x, y)
    control.focus.wait_for(*point, before=before)
    keyboard_after = control.is_keyboard_open()
    reading = control.focus.reading()
    focus_changed = reading.bounds is not None and reading.bounds != before_bounds
    input_method_opened = keyboard_before is False and keyboard_after is True
    is_focused = control.focus.contains(*point) or (
        input_method_opened and focus_changed and reading.bounds is not None
    )
    return FocusProbe(
        is_focused=is_focused,
        focus_changed=focus_changed,
        bounds=control.perception_box(reading.bounds) if is_focused else None,
    )
