"""Reaching a screen is not evidence that its capability can be exercised."""

from perception_fakes import detected, frame, read

from yewo.discovery.capability_surface import CapabilitySurfacePredicate
from yewo.discovery.control_surface import build_control_surface

VIEWPORT = (1080, 2400)


def _entry_form(save_name: str = "Save"):
    """A field to type into and a control to commit with."""
    return frame(
        elements=(
            detected((40, 300, 300, 360), name="Text", text="Name"),
            detected((40, 380, 1040, 470), name="View", roles={"text_input"}),
            detected((40, 560, 1040, 660), name="View", text=save_name),
        ),
        text_lines=(
            read((40, 300, 300, 360), "Name"),
            read((40, 560, 1040, 660), save_name),
        ),
    )


def _containing_screen():
    """The screen a capability lives on, offering no way to enter anything."""
    return frame(
        elements=(
            detected((40, 100, 600, 160), name="Text", text="Schedule"),
            detected((40, 300, 1040, 400), name="View", text="Add"),
        ),
        text_lines=(
            read((40, 100, 600, 160), "Schedule"),
            read((40, 300, 1040, 400), "Add"),
        ),
    )


def _anonymous_commit():
    """The same form, but nothing named the control that commits it."""
    return frame(
        elements=(
            detected((40, 300, 300, 360), name="Text", text="Name"),
            detected((40, 380, 1040, 470), name="View", roles={"text_input"}),
            detected((40, 1800, 140, 1900), name="ImageView"),
        ),
        text_lines=(read((40, 300, 300, 360), "Name"),),
    )


def _entry_predicate() -> CapabilitySurfacePredicate:
    return CapabilitySurfacePredicate(
        predicate_id="entry-form-surface-v1",
        minimum_editable_controls=1,
        minimum_activatable_controls=1,
    )


def test_a_form_offering_its_controls_satisfies_the_capability():
    surface = build_control_surface(_entry_form(), VIEWPORT)

    assert _entry_predicate().is_satisfied_by(surface)


def test_reaching_the_containing_screen_alone_does_not_satisfy_it():
    surface = build_control_surface(_containing_screen(), VIEWPORT)

    predicate = _entry_predicate()

    assert not predicate.is_satisfied_by(surface)
    assert "0 editable control(s)" in str(predicate.shortfalls(surface)[0])


def test_a_control_nothing_can_name_again_blocks_the_capability():
    surface = build_control_surface(_anonymous_commit(), VIEWPORT)

    predicate = _entry_predicate()

    assert not predicate.is_satisfied_by(surface)
    assert "no name and no anchor" in str(predicate.shortfalls(surface)[-1])


def test_a_capability_may_accept_an_unnameable_control_when_it_declares_so():
    surface = build_control_surface(_anonymous_commit(), VIEWPORT)

    tolerant = CapabilitySurfacePredicate(
        predicate_id="entry-form-surface-v1",
        minimum_editable_controls=1,
        minimum_activatable_controls=1,
        requires_reacquirable_controls=False,
    )

    assert tolerant.is_satisfied_by(surface)


def test_every_shortfall_is_reported_rather_than_the_first():
    surface = build_control_surface(_containing_screen(), VIEWPORT)

    demanding = CapabilitySurfacePredicate(
        predicate_id="entry-form-surface-v1",
        minimum_editable_controls=2,
        minimum_activatable_controls=5,
    )

    assert len(demanding.shortfalls(surface)) == 2


def test_a_predicate_demanding_nothing_is_satisfied_by_any_surface():
    surface = build_control_surface(_containing_screen(), VIEWPORT)

    assert CapabilitySurfacePredicate(predicate_id="any-v1").is_satisfied_by(surface)


def test_a_screen_whose_detector_declares_no_input_role_falls_short_rather_than_guessing():
    """Discovery loses coverage, never safety, when the vocabulary is silent."""
    undeclared = frame(
        elements=(
            detected((40, 300, 300, 360), name="Text", text="Name"),
            detected((40, 380, 1040, 470), name="View"),
            detected((40, 560, 1040, 660), name="View", text="Save"),
        ),
        text_lines=(read((40, 560, 1040, 660), "Save"),),
    )

    surface = build_control_surface(undeclared, VIEWPORT)

    assert not _entry_predicate().is_satisfied_by(surface)
    assert "0 editable control(s)" in str(_entry_predicate().shortfalls(surface)[0])
