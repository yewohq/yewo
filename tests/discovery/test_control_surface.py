"""Discovery reports the controls one perceived frame offered, and what it could not see.

The surface is built from vision. These cases hold it to what vision can
honestly say: a role the detector package never declared is not assumed, and a
control that was never detected is absent rather than reported inert.
"""

from perception_fakes import detected, frame, read

from yewo.discovery.control_surface import control_text_digest
from yewo.agent.perception.keyboard import KEYBOARD_KEY_ROLE, KEYBOARD_REGION_ROLE
from yewo.discovery.control_surface import (
    ControlInteraction,
    ControlLabelSource,
    build_control_surface,
    select_control_at,
)

VIEWPORT = (1080, 2400)


def _form():
    """A two-field form: labels read by OCR, fields declared as inputs."""
    return frame(
        elements=(
            detected((40, 300, 300, 360), name="Text", text="Name"),
            detected((40, 380, 1040, 470), name="View", roles={"text_input"}),
            detected((40, 900, 300, 960), name="Text", text="Trainer"),
            detected((40, 980, 1040, 1070), name="View", roles={"text_input"}),
            detected((940, 2200, 1040, 2300), name="ImageView"),
            detected((40, 1200, 1040, 1300), name="View", text="Save"),
        ),
        text_lines=(
            read((40, 300, 300, 360), "Name"),
            read((40, 900, 300, 960), "Trainer"),
            read((40, 1200, 1040, 1300), "Save"),
        ),
    )


def test_a_declared_input_role_is_reported_as_editable_not_merely_actionable():
    surface = build_control_surface(_form(), VIEWPORT)

    editable = surface.editable_controls

    assert len(editable) == 2
    assert {control.interaction for control in editable} == {ControlInteraction.EDIT}


def test_a_detector_declaring_no_input_role_offers_no_editable_control():
    """The fail-closed reading. Today's weights declare none, so none is claimed."""
    undeclared = frame(
        elements=(detected((40, 380, 1040, 470), name="View"),),
        text_lines=(read((40, 300, 300, 360), "Name"),),
    )

    surface = build_control_surface(undeclared, VIEWPORT)

    assert surface.editable_controls == ()
    assert surface.controls[0].interaction is ControlInteraction.ACTIVATE


def test_a_field_OCR_could_not_name_takes_the_line_above_it():
    surface = build_control_surface(_form(), VIEWPORT)

    first, second = surface.editable_controls

    assert first.label_source is ControlLabelSource.ASSOCIATED
    assert first.text_digest == control_text_digest("Name")
    assert second.text_digest == control_text_digest("Trainer")


def test_a_line_too_far_away_does_not_name_a_field():
    far = frame(
        elements=(detected((40, 380, 1040, 470), name="View", roles={"text_input"}),),
        text_lines=(read((40, 10, 300, 70), "Name"),),
    )

    surface = build_control_surface(far, VIEWPORT)

    field = surface.editable_controls[0]
    assert field.label_source is ControlLabelSource.ABSENT
    assert field.text_digest == ""


def test_a_control_no_text_names_is_evidence_rather_than_silence():
    surface = build_control_surface(_form(), VIEWPORT)

    unnamed = surface.unnamed_controls

    assert len(unnamed) == 1
    assert unnamed[0].role == "imageview"
    assert unnamed[0].interaction is ControlInteraction.ACTIVATE
    assert unnamed[0] in surface.unreacquirable_controls


def test_an_icon_near_a_line_is_reacquirable_by_that_anchor():
    anchored = frame(
        elements=(detected((940, 2200, 1040, 2300), name="ImageView"),),
        text_lines=(read((900, 2130, 1060, 2180), "Add"),),
    )

    surface = build_control_surface(anchored, VIEWPORT)

    icon = surface.controls[0]
    assert icon.label_source is ControlLabelSource.ABSENT
    assert icon.anchor_above_digest == control_text_digest("Add")
    assert icon.is_reacquirable


def test_fused_text_names_a_control_in_its_own_right():
    surface = build_control_surface(_form(), VIEWPORT)

    save = next(
        control
        for control in surface.controls
        if control.text_digest == control_text_digest("Save")
    )

    assert save.label_source is ControlLabelSource.OWN


def test_reading_matter_and_system_chrome_are_not_controls():
    chrome = frame(
        elements=(
            detected((0, 0, 1080, 60), name="StatusBar"),
            detected((40, 300, 300, 360), name="Text", text="Name"),
            detected((40, 700, 1040, 702), name="Line"),
            detected((40, 380, 400, 470), name="View", text="Go"),
        )
    )

    surface = build_control_surface(chrome, VIEWPORT)

    assert len(surface.controls) == 1
    assert surface.controls[0].text_digest == control_text_digest("Go")


def test_the_keyboard_is_never_offered_as_a_frontier():
    """A key reaches no capability the frontier does not already hold."""
    typing = frame(
        elements=(
            detected((0, 1800, 1080, 2400), name="Keyboard", roles={KEYBOARD_REGION_ROLE}),
            detected((100, 1900, 200, 2000), name="Keys", roles={KEYBOARD_KEY_ROLE}, text="a"),
            detected((40, 380, 400, 470), name="View", text="Go"),
        )
    )

    surface = build_control_surface(typing, VIEWPORT)

    assert len(surface.controls) == 1
    assert surface.controls[0].text_digest == control_text_digest("Go")


def test_the_backdrop_a_detector_reports_is_scenery_not_a_control():
    scenery = frame(
        elements=(
            detected((0, 0, 1080, 2400), name="View"),
            detected((40, 380, 400, 470), name="View", text="Go"),
        )
    )

    surface = build_control_surface(scenery, VIEWPORT)

    assert len(surface.controls) == 1
    assert surface.controls[0].text_digest == control_text_digest("Go")


def test_a_control_meeting_the_viewport_edge_declares_it_may_continue_off_screen():
    edge = frame(
        elements=(
            detected((40, 2340, 1040, 2400), name="View", text="More"),
            detected((40, 380, 400, 470), name="View", text="Go"),
        )
    )

    surface = build_control_surface(edge, VIEWPORT)

    partial, whole = surface.controls
    assert partial.is_clipped_by_viewport
    assert not whole.is_clipped_by_viewport


def test_text_no_detection_claimed_is_still_carried_as_evidence():
    orphaned = frame(
        elements=(detected((40, 380, 400, 470), name="View", text="Go"),),
        text_lines=(read((40, 100, 600, 160), "Schedule"),),
    )

    surface = build_control_surface(orphaned, VIEWPORT)

    assert surface.unassociated_text_digests == (control_text_digest("Schedule"),)


def test_a_control_carries_the_point_an_action_targets():
    surface = build_control_surface(_form(), VIEWPORT)

    assert surface.editable_controls[0].center == (540, 425)


def test_detection_confidence_travels_with_the_control():
    weak = frame(elements=(detected((40, 380, 400, 470), name="View", confidence=0.31),))

    surface = build_control_surface(weak, VIEWPORT)

    assert surface.controls[0].detection_confidence == 0.31


def test_the_surface_keeps_no_label_text_and_no_app_vocabulary():
    surface = build_control_surface(_form(), VIEWPORT)

    rendered = repr(surface)
    for label in ("Name", "Trainer", "Save"):
        assert label not in rendered


def test_a_frame_with_nothing_detected_is_an_empty_surface_not_a_failure():
    surface = build_control_surface(frame(), VIEWPORT)

    assert surface.controls == ()
    assert surface.unreacquirable_controls == ()


def test_a_detection_that_names_no_class_is_still_reported_as_a_control():
    """A nameless class is a package defect, not a reason to drop the target."""
    nameless = frame(
        elements=(detected((40, 380, 400, 470), name="", kind="", text="Go"),)
    )

    surface = build_control_surface(nameless, VIEWPORT)

    assert surface.controls[0].role == "control"
    assert surface.controls[0].interaction is ControlInteraction.ACTIVATE


def test_the_control_under_a_point_is_the_smallest_one_containing_it():
    """Detections nest. A point on the button belongs to the button, not the card."""
    nested = frame(
        elements=(
            detected((20, 100, 900, 600), name="View", text="Card"),
            detected((40, 200, 300, 280), name="View", text="Open"),
        )
    )

    surface = build_control_surface(nested, VIEWPORT)

    chosen = select_control_at(surface, (100, 240))
    assert chosen is not None
    assert chosen.text_digest == control_text_digest("Open")


def test_a_point_on_no_control_selects_nothing():
    surface = build_control_surface(
        frame(elements=(detected((40, 380, 400, 470), name="View", text="Go"),)),
        VIEWPORT,
    )

    assert select_control_at(surface, (900, 2000)) is None
