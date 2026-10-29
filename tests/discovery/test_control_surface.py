"""Discovery reports the controls a capability needs, including the unnamed ones."""

from yewo.discovery.control_semantics import (
    ControlInteraction,
    ControlLabelSource,
    build_control_surface,
    control_text_digest,
)

VIEWPORT = (1080, 2400)

FORM = """<hierarchy>
  <node class="android.widget.ScrollView" scrollable="true" bounds="[0,0][1080,2400]">
    <node class="android.widget.TextView" text="Name" bounds="[40,300][300,360]" />
    <node class="android.widget.EditText" bounds="[40,380][1040,470]" focusable="true" />
    <node class="android.widget.TextView" text="Trainer" bounds="[40,900][300,960]" />
    <node class="android.widget.EditText" bounds="[40,980][1040,1070]" focusable="true" />
    <node class="android.widget.ImageButton" bounds="[940,2200][1040,2300]" clickable="true" />
    <node class="android.widget.Button" text="Save" bounds="[40,2500][1040,2600]" clickable="true" />
  </node>
</hierarchy>"""


def _by_role(surface, role):
    return tuple(control for control in surface.controls if control.role == role)


def test_an_editable_control_is_reported_as_editable_not_merely_actionable():
    surface = build_control_surface(FORM, VIEWPORT)

    editable = surface.editable_controls

    assert len(editable) == 2
    assert {control.interaction for control in editable} == {ControlInteraction.EDIT}


def test_a_field_with_no_name_of_its_own_takes_the_label_above_it():
    surface = build_control_surface(FORM, VIEWPORT)

    first, second = surface.editable_controls

    assert first.label_source is ControlLabelSource.ASSOCIATED
    assert first.text_digest == control_text_digest("Name")
    assert second.text_digest == control_text_digest("Trainer")


def test_a_label_too_far_away_does_not_name_a_field():
    far = FORM.replace('bounds="[40,300][300,360]"', 'bounds="[40,10][300,70]"')

    surface = build_control_surface(far, VIEWPORT)

    first = surface.editable_controls[0]
    assert first.label_source is ControlLabelSource.ABSENT
    assert first.text_digest == ""


def test_a_control_no_metadata_names_is_evidence_rather_than_silence():
    surface = build_control_surface(FORM, VIEWPORT)

    unnamed = surface.unnamed_controls

    assert len(unnamed) == 1
    assert unnamed[0].role == "imagebutton"
    assert unnamed[0].interaction is ControlInteraction.ACTIVATE
    assert unnamed[0] in surface.unreacquirable_controls


def test_a_resource_identity_names_an_icon_only_control():
    identified = FORM.replace(
        '<node class="android.widget.ImageButton" bounds="[940,2200][1040,2300]" clickable="true" />',
        '<node class="android.widget.ImageButton" resource-id="org.example:id/fab" '
        'bounds="[940,2200][1040,2300]" clickable="true" />',
    )

    surface = build_control_surface(identified, VIEWPORT)

    icon = _by_role(surface, "imagebutton")[0]
    assert icon.label_source is ControlLabelSource.OWN
    assert icon.is_reacquirable
    assert surface.unnamed_controls == ()


def test_a_control_beyond_the_viewport_declares_that_it_needs_scrolling():
    surface = build_control_surface(FORM, VIEWPORT)

    save = _by_role(surface, "button")[0]
    first_field = surface.editable_controls[0]

    assert save.requires_scroll
    assert not first_field.requires_scroll


def test_a_control_inside_a_scrolling_container_says_so():
    surface = build_control_surface(FORM, VIEWPORT)

    assert all(control.within_scrollable for control in surface.controls)


def test_a_control_outside_any_scrolling_container_says_so():
    flat = """<hierarchy>
      <node class="android.widget.FrameLayout" bounds="[0,0][1080,2400]">
        <node class="android.widget.Button" text="Go" bounds="[40,100][300,200]" clickable="true" />
      </node>
    </hierarchy>"""

    surface = build_control_surface(flat, VIEWPORT)

    assert surface.controls[0].within_scrollable is False


def test_a_disabled_control_offers_nothing_and_is_left_out():
    disabled = FORM.replace(
        '<node class="android.widget.Button" text="Save"',
        '<node class="android.widget.Button" enabled="false" text="Save"',
    )

    surface = build_control_surface(disabled, VIEWPORT)

    assert _by_role(surface, "button") == ()


def test_focus_behavior_is_carried_with_the_control():
    surface = build_control_surface(FORM, VIEWPORT)

    assert all(control.focusable for control in surface.editable_controls)
    assert not _by_role(surface, "button")[0].focusable


def test_the_surface_keeps_no_label_text_and_no_app_vocabulary():
    surface = build_control_surface(FORM, VIEWPORT)

    rendered = repr(surface)
    for label in ("Name", "Trainer", "Save"):
        assert label not in rendered


def test_an_ios_text_field_is_editable_by_its_own_role():
    ios = """<hierarchy>
      <node type="XCUIElementTypeOther" x="0" y="0" width="390" height="844">
        <node type="XCUIElementTypeStaticText" name="Email" x="20" y="100"
              width="120" height="20" />
        <node type="XCUIElementTypeTextField" x="20" y="130" width="350" height="40"
              hittable="true" />
      </node>
    </hierarchy>"""

    surface = build_control_surface(ios, (390, 844))

    field = surface.editable_controls[0]
    assert field.interaction is ControlInteraction.EDIT
    assert field.text_digest == control_text_digest("Email")
